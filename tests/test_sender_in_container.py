#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：在发送端容器中运行，用于测试发送端与接收端之间的连接

此脚本应该在data-sender容器中运行，因为只有发送端容器才挂载了数据集。
它会启动一个模拟的接收器服务，然后使用EEGDataSender发送数据，测试连接是否正常。

使用方法：
    在data-sender容器中执行：
    python -m tests.test_sender_in_container
"""

import os
import sys
import time
import socket
import threading
import json
import argparse
import numpy as np
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入发送器模块
from sender.sender import EEGDataSender
from sender.eeg_reader import EEGReader, EEGDataInfo


# 模拟接收器服务器
class MockReceiver:
    def __init__(self, host='127.0.0.1', metadata_port=9000, data_port=9001):
        self.host = host
        self.metadata_port = metadata_port
        self.data_port = data_port
        self.metadata_socket = None
        self.data_socket = None
        self.is_running = False
        self.metadata_thread = None
        self.data_thread = None
        self.received_metadata = None
        self.received_data_packets = []
    
    def start(self):
        """启动模拟接收器服务"""
        self.is_running = True
        
        # 启动元数据服务器
        self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.metadata_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.metadata_socket.bind((self.host, self.metadata_port))
        self.metadata_socket.listen(5)
        self.metadata_thread = threading.Thread(target=self._metadata_server)
        self.metadata_thread.daemon = True
        self.metadata_thread.start()
        
        # 启动数据服务器
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_socket.bind((self.host, self.data_port))
        self.data_socket.listen(5)
        self.data_thread = threading.Thread(target=self._data_server)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        print(f"模拟接收器启动: {self.host}:{self.metadata_port}/{self.data_port}")
        print(f"接收器状态: is_running={self.is_running}, 线程状态: metadata_thread={self.metadata_thread.is_alive()}, data_thread={self.data_thread.is_alive()}")
        print(f"接收器套接字: metadata_socket={self.metadata_socket.fileno()}, data_socket={self.data_socket.fileno()}")
        print("接收器准备就绪，等待连接...")
    
    def _metadata_server(self):
        """元数据服务器线程"""
        try:
            print("元数据服务器线程启动，等待连接...")
            while self.is_running:
                try:
                    self.metadata_socket.settimeout(1.0)
                    client, addr = self.metadata_socket.accept()
                    print(f"元数据服务接收到连接: {addr}")
                    client.settimeout(10.0)  # 设置较长的超时时间
                    
                    # 接收数据长度
                    length_bytes = client.recv(4)
                    if not length_bytes or len(length_bytes) < 4:
                        print(f"接收到不完整的长度字节: {length_bytes.hex() if length_bytes else '无数据'}")
                        client.close()
                        continue
                    
                    print(f"接收到长度字节: {length_bytes.hex()}")
                    data_length = int.from_bytes(length_bytes, byteorder='big')
                    print(f"解析的数据长度: {data_length}")
                    
                    if data_length <= 0 or data_length > 10000000:  # 设置合理的上限
                        print(f"数据长度异常: {data_length}，可能是解析错误")
                        client.close()
                        continue
                    
                    # 接收数据内容
                    data = b''
                    bytes_received = 0
                    start_time = time.time()
                    
                    while bytes_received < data_length:
                        try:
                            chunk = client.recv(min(4096, data_length - bytes_received))
                            if not chunk:
                                print("接收数据中断，连接可能已关闭")
                                break
                            
                            data += chunk
                            bytes_received = len(data)
                            print(f"已接收数据: {bytes_received}/{data_length} 字节 ({bytes_received/data_length*100:.1f}%)")
                            
                            # 设置超时保护
                            if time.time() - start_time > 30:  # 30秒超时
                                print("接收数据超时，中断接收")
                                break
                        except socket.timeout:
                            print("接收数据超时")
                            break
                        except Exception as e:
                            print(f"接收数据时发生错误: {e}")
                            break
                    
                    if bytes_received == data_length:
                        # 解析元数据
                        try:
                            decoded_data = data.decode('utf-8')
                            print(f"解码后的数据: {decoded_data[:100]}...")
                            self.received_metadata = json.loads(decoded_data)
                            print(f"✅ 成功接收并解析元数据: {len(data)} 字节")
                            print(f"元数据内容: {json.dumps(self.received_metadata, indent=2, ensure_ascii=False)[:500]}...")
                        except UnicodeDecodeError as e:
                            print(f"Unicode解码错误: {e}")
                            print(f"接收到的原始数据(十六进制): {data[:50].hex()}...")
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                            print(f"接收到的原始数据: {data[:100]}...")
                    else:
                        print(f"❌ 数据接收不完整: 接收了 {bytes_received}/{data_length} 字节")
                    
                    client.close()
                    print("客户端连接已关闭")
                except socket.timeout:
                    # 正常的超时，继续循环
                    continue
                except Exception as e:
                    print(f"元数据服务器错误: {e}")
                    break
        finally:
            if self.metadata_socket:
                self.metadata_socket.close()
    
    def _data_server(self):
        """数据服务器线程"""
        try:
            while self.is_running:
                try:
                    self.data_socket.settimeout(1.0)
                    client, addr = self.data_socket.accept()
                    client.settimeout(60)  # 设置较长的超时时间
                    print(f"数据服务接收到连接: {addr}")
                    
                    # 处理数据流
                    self._handle_data_stream(client)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"数据服务器错误: {e}")
                    break
        finally:
            if self.data_socket:
                self.data_socket.close()
    
    def _handle_data_stream(self, client):
        """处理数据流连接"""
        try:
            while self.is_running:
                # 接收数据长度
                length_bytes = client.recv(4)
                if not length_bytes:
                    print("数据流连接关闭")
                    break
                
                print(f"数据流接收到长度字节: {length_bytes.hex()}")
                data_length = int.from_bytes(length_bytes, byteorder='big')
                print(f"数据流解析的数据长度: {data_length}")
                
                # 接收数据内容
                data = b''
                while len(data) < data_length:
                    chunk = client.recv(min(4096, data_length - len(data)))
                    if not chunk:
                        print("数据流接收中断，连接可能已关闭")
                        break
                    data += chunk
                    print(f"数据流已接收: {len(data)}/{data_length} 字节")
                
                if len(data) == data_length:
                    # 解析数据包
                    try:
                        decoded_data = data.decode('utf-8')
                        print(f"数据流解码后的数据: {decoded_data[:100]}...")
                        packet = json.loads(decoded_data)
                        packet_type = packet.get("type")
                        
                        if packet_type == "START":
                            print(f"接收到开始标记: {packet}")
                        elif packet_type == "DATA":
                            self.received_data_packets.append(packet)
                            print(f"接收到数据包: 序列号={packet.get('sequence_num')}, 样本数={packet.get('n_samples')}")
                        elif packet_type == "END":
                            print(f"接收到结束标记: {packet}")
                            break
                        else:
                            print(f"未知的数据包类型: {packet_type}")
                    except json.JSONDecodeError as e:
                        print(f"数据流JSON解析错误: {e}")
                        print(f"接收到的原始数据: {data[:100]}...")
                else:
                    print(f"数据流接收不完整: 接收了 {len(data)}/{data_length} 字节")
        except Exception as e:
            print(f"处理数据流错误: {e}")
        finally:
            client.close()
    
    def stop(self):
        """停止模拟接收器服务"""
        self.is_running = False
        time.sleep(1.5)  # 等待线程结束
        
        if self.metadata_socket:
            self.metadata_socket.close()
        
        if self.data_socket:
            self.data_socket.close()
        
        print("模拟接收器已停止")


def test_sender_with_mock_receiver(subject_id="01", task_name="rest", dataset_root=None):
    """测试发送器与模拟接收器的连接
    
    Args:
        subject_id: 受试者ID
        task_name: 任务名称
        dataset_root: 数据集根目录，如果为None则使用默认值
    """
    # 确定数据集根目录
    if dataset_root is None:
        # 在Docker环境中，数据集通常挂载在/app/data目录下
        if os.path.exists('/.dockerenv'):
            dataset_root = "/app/data/ds005779"
        else:
            # 本地开发环境，使用相对路径
            dataset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    print(f"\n===== 测试发送器与模拟接收器的连接 =====\n")
    print(f"使用数据集路径: {dataset_root}")
    print(f"受试者ID: {subject_id}")
    print(f"任务名称: {task_name}\n")
    print(f"Python版本: {sys.version}")
    print(f"系统平台: {sys.platform}")
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_root):
        print(f"错误: 数据集路径 {dataset_root} 不存在!")
        print("请确保数据集已正确挂载到容器中，或者提供正确的数据集路径。")
        return False
    
    # 启动模拟接收器
    print("\n----- 启动模拟接收器 -----\n")
    receiver = MockReceiver(host='127.0.0.1', metadata_port=9000, data_port=9001)
    receiver.start()
    print("模拟接收器启动完成，等待连接...\n")
    
    try:
        # 创建发送器实例，指定接收器地址为本地地址
        sender = EEGDataSender(
            host='0.0.0.0',  # 监听所有网络接口
            metadata_port=9100,  # 使用不同的端口避免冲突
            data_port=9101,
            dataset_root=dataset_root,
            receiver_host='127.0.0.1',  # 直接连接到本地模拟接收器
            receiver_metadata_port=9000,
            receiver_data_port=9001
        )
        
        # 设置Socket
        sender.setup_sockets()
        
        # 发送元数据
        print("\n----- 发送元数据 -----\n")
        payload_id = sender.send_metadata(subject_id, task_name)
        
        if payload_id:
            print(f"\n✅ 元数据发送成功，payload_id: {payload_id}\n")
            
            # 等待接收器接收元数据
            time.sleep(1)
            
            if receiver.received_metadata:
                print("✅ 接收器成功接收元数据")
            else:
                print("❌ 接收器未接收到元数据")
            
            # 开始流式发送数据
            print("\n----- 开始流式发送数据 -----\n")
            success = sender.start_streaming_data(
                subject_id=subject_id,
                task_name=task_name,
                payload_id=payload_id,
                speed_factor=10.0,  # 加速发送
                chunk_samples=100    # 减小数据块大小
            )
            
            if success:
                print("✅ 数据流发送已启动")
                
                # 等待数据发送完成
                max_wait_time = 30  # 最多等待30秒
                wait_interval = 1    # 每次检查间隔1秒
                total_waited = 0
                
                while sender.is_running and total_waited < max_wait_time:
                    time.sleep(wait_interval)
                    total_waited += wait_interval
                    print(f"等待数据发送完成... {total_waited}秒")
                
                # 停止发送
                sender.is_running = False
                if sender.data_thread and sender.data_thread.is_alive():
                    sender.data_thread.join(timeout=2)
                
                # 检查接收结果
                if receiver.received_data_packets:
                    print(f"\n✅ 接收器成功接收数据包: {len(receiver.received_data_packets)} 个")
                    return True
                else:
                    print("\n❌ 接收器未接收到任何数据包")
                    return False
            else:
                print("\n❌ 数据流发送启动失败")
                return False
        else:
            print("\n❌ 元数据发送失败")
            return False
    
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        return False
    
    finally:
        # 停止接收器
        receiver.stop()


def main():
    """主函数，解析命令行参数并执行测试"""
    parser = argparse.ArgumentParser(description="测试EEG数据发送器与接收器的连接")
    parser.add_argument("--subject", "-s", type=str, default="01", help="受试者ID")
    parser.add_argument("--task", "-t", type=str, default="rest", help="任务名称")
    parser.add_argument("--dataset", "-d", type=str, help="数据集根目录路径")
    
    args = parser.parse_args()
    
    # 执行测试
    success = test_sender_with_mock_receiver(
        subject_id=args.subject,
        task_name=args.task,
        dataset_root=args.dataset
    )
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()