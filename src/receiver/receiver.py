import os
import time
import socket
import json
import struct
import threading
import numpy as np
import logging
from datetime import datetime
import sys
import uuid

# 设置日志
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EEGReceiver")
logger.setLevel(logging.WARNING)


class DataSource:
    def __init__(self, name):
        self.name = name
        self.source_id = str(uuid.uuid4())[:8]
        self.metadata_port = None
        self.data_port = None
        self.metadata_server = None
        self.data_server = None
        self.active = True
        self.last_active = time.time()

    def close_servers(self):
        if self.metadata_server:
            self.metadata_server.close()
        if self.data_server:
            self.data_server.close()

class EEGReceiver:
    def __init__(self, host='0.0.0.0', port=9000):
        self.host = host
        self.port = port
        self.data_sources = {}
        self.data_buffers = {}
        self.metadata_buffers = {}
        self.data_callback = None
        self.running = False
        self.is_running = False
        self.base_metadata_port = 9001  # 元数据端口起始值
        self.base_data_port = 9101      # 数据端口起始值
        self.port_counter = 0           # 端口计数器
        self.main_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.main_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.main_server.bind((self.host, self.port))
        self.main_server.listen(5)
        self.received_count = 0
        self.stats = {"packets_received": 0, "bytes_received": 0}
        self.save_dir = "received_data"
        os.makedirs(self.save_dir, exist_ok=True)


    def _main_receiver_thread(self):
        """主接收线程，负责管理数据源"""
        # 使用__init__中已创建的主服务器socket实例

        
        self.main_server.settimeout(5.0)  # 设置超时以便定期检查运行状态
        

        while self.is_running:
            try:
                # 接受新连接
                try:
                    client_socket, addr = self.main_server.accept()

                    # 接收请求长度头
                    request_length_data = client_socket.recv(4)
                    if not request_length_data:
                        continue
                    request_length = struct.unpack('!I', request_length_data)[0]
                    
                    # 接收完整请求
                    request_data = b''
                    while len(request_data) < request_length:
                        chunk = client_socket.recv(request_length - len(request_data))
                        if not chunk:
                            break
                        request_data += chunk
                    
                    # 解析请求并添加数据源
                    request = json.loads(request_data.decode('utf-8'))

                    data_source = DataSource(request.get('client_id', str(uuid.uuid4())[:8]))
                    data_source.subject_id = request.get('subject_id', 'unknown')
                    data_source.task_name = request.get('task_name', '')

                    source_id = self.add_data_source(data_source)
                    response = {
                        'type': 'CONNECTION_RESPONSE',
                        'status': 'success',
                        'source_id': source_id,
                        'metadata_port': data_source.metadata_port,
                        'data_port': data_source.data_port
                    }
                    response_data = json.dumps(response).encode('utf-8')

                    try:
                        client_socket.sendall(len(response_data).to_bytes(4, byteorder='big'))
                        client_socket.sendall(response_data)
                    except Exception as e:
                        logger.error(f"发送响应失败: {e}")
                  
                    # 发送响应给发送端
                    response = {

                        "status": "success",
                        "metadata_port": data_source.metadata_port,
                        "data_port": data_source.data_port
                    }
                    response_json = json.dumps(response).encode('utf-8')
                    client_socket.sendall(len(response_json).to_bytes(4, byteorder='big'))
                    client_socket.sendall(response_json)

                    
                    # 启动数据源处理线程
                    threading.Thread(
                        target=self._receive_data_source, 
                        args=(data_source,),
                        daemon=True
                    ).start()
                except socket.timeout:
                    pass  # 超时正常，继续执行
                
                # 检查数据源状态
                inactive_sources = []
                for source_id, source_info in list(self.data_sources.items()):
                    # 超过5分钟无活动视为不活跃
                    if time.time() - source_info["last_active"] > 300:

                        inactive_sources.append(source_id)

                # 清理不活跃的数据源
                for source_id in inactive_sources:
                    self._cleanup_data_source(source_id)

                # 每30秒检查一次
                time.sleep(30)

            except Exception as e:

                time.sleep(10)  # 出错后等待10秒再继续

    def set_data_callback(self, callback):
        """设置数据处理回调函数"""
        self.data_callback = callback


    def start_receiving(self):
        """启动数据接收处理"""
        if self.is_running:

            return

        self.is_running = True


        # 主线程负责接收新的数据源连接
        main_thread = threading.Thread(target=self._main_receiver_thread, daemon=True)
        main_thread.start()


    def add_data_source(self, source_config):
        """添加新的数据源配置并启动接收"""
        if not self.is_running:

            return False

        # 为数据源分配唯一ID和端口(从9001开始)
        source_id = str(uuid.uuid4())[:8]
        metadata_port = self.base_metadata_port + self.port_counter
        data_port = self.base_data_port + self.port_counter
        self.port_counter += 1

        # 确保端口不重复分配
        while any(ds["config"].metadata_port == metadata_port or 
                ds["config"].data_port == data_port 
                for ds in self.data_sources.values()):
            metadata_port += 1
            data_port += 1

        # 统一处理数据源配置
        if not isinstance(source_config, DataSource):
            # 如果是字典或对象，转换为DataSource对象
            client_id = source_config.get("client_id", str(uuid.uuid4())[:8]) if isinstance(source_config, dict) else getattr(source_config, "client_id", str(uuid.uuid4())[:8])
            source_config = DataSource(client_id)
            source_config.subject_id = source_config.get("subject_id", "unknown") if isinstance(source_config, dict) else getattr(source_config, "subject_id", "unknown")
            source_config.task_name = source_config.get("task_name", "") if isinstance(source_config, dict) else getattr(source_config, "task_name", "")
            
        # 设置端口和状态
        source_config.metadata_port = metadata_port
        source_config.data_port = data_port
        source_config.status = "starting"

        # 启动数据源接收线程
        source_thread = threading.Thread(
            target=self._receive_data_source, args=(source_config,), daemon=True
        )
        source_thread.start()

        # 保存数据源信息
        self.data_sources[source_id] = {
            "config": source_config,
            "thread": source_thread,
            "status": "running",
            "last_active": time.time(),
        }

        logger.info(
            f"已添加数据源 {source_id}，元数据端口={metadata_port}，数据端口={data_port}"
        )
        return source_id

    def _receive_data_source(self, source_config):
        """接收单个数据源的数据"""
        source_id = source_config.source_id
        metadata_port = source_config.metadata_port
        data_port = source_config.data_port

        logger.info(f"开始接收数据源 {source_id} 的数据")

        try:
            # 创建元数据服务器
            # 检查端口是否已被占用
            def is_port_available(port):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind((self.host, port))
                        return True
                    except OSError:
                        return False
            
            if not is_port_available(metadata_port):
                logger.error(f"元数据端口 {metadata_port} 已被占用")
                return
            if not is_port_available(data_port):
                logger.error(f"数据端口 {data_port} 已被占用")
                return
                
            metadata_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            metadata_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            metadata_server.bind((self.host, metadata_port))
            metadata_server.listen(1)
            logger.info(
                f"数据源 {source_id} 元数据服务器已启动: {self.host}:{metadata_port}"
            )

            # 创建数据服务器
            data_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            data_server.bind((self.host, data_port))
            data_server.listen(1)
            logger.info(f"数据源 {source_id} 数据服务器已启动: {self.host}:{data_port}")

            # 等待元数据连接
            logger.info(f"数据源 {source_id} 等待元数据连接...")
            metadata_socket, metadata_addr = metadata_server.accept()
            logger.info(f"数据源 {source_id} 接收到元数据连接: {metadata_addr}")

            # 接收元数据
            metadata_json = self._receive_metadata(metadata_socket)
            if not metadata_json:
                logger.error(f"数据源 {source_id} 接收到空元数据")
                return

            metadata = json.loads(metadata_json)
            subject_id = metadata.get("subject_id", f"unknown_{source_id}")
            logger.info(
                f"数据源 {source_id} 接收到元数据: 受试者={subject_id}, 采样率={metadata.get('sampling_rate')}"
            )

            # 更新数据源状态
            if source_id in self.data_sources:
                self.data_sources[source_id].status = "connected"
                self.data_sources[source_id].subject_id = subject_id
                self.data_sources[source_id].last_active = time.time()

            # 为数据源创建数据缓冲区
            if not hasattr(self, 'data_buffers'):
                self.data_buffers = {}
            if not hasattr(self, 'metadata_buffers'):
                self.metadata_buffers = {}
                
            self.data_buffers[source_id] = []
            self.metadata_buffers[source_id] = metadata.copy()
            
            # 向发送器返回成功响应
            response = {
                "status": "success",
                "message": "Metadata received",
                "metadata_port": source_config.metadata_port,
                "data_port": source_config.data_port
            }
            response_json = json.dumps(response)
            response_data = response_json.encode('utf-8')
            # 先发送4字节的长度信息（大端序）
            length_bytes = len(response_data).to_bytes(4, byteorder='big')
            logger.info(f"数据源 {source_id} 发送响应长度: {len(response_data)} 字节")
            metadata_socket.sendall(length_bytes)
            metadata_socket.sendall(response_data)
            logger.info(f"数据源 {source_id} 发送响应内容: {response}")
            logger.info(f"数据源 {source_id} 已发送元数据响应")

            # 等待数据连接
            logger.info(f"数据源 {source_id} 等待数据连接...")
            logger.info(f"当前数据服务器信息: 地址={data_server.getsockname()}, 超时={data_server.gettimeout()}")
            data_socket, data_addr = data_server.accept()
            logger.info(f"数据源 {source_id} 接收到数据连接: {data_addr}")
            logger.info(f"新连接套接字信息: 本地地址={data_socket.getsockname()}, 远程地址={data_socket.getpeername()}")

            # 关闭服务器套接字，不再接受新连接
            metadata_server.close()
            data_server.close()

            # 启动数据接收线程
            data_thread = threading.Thread(
                target=self._receive_data_stream,
                args=(data_socket, source_id, subject_id, metadata),
                daemon=True,
            )
            data_thread.start()

            # 监控数据接收线程
            while data_thread.is_alive() and self.is_running:
                time.sleep(1)
                # 更新活动时间
                if source_id in self.data_sources:
                    self.data_sources[source_id]["last_active"] = time.time()

            logger.info(f"数据源 {source_id} 数据接收结束")

        except Exception as e:
            logger.error(f"数据源 {source_id} 接收出错: {e}", exc_info=True)
        finally:
            self._cleanup_data_source(source_id)

    def _receive_data_stream(self, data_socket, source_id, subject_id, metadata):
        """接收数据流"""
        logger.info(f"开始接收数据源 {source_id} 的数据流")
        data_socket.settimeout(30.0)
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            while self.is_running:
                try:
                    # 接收数据包大小
                    size_bytes = data_socket.recv(4)
                    if not size_bytes:
                        logger.warning(f"数据源 {source_id} 连接关闭")
                        break

                    data_size = int.from_bytes(size_bytes, byteorder="big")

                    # 验证数据包大小
                    if data_size <= 0 or data_size > 100 * 1024 * 1024:
                        logger.warning(
                            f"数据源 {source_id} 数据包大小异常: {data_size} 字节"
                        )
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(
                                f"数据源 {source_id} 连续接收到{consecutive_errors}个异常数据包，断开连接"
                            )
                            break
                        continue

                    # 动态调整超时时间
                    data_socket.settimeout(max(30.0, data_size / 100000))

                    # 接收数据
                    data_buffer = bytearray()
                    bytes_received = 0
                    start_time = time.time()

                    while bytes_received < data_size:
                        chunk = data_socket.recv(min(8192, data_size - bytes_received))
                        if not chunk:
                            break
                        data_buffer.extend(chunk)
                        bytes_received += len(chunk)

                        # 检查超时
                        if time.time() - start_time > 60:
                            logger.warning(
                                f"数据源 {source_id} 接收超时: {bytes_received}/{data_size} 字节"
                            )
                            break

                    # 重置超时
                    data_socket.settimeout(30.0)

                    # 检查数据完整性
                    if bytes_received < data_size:
                        logger.warning(
                            f"数据源 {source_id} 数据包不完整: {bytes_received}/{data_size} 字节"
                        )
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            logger.error(
                                f"数据源 {source_id} 连续接收到{consecutive_errors}个不完整数据包，断开连接"
                            )
                            break
                        continue

                    # 解析JSON数据
                    try:
                        json_data = data_buffer.decode("utf-8")
                        data_packet = json.loads(json_data)
                        packet_type = data_packet.get("type")

                        logger.info(
                            f"数据源 {source_id} 接收到数据包: 类型={packet_type}"
                        )

                        # 重置错误计数
                        consecutive_errors = 0

                        # 更新活动时间
                        if source_id in self.data_sources:
                            self.data_sources[source_id]["last_active"] = time.time()

                        if packet_type == "DATA":
                            # 处理EEG数据
                            raw_data = np.array(data_packet["data"])
                            n_channels = data_packet.get("n_channels", 0)
                            n_samples = data_packet.get("n_samples", 0)

                            # 验证数据形状
                            if n_channels > 0 and n_samples > 0:
                                if raw_data.shape != (n_channels, n_samples):
                                    logger.warning(
                                        f"数据源 {source_id} 数据形状不符合预期: {raw_data.shape} vs ({n_channels}, {n_samples})"
                                    )
                                    # 尝试重塑数据
                                    if len(raw_data) == n_channels * n_samples:
                                        raw_data = raw_data.reshape(
                                            n_channels, n_samples
                                        )

                            logger.info(
                                f"数据源 {source_id} 接收到EEG数据: 形状 {raw_data.shape}"
                            )

                            # 更新元数据
                            packet_metadata = metadata.copy()
                            packet_metadata.update(
                                {
                                    "subject_id": data_packet.get(
                                        "subject_id", subject_id
                                    ),
                                    "task_name": data_packet.get("task_name", ""),
                                    "n_channels": n_channels,
                                    "n_samples": n_samples,
                                    "events_info": data_packet.get("events", []),
                                }
                            )

                            # 处理接收到的数据
                            self._process_received_data(
                                source_id, subject_id, packet_metadata, raw_data
                            )

                        elif packet_type == "END":
                            logger.info(f"数据源 {source_id} 数据流接收完成")
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"数据源 {source_id} JSON解析错误: {e}")
                        consecutive_errors += 1
                    except Exception as e:
                        logger.error(
                            f"数据源 {source_id} 处理数据包时出错: {e}", exc_info=True
                        )
                        consecutive_errors += 1

                    # 检查连续错误
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"数据源 {source_id} 连续{consecutive_errors}次接收错误，断开连接"
                        )
                        break

                except socket.timeout:
                    logger.warning(f"数据源 {source_id} 接收超时")
                    # 检查连接是否仍然有效
                    if source_id not in self.data_sources:
                        logger.info(f"数据源 {source_id} 已被移除")
                        break
                except Exception as e:
                    logger.error(
                        f"数据源 {source_id} 接收数据时出错: {e}", exc_info=True
                    )
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"数据源 {source_id} 连续{consecutive_errors}次接收错误，断开连接"
                        )
                        break
                    time.sleep(1)  # 出错后短暂等待

        except (ConnectionResetError, ConnectionAbortedError) as conn_err:
            logger.error(f"数据源 {source_id} 连接中断: {conn_err}")
        except Exception as e:
            logger.error(f"数据源 {source_id} 接收数据时出错: {e}", exc_info=True)
        finally:
            try:
                data_socket.close()
                logger.info(f"数据源 {source_id} 数据套接字已关闭")
            except Exception as e:
                logger.error(f"数据源 {source_id} 关闭套接字时出错: {e}")

    def _process_received_data(self, source_id, subject_id, metadata, eeg_data):
        """处理接收到的EEG数据"""
        try:
            # 更新统计信息
            self.received_count += 1
            self.stats["packets_received"] += 1
            self.stats["bytes_received"] += eeg_data.size * eeg_data.itemsize

            # 保存原始数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(self.save_dir, f"{subject_id}_{timestamp}.npz")
            np.savez(
                save_path,
                eeg_data=eeg_data,
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if isinstance(value, (str, int, float, bool, list))
                },
            )

            logger.info(f"数据源 {source_id} 数据已保存: {save_path}")

            # 添加到缓冲区
            if source_id not in self.data_buffers:
                self.data_buffers[source_id] = []
            self.data_buffers[source_id].append(eeg_data)

            # 获取采样率
            sampling_rate = metadata.get("sampling_rate", 250)

            # 计算缓冲区总时长
            total_samples = sum(data.shape[1] for data in self.data_buffers[source_id])
            buffer_duration = total_samples / sampling_rate

            logger.info(
                f"数据源 {source_id} 缓冲区: {len(self.data_buffers[source_id])}个数据包, {buffer_duration:.2f}秒"
            )

            # 触发数据处理回调
            if self.data_callback and callable(self.data_callback):
                try:
                    # 创建处理数据副本
                    process_data = np.concatenate(self.data_buffers[source_id], axis=1)

                    # 更新元数据
                    process_metadata = metadata.copy()
                    process_metadata.update(
                        {
                            "n_samples": process_data.shape[1],
                            "duration": buffer_duration,
                        }
                    )

                    # 调用回调
                    from common.eegrdd import EEGRDD
                    eegrdd = EEGRDD.from_metadata_and_data(
                metadata={
                    "subject_id": subject_id,
                    "sampling_rate": process_metadata.get("sampling_rate", 1000),
                    "n_channels": len(process_data),
                    "n_samples": len(process_data[0]) if len(process_data) > 0 else 0,
                    **process_metadata
                },
                data=process_data
            )
                    # 仅在driver节点执行Spark操作
                    try:
                        from pyspark import SparkContext
                        if isinstance(SparkContext._active_spark_context, SparkContext):
                            # 直接传递EEGRDD对象给回调函数，由回调函数处理RDD创建
                            self.data_callback(eegrdd)
                        else:
                            logger.warning("当前在worker节点，跳过Spark操作")
                            # 在worker节点也清空缓冲区以避免内存泄漏
                            self.data_buffers[source_id] = []
                            return
                    except Exception as e:
                        logger.error(f"节点类型检查失败: {e}")
                        return

                    # 清空缓冲区
                    self.data_buffers[source_id] = []

                    logger.info(f"数据源 {source_id} 数据处理完成，缓冲区已清空")
                except Exception as e:
                    logger.error(f"数据源 {source_id} 回调执行失败: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"数据源 {source_id} 处理接收数据时出错: {e}", exc_info=True)

    def _receive_metadata(self, sock):
        """从套接字接收元数据JSON"""
        try:
            # 接收消息长度
            length_bytes = sock.recv(4)
            if not length_bytes:
                return None

            message_length = int.from_bytes(length_bytes, byteorder="big")

            # 接收完整消息
            metadata_buffer = b""
            remaining = message_length

            while remaining > 0:
                chunk = sock.recv(min(remaining, 4096))
                if not chunk:
                    break
                metadata_buffer += chunk
                remaining -= len(chunk)

            # 检查是否接收完整
            if len(metadata_buffer) < message_length:
                logger.warning(
                    f"元数据接收不完整: {len(metadata_buffer)}/{message_length} 字节"
                )
                return None

            return metadata_buffer.decode("utf-8")

        except Exception as e:
            logger.error(f"接收元数据出错: {e}", exc_info=True)
            return None

    def _cleanup_data_source(self, source_id):
        """清理数据源资源"""
        if source_id in self.data_sources:
            # 更新状态
            self.data_sources[source_id]["status"] = "stopped"

            # 清理缓冲区
            if source_id in self.data_buffers:
                del self.data_buffers[source_id]
            if source_id in self.metadata_buffers:
                del self.metadata_buffers[source_id]

            logger.info(f"数据源 {source_id} 资源已清理")

    def stop_receiving(self):
        """停止数据接收处理"""
        if not self.is_running:
            return

        logger.info("正在停止EEG数据接收处理...")
        self.is_running = False

        # 清理所有数据源
        for source_id in list(self.data_sources.keys()):
            self._cleanup_data_source(source_id)

        # 清空数据结构
        self.data_sources.clear()
        self.data_buffers.clear()
        self.metadata_buffers.clear()

        logger.info("EEG数据接收处理已停止")

    def get_data_sources(self):
        """获取当前数据源信息"""
        return self.data_sources.copy()

    def get_stats(self):
        """获取接收器统计信息"""
        return self.stats.copy()
