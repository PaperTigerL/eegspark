import os
import time
import socket
import threading
import json
import sys
import numpy as np
import logging
import uuid
import argparse
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取 tests 目录路径
tests_dir = os.path.dirname(current_file_path)
# 获取 src 目录路径（因为 tests 在 src 下）
src_dir = os.path.dirname(tests_dir)

# 将 src 目录加入 Python 路径中
if src_dir not in sys.path:
    sys.path.append(src_dir)
from src.sender.eeg_reader import EEGReader, EEGDataInfo

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EEGDataSender")


class EEGDataSender:
    """EEG数据发送器 - 使用接收器分配的端口"""

    def __init__(
        self,
        dataset_root: str = "/app/data/ds005779",
        service_id: str = None,
        receiver_host: str = None,
        receiver_connection_port: int = 9000,
    ):
        """初始化EEG数据发送器

        Args:
            dataset_root: 数据集根目录路径
            service_id: 服务ID，用于服务注册
            receiver_host: 接收器主机地址，如果指定则直接连接到该地址
            receiver_connection_port: 接收器连接请求端口
        """
        self.dataset_root = dataset_root
        self.eeg_reader = EEGReader(dataset_root)
        self.receiver_connection_port = receiver_connection_port
        self.receiver_host = receiver_host or "spark-worker"
        self.client_id = str(uuid.uuid4())
        self.allocated_metadata_port = None
        self.allocated_data_port = None

    def generate_payload_id(self, subject_id: str, task_name: str) -> str:
        """生成唯一的数据负载ID"""
        timestamp = int(time.time())
        return f"{subject_id}_{task_name}_{timestamp}"

    def request_connection(self, subject_id: str, task_name: str) -> dict:
        """向接收端请求连接和资源分配

        Args:
            subject_id: 受试者ID
            task_name: 任务名称

        Returns:
            dict: 包含分配的端口和连接信息，失败时返回None
        """
        logger.info(f"客户端ID: {self.client_id}")
        
        # 读取指定受试者和任务的元数据
        actual_subject_id = subject_id[4:] if subject_id.startswith("sub-") else subject_id
        self.eeg_reader.read_eeg_data(actual_subject_id, task_name)
        
        # 向接收端请求连接和资源分配
        logger.info("向接收端请求连接和资源分配...")

        try:
            # 确定接收端地址
            host = self.receiver_host
            port = self.receiver_connection_port

            # 解析主机名
            ip = socket.gethostbyname(host)
            logger.info(f"解析接收器主机 {host} -> IP: {ip}")

            # 创建客户端套接字
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(None)  # 无限等待
            client_socket.connect((ip, port))
            logger.info(f"已连接到接收器 {ip}:{port}")

            # 创建连接请求
            connection_request = {
                "type": "CONNECTION_REQUEST",
                "client_id": self.client_id,
                "subject_id": subject_id,
                "task_name": task_name,
                "timestamp": time.time(),
                "sender_info": {
                    "capabilities": ["metadata", "streaming_data"],
                },
            }

            # 发送连接请求
            request_json = json.dumps(connection_request).encode("utf-8")
            client_socket.sendall(len(request_json).to_bytes(4, byteorder="big"))
            client_socket.sendall(request_json)
            logger.info("已发送连接请求，等待接收端响应...")
            logger.info(f"发送的连接请求内容: {connection_request}")

            # 接收响应
            logger.info("开始接收响应长度...")
            # 循环接收4字节长度
            length_bytes = b''
            while len(length_bytes) < 4:
                chunk = client_socket.recv(4 - len(length_bytes))
                if not chunk:
                    raise ConnectionError("连接已关闭")
                length_bytes += chunk
            response_length = int.from_bytes(length_bytes, byteorder="big")
            logger.info(f"接收到的响应长度: {response_length}字节")

            # 循环接收响应数据
            logger.info("开始接收响应数据...")
            response_data = b''
            while len(response_data) < response_length:
                chunk = client_socket.recv(min(4096, response_length - len(response_data)))
                if not chunk:
                    raise ConnectionError("连接已关闭")
                response_data += chunk
            
            # 记录接收到的响应数据
            logger.info(f"接收到的响应数据: {response_data}")
            
            # 解析响应数据
            try:
                response = json.loads(response_data.decode("utf-8"))
                logger.info(f"解析后的响应内容: {response}")
            except Exception as e:
                logger.error(f"解析响应数据失败: {e}")
                response = None

            if response.get("status") == "success":
                # 直接从响应中获取数据端口和元数据端口
                self.allocated_metadata_port = response.get("metadata_port")
                self.allocated_data_port = response.get("data_port")
                logger.info(
                    f"连接请求成功，分配端口: 元数据={self.allocated_metadata_port}, 数据={self.allocated_data_port}"
                )
                # 返回包含两个端口的字典
                return {"metadata_port": self.allocated_metadata_port, "data_port": self.allocated_data_port}
            else:
                error_msg = response.get("message", "未知错误")
                logger.error(f"连接请求失败: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"连接请求失败: {e}")
            if isinstance(e, socket.timeout):
                logger.error("连接超时，可能是接收端未及时响应或网络延迟")
            return None

    def send_metadata(
        self, subject_id: str, task_name: str, allocated_info: dict = None
    ) -> str:
        """发送元数据到接收器

        Returns:
            str: 生成的payload_id，如果发送失败则返回None
        """
        logger.info("发送元数据到接收器...")

        try:
            # 检查是否已分配端口
            if not self.allocated_metadata_port:
                logger.error("未分配元数据端口，请先调用request_connection()")
                return None

            # 处理subject_id（移除'sub-'前缀）
            actual_subject_id = subject_id[4:] if subject_id.startswith("sub-") else subject_id

            # 读取EEG数据信息
            eeg_info = self.eeg_reader.read_eeg_data(actual_subject_id, task_name)

            # 生成payload_id
            payload_id = self.generate_payload_id(subject_id, task_name)

            # 准备元数据
            metadata = self._prepare_metadata(eeg_info, payload_id)

            # 创建数据负载
            data_payload = {
                "payload_id": payload_id,
                "subject_id": subject_id,
                "task_name": task_name,
                "timestamp": time.time(),
                "payload_type": "metadata",
                "content": metadata,
                "client_id": self.client_id,
            }

            # 使用分配的元数据端口
            host = self.receiver_host
            port = self.allocated_metadata_port
            ip = socket.gethostbyname(host)

            logger.info(f"连接到分配的元数据服务器 {ip}:{port}...")

            # 创建客户端套接字
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(None)  # 无限等待
            client_socket.connect((ip, port))

            # 将数据负载转换为JSON字符串并编码为字节
            payload_json = json.dumps(data_payload).encode("utf-8")

            # 发送数据长度和内容
            client_socket.sendall(len(payload_json).to_bytes(4, byteorder="big"))
            client_socket.sendall(payload_json)

            # 接收接收器响应
            logger.info("开始接收响应长度...")
            try:
                # 增加超时设置并添加详细日志
                response_length_bytes = client_socket.recv(4)
                if not response_length_bytes:
                    logger.error("接收响应长度失败: 连接已关闭")
                    return None
                
                response_length = int.from_bytes(response_length_bytes, byteorder="big")
                logger.info(f"接收到响应长度: {response_length} 字节")
                
                # 接收实际响应数据
                response_data = b''
                while len(response_data) < response_length:
                    chunk = client_socket.recv(min(4096, response_length - len(response_data)))
                    if not chunk:
                        raise ConnectionError("连接中断")
                    response_data += chunk
                
                response = json.loads(response_data.decode("utf-8"))
                logger.info(f"接收到响应: {response}")
            except socket.timeout:
                logger.error("接收响应超时，请检查接收端是否发送了响应")
                return None
            except Exception as e:
                logger.error(f"接收响应出错: {e}")
                return None
            
            if response.get("status") == "success":
                logger.info(f"接收端已确认元数据接收: {response.get('message')}")
                logger.info(f"分配的数据端口: {response.get('data_port')}")
            else:
                logger.error(f"接收端返回错误: {response.get('message')}")
            
            logger.info(f"成功发送元数据，payload_id: {payload_id}")

            # 关闭连接
            client_socket.close()

            return payload_id
        except Exception as e:
            logger.error(f"发送元数据失败: {e}")
            return None

    def _prepare_metadata(self, eeg_info: EEGDataInfo, payload_id: str) -> dict:
        """准备要发送的元数据"""
        # 从EEGDataInfo中提取所有不包含原始数据的信息
        metadata = {
            "payload_id": payload_id,
            "subject_id": eeg_info.subject_id,
            "task_name": eeg_info.task_name,
            "sampling_rate": eeg_info.sampling_rate,
            "n_channels": eeg_info.n_channels,
            "n_samples": eeg_info.n_samples,
            "duration": eeg_info.duration,
            "reference": eeg_info.reference,
            "filters": eeg_info.filters,
            "recording_date": eeg_info.recording_date,
            "channels_info": getattr(eeg_info, "channels_info", {}),
            "events_info": getattr(eeg_info, "events_info", []),
            "json_metadata": getattr(eeg_info, "json_metadata", {}),
            "channel_names": [ch.name for ch in eeg_info.channel_locations],
            "channel_types": [ch.type for ch in eeg_info.channel_locations],
            "channel_units": [ch.unit for ch in eeg_info.channel_locations],
            "channel_locations": [
                {
                    "name": ch.name,
                    "type": ch.type,
                    "unit": ch.unit,
                    "x": ch.x,
                    "y": ch.y,
                    "z": ch.z,
                }
                for ch in eeg_info.channel_locations
            ],
            "annotations": eeg_info.annotations,
        }
        return metadata
        
       
    def start_streaming_data(
        self,
        subject_id: str,
        task_name: str,
        payload_id: str,
        speed_factor: float = 1.0,
        chunk_samples: int = 250,
    ):
        """开始流式发送EEG数据

        Args:
            subject_id: 受试者ID
            task_name: 任务名称
            payload_id: 数据负载ID，用于关联元数据
            speed_factor: 速度因子（>1加速，<1减速）
            chunk_samples: 每次发送的样本数
        """
        if hasattr(self, 'data_thread') and self.data_thread.is_alive():
            logger.warning("已有数据流在发送中，请先停止")
            return

        try:
            # 处理subject_id（移除'sub-'前缀）
            actual_subject_id = subject_id[4:] if subject_id.startswith("sub-") else subject_id

            # 读取EEG数据
            eeg_info = self.eeg_reader.read_eeg_data(actual_subject_id, task_name)
            raw_data = eeg_info.raw.get_data()
            total_samples = raw_data.shape[1]
            total_chunks = (total_samples + chunk_samples - 1) // chunk_samples

            # 启动数据发送线程
            self.is_running = True
            self.data_thread = threading.Thread(
                target=self._stream_data_thread,
                args=(raw_data, eeg_info, payload_id, speed_factor, chunk_samples),
                daemon=True,
            )
            self.data_thread.start()
            logger.info(f"开始发送数据流: 受试者 {subject_id}, 任务 {task_name}")
            return True

        except Exception as e:
            logger.error(f"启动数据流失败: {e}")
            return False

    def _stream_data_thread(
        self,
        raw_data: np.ndarray,
        eeg_info: EEGDataInfo,
        payload_id: str,
        speed_factor: float,
        chunk_samples: int,
    ):
        """数据流发送线程"""
        try:
            total_samples = raw_data.shape[1]
            total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            chunk_duration = chunk_samples / eeg_info.sampling_rate

            logger.info(
                f"数据流参数: {eeg_info.n_channels}通道, {total_samples}样本, "
                f"采样率{eeg_info.sampling_rate}Hz, {total_chunks}数据块"
            )

            # 使用分配的数据端口
            host = self.receiver_host
            port = self.allocated_data_port
            ip = socket.gethostbyname(host)

            logger.info(f"连接到分配的数据服务器 {ip}:{port}...")

            # 创建客户端套接字
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(None)  # 无限等待
            client_socket.connect((ip, port))
            logger.info(f"成功连接到接收器数据服务")

            # 发送开始标记
            start_msg = {
                "type": "START",
                "client_id": self.client_id,
                "payload_id": payload_id,
                "total_chunks": total_chunks,
                "chunk_samples": chunk_samples,
            }
            start_data = json.dumps(start_msg).encode("utf-8")
            client_socket.sendall(len(start_data).to_bytes(4, byteorder="big"))
            client_socket.sendall(start_data)

            # 逐块发送数据
            current_sample = 0

            for chunk_idx in range(total_chunks):
                if not self.is_running:
                    break

                start_time = time.time()

                # 准备当前数据块
                end_sample = min(current_sample + chunk_samples, total_samples)
                chunk_data = raw_data[:, current_sample:end_sample]
                actual_samples = end_sample - current_sample

                # 构建数据包
                data_packet = {
                    "type": "DATA",
                    "client_id": self.client_id,
                    "payload_id": payload_id,
                    "sequence_num": chunk_idx + 1,
                    "total_chunks": total_chunks,
                    "n_channels": eeg_info.n_channels,
                    "n_samples": actual_samples,
                    "timestamp": time.time(),
                    "data": chunk_data.tolist(),  # 转换为列表
                }

                # 序列化并发送
                json_data = json.dumps(data_packet).encode("utf-8")
                client_socket.sendall(len(json_data).to_bytes(4, byteorder="big"))
                client_socket.sendall(json_data)

                logger.info(
                    f"发送数据块 {chunk_idx+1}/{total_chunks}: "
                    f"{actual_samples}样本, {len(json_data)}字节"
                )

                # 更新当前样本位置
                current_sample = end_sample

                # 控制发送速率以模拟实时数据生成
                elapsed = time.time() - start_time
                sleep_time = (chunk_duration / speed_factor) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 发送结束标记
            if self.is_running:
                end_msg = {
                    "type": "END",
                    "client_id": self.client_id,
                    "payload_id": payload_id,
                }
                end_data = json.dumps(end_msg).encode("utf-8")
                client_socket.sendall(len(end_data).to_bytes(4, byteorder="big"))
                client_socket.sendall(end_data)
                logger.info(f"数据流发送完成")

            # 关闭连接
            client_socket.close()

        except Exception as e:
            logger.error(f"数据流发送错误: {e}")
        finally:
            self.is_running = False

    def stop_streaming(self):
        """停止数据流发送"""
        self.is_running = False
        if hasattr(self, 'data_thread') and self.data_thread.is_alive():
            self.data_thread.join(timeout=2.0)
            logger.info("数据流发送已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="EEG数据发送器")
    parser.add_argument(
        "--subject-id", default="01", help="受试者ID (例如: '01' 或 'sub-01')"
    )
    parser.add_argument("--task", default="resting", help="任务名称")
    parser.add_argument("--dataset-root", default="/app/data/ds005779", help="数据集根目录")
    parser.add_argument("--receiver-host", default="spark-worker", help="接收器主机名")
    parser.add_argument("--connection-port", type=int, default=9000, help="接收器连接端口")
    parser.add_argument("--speed", type=float, default=2.0, help="数据发送速度因子")
    parser.add_argument("--chunk-size", type=int, default=500, help="每次发送的样本数")
    
    args = parser.parse_args()

    try:
        logger.info(f"=== EEG数据发送器 - 受试者 {args.subject_id} ===")

        # 创建发送器
        sender = EEGDataSender(
            dataset_root=args.dataset_root,
            receiver_host=args.receiver_host,
            receiver_connection_port=args.connection_port,
        )

        logger.info(f"客户端ID: {sender.client_id}")

        # 请求连接并获取分配的端口
        allocated_info = sender.request_connection(args.subject_id, args.task)
        if not allocated_info:
            logger.error("连接请求失败，退出")
            return

        # 发送元数据
        payload_id = sender.send_metadata(args.subject_id, args.task, allocated_info)
        if not payload_id:
            logger.error("发送元数据失败，退出")
            return

        # 开始发送数据流
        sender.start_streaming_data(
            args.subject_id, args.task, payload_id, args.speed, args.chunk_size
        )

        # 等待数据发送完成或中断
        try:
            while hasattr(sender, 'is_running') and sender.is_running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("\n接收到中断，停止发送")
            sender.stop_streaming()

    except Exception as e:
        logger.error(f"错误: {e}")
    except KeyboardInterrupt:
        logger.info("\n程序被中断")


if __name__ == "__main__":
    main()