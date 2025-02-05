from receiver.receive_data import MetadataReceiver, EEGDataReceiver
from processor.visualizer import EEGVisualizer
import time

class DataDebugger:
    def process(self, eegrdd):
        """打印接收到的数据信息"""
        try:
            print(f"\n[数据接收] "
                  f"通道: {eegrdd.channel}, "
                  f"数据包: #{eegrdd.chunk_idx}, "
                  f"时间点数: {len(eegrdd.time_points)}, "
                  f"时间范围: [{eegrdd.time_points[0]:.3f}, {eegrdd.time_points[-1]:.3f}]s, "
                  f"电压范围: [{eegrdd.voltage_data.min():.2f}, {eegrdd.voltage_data.max():.2f}]μV")
        except Exception as e:
            print(f"[错误] 数据打印失败: {e}")

def main():
    save_dir = "/opt/eegproject/data/output"
    
    print("[初始化] 启动EEG可视化系统...")
    
    visualizer = EEGVisualizer(host='0.0.0.0', port=5000)
    visualizer.start()
    
    time.sleep(1)  # 等待可视化服务器启动
    
    metadata_receiver = MetadataReceiver(save_dir)
    metadata = metadata_receiver.start()
    
    if metadata:
        eeg_receiver = EEGDataReceiver(save_dir)
        eeg_receiver.add_processor(visualizer)
        
        print("\n[开始] 启动实时可视化...")
        print(f"[提示] 访问 http://localhost:5000 查看EEG数据")
        
        try:
            eeg_receiver.start(metadata)
        except KeyboardInterrupt:
            print("\n[退出] 正在停止服务...")
        except Exception as e:
            print(f"\n[错误] 运行时错误: {e}")
    else:
        print("[错误] 元数据接收失败")

if __name__ == "__main__":
    main()
