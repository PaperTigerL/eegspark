import os
import json
import numpy as np
import pandas as pd
import mne
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class ChannelLocation:
    """通道位置信息"""
    name: str                # 通道名称 (e.g., 'Fp1', 'Fp2')
    type: str                # 通道类型 (e.g., 'EEG', 'EOG')
    unit: str                # 单位 (e.g., 'µV')
    x: float                 # X坐标
    y: float                 # Y坐标
    z: float                 # Z坐标
    reference: str           # 参考方式

@dataclass
class EEGDataInfo:
    """EEG数据信息"""
    subject_id: str          # 受试者ID
    task_name: str           # 任务名称
    sampling_rate: float     # 采样率
    n_channels: int          # 通道数
    n_samples: int           # 样本数
    duration: float          # 持续时间(秒)
    channel_locations: List[ChannelLocation]  # 通道位置信息
    reference: str          # 参考方式
    filters: Dict           # 滤波器设置
    annotations: List[Dict] # 事件标注
    recording_date: str     # 记录日期
    raw: Any = None         # 添加raw属性，存储mne.io.Raw对象（符合国际EEG处理惯例）

class EEGReader:
    """EEG数据读取器"""
    def __init__(self, dataset_root: str):
        """初始化EEG数据读取器
        
        Args:
            dataset_root: 数据集根目录路径
        """
        self.dataset_root = dataset_root
        self.dataset_info = self._load_dataset_info()
        self.montage = None
        self._load_standard_montage()

    def _load_dataset_info(self) -> Dict:
        """加载数据集描述信息"""
        info = {}
        
        # 读取dataset_description.json
        desc_path = os.path.join(self.dataset_root, 'dataset_description.json')
        if os.path.exists(desc_path):
            with open(desc_path, 'r', encoding='utf-8') as f:
                info['description'] = json.load(f)

        # 读取participants.tsv
        participants_path = os.path.join(self.dataset_root, 'participants.tsv')
        if os.path.exists(participants_path):
            try:
                participants_df = pd.read_csv(participants_path, sep='\t')
                # 确保participant_id列存在
                if 'participant_id' in participants_df.columns:
                    # 转换为字典格式，方便查询
                    info['participants'] = {}
                    for _, row in participants_df.iterrows():
                        subject_id = row['participant_id'].replace('sub-', '')
                        info['participants'][subject_id] = row.to_dict()
                else:
                    print(f"警告: participants.tsv中未找到participant_id列")
            except Exception as e:
                print(f"警告: 读取participants.tsv失败: {e}")
                info['participants'] = {}

        # 读取README
        readme_path = os.path.join(self.dataset_root, 'README')
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                info['readme'] = f.read().strip()

        # 扫描受试者目录
        info['subjects'] = []
        for item in os.listdir(self.dataset_root):
            if item.startswith('sub-'):
                subject_id = item.replace('sub-', '')
                subject_eeg_dir = os.path.join(self.dataset_root, item, 'eeg')
                if os.path.exists(subject_eeg_dir):
                    info['subjects'].append(subject_id)

        if not info['subjects']:
            print(f"警告: 在{self.dataset_root}中未找到任何受试者数据")

        return info

    def _load_standard_montage(self):
        """加载标准10-20系统电极位置"""
        try:
            self.montage = mne.channels.make_standard_montage('standard_1020')
        except Exception as e:
            print(f"警告: 无法加载标准电极位置: {e}")

    def read_eeg_data(self, subject_id: str, task_name: str) -> EEGDataInfo:
        """读取EEG数据和元数据
        
        Args:
            subject_id: 受试者ID
            task_name: 任务名称
            
        Returns:
            EEGDataInfo: EEG数据信息对象
        """
        # 构建文件路径
        base_path = os.path.join(self.dataset_root, f"sub-{subject_id}", 'eeg')
        eeg_file = os.path.join(base_path, f"sub-{subject_id}_task-{task_name}_eeg")
        
        # 读取EEG数据
        raw = None
        if os.path.exists(eeg_file + '.fif'):
            raw = mne.io.read_raw_fif(eeg_file + '.fif', preload=True)
        elif os.path.exists(eeg_file + '.vhdr'):
            raw = mne.io.read_raw_brainvision(eeg_file + '.vhdr', preload=True)
        elif os.path.exists(eeg_file + '.edf'):
            raw = mne.io.read_raw_edf(eeg_file + '.edf', preload=True)
        else:
            raise FileNotFoundError(f"找不到支持的EEG数据文件: {eeg_file}.*")

        # 读取元数据JSON文件
        metadata = {}
        json_file = eeg_file + '.json'
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # 获取通道位置信息
        channel_locations = []
        if self.montage:
            for ch_name in raw.ch_names:
                if ch_name in self.montage.ch_names:
                    pos = self.montage.get_positions()['ch_pos'][ch_name]
                    channel_locations.append(ChannelLocation(
                        name=ch_name,
                        type='EEG',
                        unit='µV',
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=float(pos[2]),
                        reference=metadata.get('EEGReference', 'unknown')
                    ))

        # 获取事件标注
        events = []
        try:
            # 首先尝试从MNE对象中获取事件
            events_data = mne.events_from_annotations(raw)
            for event_id, event_name in events_data[1].items():
                mask = events_data[0][:, 2] == int(event_id)
                event_times = events_data[0][mask, 0] / raw.info['sfreq']
                for t in event_times:
                    events.append({
                        'type': event_name,
                        'onset': float(t),
                        'duration': metadata.get('TaskEvents', {}).get(event_name, {}).get('Duration', 0.0)
                    })
            
            # 如果从MNE获取的事件为空，尝试读取events.tsv文件
            if len(events) == 0:
                events_tsv_file = os.path.join(base_path, f"sub-{subject_id}_task-{task_name}_events.tsv")
                if os.path.exists(events_tsv_file):
                    print(f"从events.tsv文件读取事件: {events_tsv_file}")
                    try:
                        events_df = pd.read_csv(events_tsv_file, sep='\t')
                        for _, row in events_df.iterrows():
                            event = {
                                'type': row.get('type', 'unknown'),
                                'onset': float(row.get('onset', 0.0)),
                                'duration': float(row.get('duration', 0.0))
                            }
                            events.append(event)
                        print(f"从events.tsv文件读取了{len(events)}个事件")
                    except Exception as e:
                        print(f"读取events.tsv文件出错: {e}")
        except Exception as e:
            print(f"警告: 无法读取事件标注: {e}")
            
            # 如果从MNE获取事件失败，尝试读取events.tsv文件
            events_tsv_file = os.path.join(base_path, f"sub-{subject_id}_task-{task_name}_events.tsv")
            if os.path.exists(events_tsv_file):
                print(f"从events.tsv文件读取事件: {events_tsv_file}")
                try:
                    events_df = pd.read_csv(events_tsv_file, sep='\t')
                    for _, row in events_df.iterrows():
                        event = {
                            'type': row.get('type', 'unknown'),
                            'onset': float(row.get('onset', 0.0)),
                            'duration': float(row.get('duration', 0.0))
                        }
                        events.append(event)
                    print(f"从events.tsv文件读取了{len(events)}个事件")
                except Exception as e:
                    print(f"读取events.tsv文件出错: {e}")
        # 处理日期
        meas_date = raw.info.get('meas_date')
        recording_date = None
        if meas_date is not None:
            try:
                recording_date = meas_date.strftime('%Y-%m-%d %H:%M:%S')
            except:
                recording_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            recording_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 构建EEG数据信息对象
        return EEGDataInfo(
            subject_id=subject_id,
            task_name=task_name,
            sampling_rate=raw.info['sfreq'],
            n_channels=len(raw.ch_names),
            n_samples=len(raw.times),
            duration=len(raw.times) / raw.info['sfreq'],
            channel_locations=channel_locations,
            reference=metadata.get('EEGReference', 'unknown'),
            filters={
                'highpass': raw.info.get('highpass', 0.0),
                'lowpass': raw.info.get('lowpass', raw.info['sfreq'] / 2)
            },
            annotations=events,
            recording_date=recording_date,  # 使用安全处理后的变量
            raw=raw  # 添加raw属性
        )

    def get_channel_layout(self) -> Dict[str, Any]:
        """获取通道布局信息"""
        if not self.montage:
            return {}
            
        layout = mne.channels.find_layout(self.montage)
        return {
            'names': layout.names,
            'pos': layout.pos.tolist(),
            'box': layout.box.tolist(),
            'ids': layout.ids.tolist()
        }

    def plot_topography(self, raw_data, channel_names=None):
        """绘制地形图
        
        Args:
            raw_data: 原始EEG数据
            channel_names: 通道名称列表
        """
        if not self.montage:
            raise ValueError("未加载电极位置信息，无法绘制地形图")
            
        if channel_names is None:
            channel_names = self.montage.ch_names
            
        # 创建Info对象
        info = mne.create_info(channel_names, sfreq=250., ch_types='eeg')
        info.set_montage(self.montage)
        
        # 创建Evoked对象并绘制地形图
        evoked = mne.EvokedArray(raw_data, info)
        evoked.plot_topomap()

def main():
    """主函数示例"""
    # 示例用法
    dataset_root = "path/to/your/dataset"
    reader = EEGReader(dataset_root)
    
    try:
        # 读取特定受试者的特定任务数据
        data_info = reader.read_eeg_data("001", "rest")
        print("\nEEG数据信息:")
        print(f"受试者ID: {data_info.subject_id}")
        print(f"任务名称: {data_info.task_name}")
        print(f"采样率: {data_info.sampling_rate} Hz")
        print(f"通道数: {data_info.n_channels}")
        print(f"样本数: {data_info.n_samples}")
        print(f"持续时间: {data_info.duration:.2f} 秒")
        print(f"参考方式: {data_info.reference}")
        
        # 打印通道位置信息
        print("\n通道位置信息:")
        for ch in data_info.channel_locations[:5]:  # 只打印前5个通道
            print(f"{ch.name}: ({ch.x:.2f}, {ch.y:.2f}, {ch.z:.2f})")
            
        # 打印事件信息
        print("\n事件标注:")
        for event in data_info.annotations[:5]:  # 只打印前5个事件
            print(f"类型: {event['type']}, 开始时间: {event['onset']:.2f}s, 持续时间: {event['duration']:.2f}s")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()