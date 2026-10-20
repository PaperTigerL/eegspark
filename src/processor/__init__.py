# EEG处理器模块
from processor.eeg_processor import EEGProcessor, ProcessingConfig
from processor.realtime_eeg_processor import RealtimeEEGProcessor, RealtimeProcessingConfig
from processor.time_frequency import TimeFrequencyAnalyzer
from processor.connectivity import ConnectivityAnalyzer
from processor.source_localization import SourceLocalizer
from processor.statistical import StatisticalAnalyzer

__all__ = [
    'EEGProcessor',
    'ProcessingConfig',
    'RealtimeEEGProcessor',
    'RealtimeProcessingConfig',
    'TimeFrequencyAnalyzer',
    'ConnectivityAnalyzer',
    'SourceLocalizer',
    'StatisticalAnalyzer'
]
from processor.eeg_processor import EEGProcessor, ProcessingConfig
from processor.pipeline import EEGProcessingPipeline

__all__ = ['EEGProcessor', 'ProcessingConfig', 'EEGProcessingPipeline']