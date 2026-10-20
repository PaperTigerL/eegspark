import numpy as np
import mne
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SourceLocalization')

class SourceLocalizer:
    """EEG源定位器 - 实现前向模型和反向模型进行脑源定位"""
    
    def __init__(self):
        """初始化源定位器"""
        self.last_error = None
        self._forward_model = None
        self._inverse_operator = None
    
    def setup_forward_model(self, raw: mne.io.Raw, 
                           trans: Optional[str] = None,
                           bem: Optional[str] = None,
                           spacing: str = 'oct6') -> Dict[str, Any]:
        """设置前向模型
        
        Args:
            raw: MNE Raw对象
            trans: 头部位置文件路径，如果为None则使用默认值
            bem: BEM模型文件路径，如果为None则使用默认值
            spacing: 源空间间距，默认为'oct6'
            
        Returns:
            Dict: 前向模型设置结果
        """
        try:
            # 检查是否有通道位置信息
            if not raw.info['dig']:
                logger.warning("Raw对象没有通道位置信息，尝试使用标准位置")
                try:
                    # 尝试设置标准montage
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage)
                except Exception as e:
                    logger.error(f"设置标准电极位置失败: {e}")
                    return {
                        'success': False,
                        'error': f"设置标准电极位置失败: {e}"
                    }
            
            # 创建源空间
            try:
                # 使用MNE的样本数据集的头部模型
                sample_data_folder = mne.datasets.sample.data_path()
                subjects_dir = sample_data_folder / 'subjects'
                subject = 'sample'
                
                # 创建源空间
                src = mne.setup_source_space(subject, spacing=spacing,
                                           subjects_dir=subjects_dir)
                
                # 使用样本数据集的BEM模型
                if bem is None:
                    bem = sample_data_folder / 'subjects' / 'sample' / 'bem' / 'sample-5120-bem-sol.fif'
                
                # 使用样本数据集的trans文件
                if trans is None:
                    trans = sample_data_folder / 'MEG' / 'sample' / 'sample_audvis_raw-trans.fif'
                
                # 计算前向模型
                fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                              bem=bem, meg=False, eeg=True)
                
                # 保存前向模型
                self._forward_model = fwd
                
                return {
                    'success': True,
                    'forward_model': fwd
                }
                
            except Exception as e:
                logger.error(f"创建源空间失败: {e}")
                return {
                    'success': False,
                    'error': f"创建源空间失败: {e}"
                }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"设置前向模型失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def compute_inverse_operator(self, raw: mne.io.Raw, 
                               noise_cov: Optional[mne.Covariance] = None,
                               method: str = 'MNE') -> Dict[str, Any]:
        """计算反向算子
        
        Args:
            raw: MNE Raw对象
            noise_cov: 噪声协方差矩阵，如果为None则自动计算
            method: 反向方法，可选'MNE', 'dSPM', 'sLORETA', 'eLORETA'
            
        Returns:
            Dict: 反向算子计算结果
        """
        try:
            # 检查前向模型是否已设置
            if self._forward_model is None:
                logger.error("前向模型未设置，请先调用setup_forward_model")
                return {
                    'success': False,
                    'error': "前向模型未设置，请先调用setup_forward_model"
                }
            
            # 如果未提供噪声协方差矩阵，则自动计算
            if noise_cov is None:
                logger.info("自动计算噪声协方差矩阵")
                try:
                    # 创建空数据段
                    empty_room_raw = mne.io.RawArray(np.zeros_like(raw.get_data()), raw.info)
                    noise_cov = mne.compute_raw_covariance(empty_room_raw)
                except Exception as e:
                    logger.error(f"计算噪声协方差矩阵失败: {e}")
                    return {
                        'success': False,
                        'error': f"计算噪声协方差矩阵失败: {e}"
                    }
            
            # 计算反向算子
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                raw.info, self._forward_model, noise_cov,
                loose=0.2, depth=0.8
            )
            
            # 保存反向算子
            self._inverse_operator = inverse_operator
            
            return {
                'success': True,
                'inverse_operator': inverse_operator
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"计算反向算子失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def apply_inverse(self, raw: mne.io.Raw, 
                     method: str = 'MNE',
                     snr: float = 3.0,
                     lambda2: Optional[float] = None) -> Dict[str, Any]:
        """应用反向算子进行源定位
        
        Args:
            raw: MNE Raw对象
            method: 反向方法，可选'MNE', 'dSPM', 'sLORETA', 'eLORETA'
            snr: 信噪比
            lambda2: 正则化参数，如果为None则自动计算
            
        Returns:
            Dict: 源定位结果
        """
        try:
            # 检查反向算子是否已计算
            if self._inverse_operator is None:
                logger.error("反向算子未计算，请先调用compute_inverse_operator")
                return {
                    'success': False,
                    'error': "反向算子未计算，请先调用compute_inverse_operator"
                }
            
            # 计算lambda2
            if lambda2 is None:
                lambda2 = 1.0 / snr ** 2
            
            # 应用反向算子
            stc = mne.minimum_norm.apply_inverse_raw(
                raw, self._inverse_operator, lambda2=lambda2,
                method=method, pick_ori=None
            )
            
            return {
                'success': True,
                'source_time_course': stc,
                'method': method
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"应用反向算子失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def compute_beamformer(self, raw: mne.io.Raw, 
                         data_cov: Optional[mne.Covariance] = None,
                         noise_cov: Optional[mne.Covariance] = None,
                         reg: float = 0.05,
                         method: str = 'dics') -> Dict[str, Any]:
        """计算波束成形器
        
        Args:
            raw: MNE Raw对象
            data_cov: 数据协方差矩阵，如果为None则自动计算
            noise_cov: 噪声协方差矩阵，如果为None则自动计算
            reg: 正则化参数
            method: 波束成形方法，可选'lcmv', 'dics'
            
        Returns:
            Dict: 波束成形结果
        """
        try:
            # 检查前向模型是否已设置
            if self._forward_model is None:
                logger.error("前向模型未设置，请先调用setup_forward_model")
                return {
                    'success': False,
                    'error': "前向模型未设置，请先调用setup_forward_model"
                }
            
            # 如果未提供数据协方差矩阵，则自动计算
            if data_cov is None:
                logger.info("自动计算数据协方差矩阵")
                try:
                    data_cov = mne.compute_raw_covariance(raw)
                except Exception as e:
                    logger.error(f"计算数据协方差矩阵失败: {e}")
                    return {
                        'success': False,
                        'error': f"计算数据协方差矩阵失败: {e}"
                    }
            
            # 如果未提供噪声协方差矩阵，则自动计算
            if noise_cov is None:
                logger.info("自动计算噪声协方差矩阵")
                try:
                    # 创建空数据段
                    empty_room_raw = mne.io.RawArray(np.zeros_like(raw.get_data()), raw.info)
                    noise_cov = mne.compute_raw_covariance(empty_room_raw)
                except Exception as e:
                    logger.error(f"计算噪声协方差矩阵失败: {e}")
                    return {
                        'success': False,
                        'error': f"计算噪声协方差矩阵失败: {e}"
                    }
            
            # 根据方法选择波束成形器
            if method.lower() == 'lcmv':
                # 计算LCMV波束成形器
                filters = mne.beamformer.make_lcmv(
                    raw.info, self._forward_model, data_cov,
                    reg=reg, noise_cov=noise_cov,
                    pick_ori='max-power'
                )
                
                # 应用波束成形器
                stc = mne.beamformer.apply_lcmv_raw(
                    raw, filters
                )
                
            elif method.lower() == 'dics':
                # 计算DICS波束成形器
                # 首先计算交叉谱密度
                csd = mne.time_frequency.csd_morlet(
                    raw, frequencies=[10], n_cycles=5, decim=1
                )
                
                # 计算DICS波束成形器
                filters = mne.beamformer.make_dics(
                    raw.info, self._forward_model, csd,
                    reg=reg, noise_csd=None,
                    pick_ori='max-power'
                )
                
                # 应用波束成形器
                stc = mne.beamformer.apply_dics_csd(
                    csd, filters
                )
                
            else:
                logger.error(f"不支持的波束成形方法: {method}")
                return {
                    'success': False,
                    'error': f"不支持的波束成形方法: {method}"
                }
            
            return {
                'success': True,
                'source_time_course': stc,
                'method': method
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"计算波束成形器失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }