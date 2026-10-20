import numpy as np
import mne
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy import stats

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StatisticalAnalysis')

class StatisticalAnalyzer:
    """EEG统计分析器 - 实现多重比较校正和置换检验等统计分析"""
    
    def __init__(self):
        """初始化统计分析器"""
        self.last_error = None
    
    def multiple_comparison_correction(self, p_values: np.ndarray, 
                                     method: str = 'fdr_bh',
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """多重比较校正
        
        Args:
            p_values: p值数组
            method: 校正方法，可选'bonferroni', 'fdr_bh', 'fdr_by', 'holm'
            alpha: 显著性水平
            
        Returns:
            Dict: 校正结果，包含corrected_p_values, significant_mask等信息
        """
        try:
            # 确保p_values是一维数组
            p_values = np.asarray(p_values).flatten()
            
            # 根据方法进行校正
            if method.lower() == 'bonferroni':
                # Bonferroni校正
                corrected_p = np.minimum(p_values * len(p_values), 1.0)
                significant = corrected_p < alpha
                logger.info(f"Bonferroni校正: {np.sum(significant)}/{len(p_values)}个显著结果")
                
            elif method.lower() == 'fdr_bh':
                # Benjamini-Hochberg FDR校正
                from statsmodels.stats.multitest import fdrcorrection
                significant, corrected_p = fdrcorrection(p_values, alpha=alpha, method='indep')
                logger.info(f"FDR-BH校正: {np.sum(significant)}/{len(p_values)}个显著结果")
                
            elif method.lower() == 'fdr_by':
                # Benjamini-Yekutieli FDR校正
                from statsmodels.stats.multitest import fdrcorrection
                significant, corrected_p = fdrcorrection(p_values, alpha=alpha, method='negcorr')
                logger.info(f"FDR-BY校正: {np.sum(significant)}/{len(p_values)}个显著结果")
                
            elif method.lower() == 'holm':
                # Holm-Bonferroni校正
                from statsmodels.stats.multitest import multipletests
                significant, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='holm')
                logger.info(f"Holm-Bonferroni校正: {np.sum(significant)}/{len(p_values)}个显著结果")
                
            else:
                logger.error(f"不支持的校正方法: {method}")
                return {
                    'success': False,
                    'error': f"不支持的校正方法: {method}"
                }
            
            return {
                'success': True,
                'corrected_p_values': corrected_p,
                'significant_mask': significant,
                'method': method,
                'alpha': alpha
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"多重比较校正失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def permutation_test(self, data1: np.ndarray, data2: np.ndarray, 
                        n_permutations: int = 1000,
                        statistic: str = 'mean',
                        tail: int = 0) -> Dict[str, Any]:
        """置换检验
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            n_permutations: 置换次数
            statistic: 统计量，可选'mean', 'median', 't-test'
            tail: 检验类型，0表示双尾，1表示右尾，-1表示左尾
            
        Returns:
            Dict: 置换检验结果，包含p_value, statistic_value等信息
        """
        try:
            # 确保数据是一维数组
            data1 = np.asarray(data1).flatten()
            data2 = np.asarray(data2).flatten()
            
            # 计算原始统计量
            if statistic.lower() == 'mean':
                # 均值差
                orig_stat = np.mean(data1) - np.mean(data2)
                stat_func = lambda x, y: np.mean(x) - np.mean(y)
                
            elif statistic.lower() == 'median':
                # 中位数差
                orig_stat = np.median(data1) - np.median(data2)
                stat_func = lambda x, y: np.median(x) - np.median(y)
                
            elif statistic.lower() == 't-test':
                # t检验
                orig_stat, _ = stats.ttest_ind(data1, data2, equal_var=False)
                stat_func = lambda x, y: stats.ttest_ind(x, y, equal_var=False)[0]
                
            else:
                logger.error(f"不支持的统计量: {statistic}")
                return {
                    'success': False,
                    'error': f"不支持的统计量: {statistic}"
                }
            
            # 合并数据
            pooled = np.concatenate([data1, data2])
            n1, n2 = len(data1), len(data2)
            n = n1 + n2
            
            # 置换检验
            perm_stats = np.zeros(n_permutations)
            
            for i in range(n_permutations):
                # 随机打乱数据
                perm_idx = np.random.permutation(n)
                perm_data1 = pooled[perm_idx[:n1]]
                perm_data2 = pooled[perm_idx[n1:]]
                
                # 计算置换统计量
                perm_stats[i] = stat_func(perm_data1, perm_data2)
            
            # 计算p值
            if tail == 0:  # 双尾
                p_value = np.mean(np.abs(perm_stats) >= np.abs(orig_stat))
            elif tail == 1:  # 右尾
                p_value = np.mean(perm_stats >= orig_stat)
            elif tail == -1:  # 左尾
                p_value = np.mean(perm_stats <= orig_stat)
            else:
                logger.error(f"不支持的检验类型: {tail}")
                return {
                    'success': False,
                    'error': f"不支持的检验类型: {tail}"
                }
            
            logger.info(f"置换检验结果: 原始统计量={orig_stat:.4f}, p值={p_value:.4f}")
            
            return {
                'success': True,
                'p_value': p_value,
                'statistic_value': orig_stat,
                'permutation_distribution': perm_stats,
                'method': f"permutation_{statistic}",
                'n_permutations': n_permutations
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"置换检验失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def cluster_based_permutation(self, data1: np.ndarray, data2: np.ndarray,
                               adjacency: Optional[np.ndarray] = None,
                               n_permutations: int = 1000,
                               threshold: float = 2.0,
                               tail: int = 0) -> Dict[str, Any]:
        """基于聚类的置换检验
        
        Args:
            data1: 第一组数据，形状为(n_observations, n_features)
            data2: 第二组数据，形状为(n_observations, n_features)
            adjacency: 邻接矩阵，形状为(n_features, n_features)，如果为None则假设所有特征都相邻
            n_permutations: 置换次数
            threshold: 聚类阈值
            tail: 检验类型，0表示双尾，1表示右尾，-1表示左尾
            
        Returns:
            Dict: 基于聚类的置换检验结果
        """
        try:
            # 确保数据是二维数组
            if data1.ndim == 1:
                data1 = data1.reshape(-1, 1)
            if data2.ndim == 1:
                data2 = data2.reshape(-1, 1)
            
            # 使用MNE的cluster_level_mne实现
            try:
                from mne.stats import permutation_cluster_test
                
                # 准备数据
                X = [data1, data2]
                
                # 执行基于聚类的置换检验
                T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
                    X, n_permutations=n_permutations, threshold=threshold,
                    tail=tail, adjacency=adjacency, n_jobs=1
                )
                
                # 找出显著的聚类
                significant_clusters = [i for i, p in enumerate(cluster_p_values) if p < 0.05]
                logger.info(f"基于聚类的置换检验: {len(significant_clusters)}/{len(clusters)}个显著聚类")
                
                return {
                    'success': True,
                    'T_obs': T_obs,
                    'clusters': clusters,
                    'cluster_p_values': cluster_p_values,
                    'significant_clusters': significant_clusters,
                    'method': 'cluster_permutation',
                    'n_permutations': n_permutations
                }
                
            except ImportError:
                logger.warning("MNE stats模块不可用，使用简化的基于聚类的置换检验实现")
                
                # 简化实现
                # 计算t统计量
                t_values = stats.ttest_ind(data1, data2, axis=0, equal_var=False)[0]
                
                # 创建邻接矩阵（如果未提供）
                n_features = data1.shape[1]
                if adjacency is None:
                    adjacency = np.ones((n_features, n_features)) - np.eye(n_features)
                
                # 找出超过阈值的点
                if tail == 0:  # 双尾
                    threshold_mask = np.abs(t_values) > threshold
                elif tail == 1:  # 右尾
                    threshold_mask = t_values > threshold
                elif tail == -1:  # 左尾
                    threshold_mask = t_values < -threshold
                
                # 找出聚类
                from scipy.sparse.csgraph import connected_components
                
                # 创建邻接图
                adjacency_graph = adjacency.copy()
                adjacency_graph[~threshold_mask, :] = 0
                adjacency_graph[:, ~threshold_mask] = 0
                
                # 找出连通分量
                n_components, labels = connected_components(adjacency_graph)
                
                # 计算每个聚类的统计量
                cluster_stats = np.zeros(n_components)
                for i in range(n_components):
                    if i == 0 and not threshold_mask[labels == 0].any():  # 跳过背景
                        continue
                    cluster_stats[i] = np.sum(t_values[labels == i])
                
                # 置换检验
                perm_cluster_stats = np.zeros((n_permutations, n_components))
                
                # 合并数据
                pooled = np.vstack([data1, data2])
                n1, n2 = data1.shape[0], data2.shape[0]
                n = n1 + n2
                
                for i in range(n_permutations):
                    # 随机打乱数据
                    perm_idx = np.random.permutation(n)
                    perm_data1 = pooled[perm_idx[:n1]]
                    perm_data2 = pooled[perm_idx[n1:]]
                    
                    # 计算t统计量
                    perm_t_values = stats.ttest_ind(perm_data1, perm_data2, axis=0, equal_var=False)[0]
                    
                    # 找出超过阈值的点
                    if tail == 0:  # 双尾
                        perm_threshold_mask = np.abs(perm_t_values) > threshold
                    elif tail == 1:  # 右尾
                        perm_threshold_mask = perm_t_values > threshold
                    elif tail == -1:  # 左尾
                        perm_threshold_mask = perm_t_values < -threshold
                    
                    # 创建邻接图
                    perm_adjacency_graph = adjacency.copy()
                    perm_adjacency_graph[~perm_threshold_mask, :] = 0
                    perm_adjacency_graph[:, ~perm_threshold_mask] = 0
                    
                    # 找出连通分量
                    perm_n_components, perm_labels = connected_components(perm_adjacency_graph)
                    
                    # 计算每个聚类的统计量
                    for j in range(perm_n_components):
                        if j == 0 and not perm_threshold_mask[perm_labels == 0].any():  # 跳过背景
                            continue
                        perm_cluster_stats[i, j] = np.sum(perm_t_values[perm_labels == j])
                
                # 计算p值
                cluster_p_values = np.zeros(n_components)
                for i in range(n_components):
                    if i == 0 and not threshold_mask[labels == 0].any():  # 跳过背景
                        continue
                    # 计算最大统计量的分布
                    max_perm_stats = np.max(perm_cluster_stats, axis=1)
                    # 计算p值
                    cluster_p_values[i] = np.mean(max_perm_stats >= cluster_stats[i])
                
                # 找出显著的聚类
                significant_clusters = [i for i, p in enumerate(cluster_p_values) if p < 0.05]
                logger.info(f"基于聚类的置换检验: {len(significant_clusters)}/{n_components}个显著聚类")
                
                return {
                    'success': True,
                    'T_obs': t_values,
                    'clusters': labels,
                    'cluster_p_values': cluster_p_values,
                    'significant_clusters': significant_clusters,
                    'method': 'cluster_permutation_simplified',
                    'n_permutations': n_permutations
                }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"基于聚类的置换检验失败: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }