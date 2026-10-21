import numpy as np
import mne
import logging
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AsyncSourceLocalization")


class AsyncSourceLocalizer:
    """异步EEG源定位器 - 实现SVD计算的异步执行，避免阻塞主处理流程"""

    def __init__(self):
        """初始化异步源定位器"""
        self.last_error = None
        self._task_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._worker_thread = None
        self._running = False
        self._callbacks = {}
        self._task_counter = 0
        self._start_worker()

    def _start_worker(self):
        """启动工作线程"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True
            )
            self._worker_thread.start()
            logger.info("异步源定位工作线程已启动")

    def _worker_loop(self):
        """工作线程主循环"""
        while self._running:
            try:
                # 尝试从队列获取任务，设置超时以便能够响应停止信号
                try:
                    task_id, task_type, args, kwargs = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                logger.info(f"开始处理异步源定位任务 {task_id}, 类型: {task_type}")
                start_time = time.time()

                # 根据任务类型执行相应的计算
                try:
                    if task_type == "inverse_operator":
                        result = self._compute_inverse_operator(*args, **kwargs)
                    elif task_type == "apply_inverse":
                        result = self._apply_inverse(*args, **kwargs)
                    else:
                        result = {
                            "success": False,
                            "error": f"未知的任务类型: {task_type}",
                        }

                    # 添加任务信息到结果
                    result["task_id"] = task_id
                    result["task_type"] = task_type
                    result["processing_time"] = time.time() - start_time

                    # 将结果放入结果队列
                    self._result_queue.put(result)

                    # 如果有回调函数，执行回调
                    if task_id in self._callbacks:
                        callback = self._callbacks.pop(task_id)
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"执行回调函数时出错: {e}")

                    logger.info(
                        f"异步源定位任务 {task_id} 完成，耗时: {result['processing_time']:.2f}秒"
                    )

                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "task_id": task_id,
                        "task_type": task_type,
                    }
                    self._result_queue.put(error_result)
                    logger.error(f"异步源定位任务 {task_id} 失败: {e}", exc_info=True)

                    # 如果有回调函数，执行回调
                    if task_id in self._callbacks:
                        callback = self._callbacks.pop(task_id)
                        try:
                            callback(error_result)
                        except Exception as callback_error:
                            logger.error(f"执行回调函数时出错: {callback_error}")

                # 标记任务完成
                self._task_queue.task_done()

            except Exception as e:
                logger.error(f"工作线程循环中发生错误: {e}", exc_info=True)

    def stop(self):
        """停止工作线程"""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            logger.info("异步源定位工作线程已停止")

    def _compute_inverse_operator(
        self,
        raw: mne.io.Raw,
        forward_model: mne.Forward,
        noise_cov: Optional[mne.Covariance] = None,
        loose: float = 0.2,
        depth: float = 0.8,
        fixed: bool = False,
    ) -> Dict[str, Any]:
        """计算反向算子（在工作线程中执行）

        Args:
            raw: MNE Raw对象
            forward_model: 前向模型
            noise_cov: 噪声协方差矩阵，如果为None则自动计算
            loose: loose参数
            depth: depth参数
            fixed: fixed参数

        Returns:
            Dict: 反向算子计算结果
        """
        try:
            # 如果未提供噪声协方差矩阵，则自动计算
            if noise_cov is None:
                logger.info("自动计算噪声协方差矩阵")
                try:
                    # 创建空数据段
                    empty_room_raw = mne.io.RawArray(
                        np.zeros_like(raw.get_data()), raw.info
                    )
                    noise_cov = mne.compute_raw_covariance(empty_room_raw)
                except Exception as e:
                    logger.error(f"计算噪声协方差矩阵失败: {e}")
                    return {"success": False, "error": f"计算噪声协方差矩阵失败: {e}"}

            # 尝试不同的参数组合计算反向算子
            try:
                # 首先尝试使用提供的参数
                logger.info(
                    f"尝试计算反向算子，参数: loose={loose}, depth={depth}, fixed={fixed}"
                )
                inverse_operator = mne.minimum_norm.make_inverse_operator(
                    raw.info,
                    forward_model,
                    noise_cov,
                    loose=loose,
                    depth=depth,
                    fixed=fixed,
                )
                logger.info("成功计算反向算子")
            except np.linalg.LinAlgError as e:
                if "SVD did not converge" in str(e):
                    logger.warning(f"SVD不收敛，尝试使用更稳定的参数")
                    # 尝试使用更稳定的参数
                    try:
                        logger.info(
                            "尝试使用更稳定的参数: loose=0.5, depth=None, fixed=False"
                        )
                        inverse_operator = mne.minimum_norm.make_inverse_operator(
                            raw.info,
                            forward_model,
                            noise_cov,
                            loose=0.5,
                            depth=None,
                            fixed=False,
                            rank="info",
                        )
                        logger.info("使用更稳定的参数成功计算反向算子")
                    except Exception as fallback_error:
                        # 尝试第三种方法 - 使用最简单的配置
                        try:
                            logger.info(
                                "第二种方法失败，尝试使用最简单的配置: loose=1.0, fixed=False, use_cps=False"
                            )
                            inverse_operator = mne.minimum_norm.make_inverse_operator(
                                raw.info,
                                forward_model,
                                noise_cov,
                                loose=1.0,
                                fixed=False,
                                use_cps=False,
                                rank=None,
                            )
                            logger.info("使用最简单的配置成功计算反向算子")
                        except Exception as third_error:
                            # 所有方法都失败
                            logger.error(
                                f"所有反向算子计算方法都失败: {str(third_error)}"
                            )
                            return {
                                "success": False,
                                "error": str(third_error),
                                "error_type": "SVD_convergence_all_methods_failed",
                                "original_error": str(e),
                                "fallback_error": str(fallback_error),
                                "third_error": str(third_error),
                            }
                else:
                    # 其他线性代数错误
                    raise

            return {"success": True, "inverse_operator": inverse_operator}

        except Exception as e:
            logger.error(f"计算反向算子失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _apply_inverse(
        self,
        raw: mne.io.Raw,
        inverse_operator: mne.minimum_norm.InverseOperator,
        method: str = "MNE",
        lambda2: float = 1.0 / 3.0,
        pick_ori: Optional[str] = None,
    ) -> Dict[str, Any]:
        """应用反向算子进行源定位（在工作线程中执行）

        Args:
            raw: MNE Raw对象
            inverse_operator: 反向算子
            method: 反向方法，可选'MNE', 'dSPM', 'sLORETA', 'eLORETA'
            lambda2: 正则化参数
            pick_ori: 方向选择

        Returns:
            Dict: 源定位结果
        """
        try:
            # 尝试应用反向算子
            try:
                logger.info(f"应用反向算子，方法: {method}, lambda2={lambda2}")
                stc = mne.minimum_norm.apply_inverse_raw(
                    raw,
                    inverse_operator,
                    lambda2=lambda2,
                    method=method,
                    pick_ori=pick_ori,
                )
                logger.info("成功应用反向算子")
            except np.linalg.LinAlgError as e:
                if "SVD did not converge" in str(e):
                    # 尝试使用更简单的方法和更高的正则化
                    try:
                        logger.info("SVD不收敛，尝试使用MNE方法和更高的正则化参数")
                        lambda2_higher = lambda2 * 10
                        stc = mne.minimum_norm.apply_inverse_raw(
                            raw,
                            inverse_operator,
                            lambda2=lambda2_higher,
                            method="MNE",
                            pick_ori=None,
                        )
                        logger.info("使用备选方法成功计算源定位")

                        return {
                            "success": True,
                            "source_time_course": stc,
                            "method": "MNE (fallback)",
                            "original_method": method,
                            "lambda2": lambda2_higher,
                            "warning": "由于SVD不收敛，使用了MNE方法和更高的正则化参数",
                        }
                    except Exception as fallback_error:
                        # 尝试第三种方法 - 使用极高的正则化
                        try:
                            logger.info(
                                "第二种方法失败，尝试使用极高的正则化参数: lambda2=5.0"
                            )
                            lambda2_extreme = 5.0
                            stc = mne.minimum_norm.apply_inverse_raw(
                                raw,
                                inverse_operator,
                                lambda2=lambda2_extreme,
                                method="MNE",
                                pick_ori=None,
                            )
                            logger.info("使用极高的正则化参数成功计算源定位")

                            return {
                                "success": True,
                                "source_time_course": stc,
                                "method": "MNE (extreme fallback)",
                                "original_method": method,
                                "lambda2": lambda2_extreme,
                                "warning": "由于SVD不收敛，使用了MNE方法和极高的正则化参数",
                            }
                        except Exception as third_error:
                            # 所有方法都失败
                            logger.error(f"所有源定位方法都失败: {str(third_error)}")
                            return {
                                "success": False,
                                "error": str(third_error),
                                "error_type": "SVD_convergence_all_methods_failed",
                                "original_error": str(e),
                                "fallback_error": str(fallback_error),
                                "third_error": str(third_error),
                            }
                else:
                    # 其他线性代数错误
                    raise

            return {
                "success": True,
                "source_time_course": stc,
                "method": method,
                "lambda2": lambda2,
            }

        except Exception as e:
            logger.error(f"应用反向算子失败: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def compute_inverse_operator_async(
        self,
        raw: mne.io.Raw,
        forward_model: mne.Forward,
        noise_cov: Optional[mne.Covariance] = None,
        loose: float = 0.2,
        depth: float = 0.8,
        fixed: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> int:
        """异步计算反向算子

        Args:
            raw: MNE Raw对象
            forward_model: 前向模型
            noise_cov: 噪声协方差矩阵，如果为None则自动计算
            loose: loose参数
            depth: depth参数
            fixed: fixed参数
            callback: 计算完成后的回调函数

        Returns:
            int: 任务ID
        """
        # 确保工作线程在运行
        self._start_worker()

        # 创建任务ID
        task_id = self._task_counter
        self._task_counter += 1

        # 如果提供了回调函数，保存它
        if callback is not None:
            self._callbacks[task_id] = callback

        # 将任务添加到队列
        self._task_queue.put(
            (
                task_id,
                "inverse_operator",
                (raw, forward_model),
                {
                    "noise_cov": noise_cov,
                    "loose": loose,
                    "depth": depth,
                    "fixed": fixed,
                },
            )
        )

        logger.info(f"已添加异步反向算子计算任务 {task_id} 到队列")
        return task_id

    def apply_inverse_async(
        self,
        raw: mne.io.Raw,
        inverse_operator: mne.minimum_norm.InverseOperator,
        method: str = "MNE",
        lambda2: float = 1.0 / 3.0,
        pick_ori: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> int:
        """异步应用反向算子进行源定位

        Args:
            raw: MNE Raw对象
            inverse_operator: 反向算子
            method: 反向方法，可选'MNE', 'dSPM', 'sLORETA', 'eLORETA'
            lambda2: 正则化参数
            pick_ori: 方向选择
            callback: 计算完成后的回调函数

        Returns:
            int: 任务ID
        """
        # 确保工作线程在运行
        self._start_worker()

        # 创建任务ID
        task_id = self._task_counter
        self._task_counter += 1

        # 如果提供了回调函数，保存它
        if callback is not None:
            self._callbacks[task_id] = callback

        # 将任务添加到队列
        self._task_queue.put(
            (
                task_id,
                "apply_inverse",
                (raw, inverse_operator),
                {"method": method, "lambda2": lambda2, "pick_ori": pick_ori},
            )
        )

        logger.info(f"已添加异步源定位任务 {task_id} 到队列")
        return task_id

    def get_result(
        self, task_id: int, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """获取指定任务的结果

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒），如果为None则一直等待

        Returns:
            Dict: 任务结果，如果超时则返回None
        """
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            # 检查结果队列
            try:
                result = self._result_queue.get(block=False)
                self._result_queue.task_done()

                # 如果是我们要找的任务，返回结果
                if result.get("task_id") == task_id:
                    return result

                # 否则，将结果放回队列
                self._result_queue.put(result)

                # 短暂休眠以避免CPU占用过高
                time.sleep(0.1)
            except queue.Empty:
                # 队列为空，短暂休眠
                time.sleep(0.1)

        # 超时
        return None

    def get_all_results(self) -> List[Dict[str, Any]]:
        """获取所有已完成任务的结果

        Returns:
            List[Dict]: 所有已完成任务的结果
        """
        results = []
        while not self._result_queue.empty():
            try:
                result = self._result_queue.get(block=False)
                results.append(result)
                self._result_queue.task_done()
            except queue.Empty:
                break
        return results

    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态

        Returns:
            Dict: 队列状态信息
        """
        return {
            "pending_tasks": self._task_queue.qsize(),
            "completed_tasks": self._result_queue.qsize(),
            "total_tasks_created": self._task_counter,
        }

    def __del__(self):
        """析构函数，确保工作线程停止"""
        self.stop()
