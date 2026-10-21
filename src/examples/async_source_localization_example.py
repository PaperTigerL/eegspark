# 异步源定位示例
# 演示如何使用异步源定位功能，避免SVD计算阻塞主处理流程

import sys
import os
import time
import logging
import numpy as np
import mne

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入EEG处理器
from processor.eeg_processor import EEGProcessor
from common.eegrdd import EEGRDD

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AsyncSourceExample")


def create_test_eegrdd():
    """创建测试用的EEGRDD对象"""
    # 创建随机EEG数据 (19通道 x 5000个时间点)
    n_channels = 19
    n_times = 5000
    sfreq = 250  # 采样率

    # 生成随机数据
    data = np.random.randn(n_channels, n_times) * 10  # 微伏级别

    # 创建通道名称 (国际10-20系统)
    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "Pz",
        "P4",
        "T6",
        "O1",
        "O2",
    ]

    # 创建EEGRDD对象
    eegrdd = EEGRDD(
        voltage_data=data,
        channel_names=ch_names,
        sampling_rate=sfreq,
        channel_types=["eeg"] * n_channels,
    )

    return eegrdd


def source_localization_callback(result):
    """源定位计算完成后的回调函数"""
    if result["success"]:
        logger.info(f"源定位计算完成，方法: {result.get('method', 'unknown')}")
        logger.info(f"处理时间: {result.get('processing_time', 0):.2f}秒")

        # 如果有源时间曲线，可以进行进一步处理
        if "source_time_course" in result:
            stc = result["source_time_course"]
            logger.info(f"源时间曲线形状: {stc.data.shape}")
            logger.info(f"最大活动值: {np.max(stc.data)}")
    else:
        logger.error(f"源定位计算失败: {result.get('error')}")


def main():
    """主函数"""
    logger.info("开始异步源定位示例")

    # 创建EEG处理器
    processor = EEGProcessor()

    # 创建测试数据
    eegrdd = create_test_eegrdd()
    logger.info(
        f"创建了测试数据: {eegrdd.voltage_data.shape[0]}通道 x {eegrdd.voltage_data.shape[1]}个时间点"
    )

    # 转换为MNE Raw对象
    raw = processor.eegrdd_to_mne(eegrdd)
    logger.info(f"转换为MNE Raw对象: {len(raw.ch_names)}通道, {raw.info['sfreq']}Hz")

    # 应用电极位置
    raw = processor.apply_montage(raw, eegrdd)

    # 启动异步源定位计算
    logger.info("启动异步源定位计算...")
    result = processor.compute_source_localization(
        raw=raw,
        method="eLORETA",
        async_mode=True,  # 使用异步模式
        callback=source_localization_callback,  # 设置回调函数
    )

    if result["success"] and result.get("async", False):
        task_id = result["task_id"]
        logger.info(f"异步源定位任务已启动，任务ID: {task_id}")

        # 模拟主线程继续执行其他操作
        for i in range(10):
            logger.info(f"主线程继续执行其他操作... {i+1}/10")

            # 每隔一段时间检查任务状态
            if i % 3 == 0:
                status = processor.check_source_localization_status(task_id)
                logger.info(f"任务状态: {status.get('status', 'unknown')}")

                # 如果任务已完成，可以处理结果
                if status.get("status") != "processing":
                    if status.get("success", False):
                        logger.info("任务已成功完成")
                    else:
                        logger.error(f"任务失败: {status.get('error')}")
                    break

            # 模拟其他处理
            time.sleep(1)

        # 获取所有已完成的源定位结果
        all_results = processor.get_all_source_localization_results()
        logger.info(f"共有{len(all_results)}个已完成的源定位任务")

    # 对比同步模式
    logger.info("\n使用同步模式进行源定位计算...")
    sync_start = time.time()
    sync_result = processor.compute_source_localization(
        raw=raw, method="eLORETA", async_mode=False  # 使用同步模式
    )
    sync_time = time.time() - sync_start

    logger.info(f"同步模式源定位计算完成，耗时: {sync_time:.2f}秒")
    logger.info(f"计算结果: {'成功' if sync_result.get('success', False) else '失败'}")

    logger.info("异步源定位示例结束")


if __name__ == "__main__":
    main()
