# 异步源定位功能说明

## 功能概述

本功能实现了EEG源定位计算的异步执行机制，解决了SVD计算可能不收敛导致主处理流程被阻塞的问题。通过将计算密集型的SVD操作放入后台线程执行，主线程可以继续处理其他任务，提高了系统的响应性和稳定性。

## 实现原理

1. **线程池设计**：使用Python的`threading`模块创建工作线程，通过队列机制管理任务和结果。
2. **异步计算**：将源定位计算任务提交到队列中，由工作线程异步执行。
3. **回调机制**：支持设置回调函数，当计算完成时自动调用，便于结果处理。
4. **错误处理**：针对SVD不收敛问题，实现了多级降级策略，尝试不同参数组合以提高计算成功率。

## 主要组件

### 1. AsyncSourceLocalizer类

位于`async_source_localization.py`文件中，负责异步执行源定位计算任务：

- 维护任务队列和结果队列
- 管理工作线程的生命周期
- 提供异步计算接口和结果查询接口
- 实现多级降级策略处理SVD不收敛问题

### 2. EEGProcessor类扩展

在`eeg_processor.py`中扩展了以下功能：

- `compute_source_localization`方法增加了`async_mode`参数，支持异步计算
- 新增`check_source_localization_status`方法，用于查询异步任务状态
- 新增`get_all_source_localization_results`方法，用于获取所有已完成任务的结果

## 使用方法

### 1. 启动异步源定位计算

```python
# 创建EEG处理器
processor = EEGProcessor()

# 启动异步源定位计算
result = processor.compute_source_localization(
    raw=raw,  # MNE Raw对象
    method='eLORETA',  # 源定位方法
    async_mode=True,  # 使用异步模式
    callback=my_callback_function  # 可选的回调函数
)

# 获取任务ID
if result['success'] and result.get('async', False):
    task_id = result['task_id']
    print(f"异步任务ID: {task_id}")
```

### 2. 查询任务状态

```python
# 查询任务状态
status = processor.check_source_localization_status(task_id)

# 判断任务是否完成
if status.get('status') != 'processing':
    if status.get('success', False):
        print("任务已成功完成")
        # 处理结果
        stc = status.get('source_time_course')
    else:
        print(f"任务失败: {status.get('error')}")
```

### 3. 获取所有已完成的任务结果

```python
# 获取所有已完成的源定位结果
all_results = processor.get_all_source_localization_results()
print(f"共有{len(all_results)}个已完成的源定位任务")
```

### 4. 设置回调函数

```python
def source_localization_callback(result):
    """源定位计算完成后的回调函数"""
    if result['success']:
        print(f"源定位计算完成，方法: {result.get('method', 'unknown')}")
        print(f"处理时间: {result.get('processing_time', 0):.2f}秒")
        
        # 处理结果
        if 'source_time_course' in result:
            stc = result['source_time_course']
            # 进一步处理...
    else:
        print(f"源定位计算失败: {result.get('error')}")

# 启动异步计算并设置回调
result = processor.compute_source_localization(
    raw=raw,
    method='eLORETA',
    async_mode=True,
    callback=source_localization_callback
)
```

## 示例代码

完整示例请参考`src/examples/async_source_localization_example.py`文件，该示例演示了：

1. 如何启动异步源定位计算
2. 如何在主线程继续执行其他操作
3. 如何查询任务状态
4. 如何使用回调函数处理结果
5. 与同步模式的对比

## 注意事项

1. 异步模式下，`compute_source_localization`方法会立即返回，返回值中包含任务ID而非计算结果
2. 任务结果需要通过`check_source_localization_status`方法或回调函数获取
3. 工作线程是守护线程，主程序退出时会自动结束
4. 对于大型数据集，建议使用异步模式以避免主线程阻塞