# EEGSpark

基于 Docker 与 Apache Spark 的大规模 EEG 流式数据处理系统（本科毕业设计，2025）。

## 背景

单机 EEG 处理在数据量增大后存在吞吐低、延迟高、扩展性差的问题。本项目把 EEG 数据的接收、
分区化存储、并行分析与实时可视化拆成独立模块，用 Spark 做计算、用 Docker Compose 做部署。

## 结构

```
src/
  sender/        模拟数据发送端，读取 OpenNeuro EEG 数据并按通道回放
  receiver/      socket 接收端，聚合多路数据流
  processor/     核心算子
    eeg_processor.py        单条/批量预处理
    time_frequency.py       自适应时频分析
    source_localization.py  短数据源定位
    connectivity.py         并行连接性分析
    statistical.py          统计特征
    realtime_eeg_processor.py  实时处理循环
    pipeline.py             算子编排
  visualization/ Flask + WebSocket 实时波形/频谱/地形图
  common/eegrdd.py           EEGRDD：分区存储与索引
docker-compose.yml           多发送端 + 处理节点编排
docs/                        设计文档
```

## 运行

1. 准备数据：从 OpenNeuro 下载 ds005779 / ds005811 等数据集，放到 `eegspark/data/`（不随仓库分发）。
2. 构建镜像并启动发送端与 Spark 节点：

```bash
docker compose up -d
```

3. 本地启动处理主程序（需要 Python 3、PySpark、eventlet 等依赖）：

```bash
python src/main.py
```

4. 打开可视化页面：`http://localhost:5000`。

## 设计要点

- **EEGRDD**：按通道/时间窗分区，建立分区索引，避免每次分析都全量扫描。
- **多路接收**：多发送端按 subject 并发推送，接收端按流聚合，处理端用 Spark 并行执行。
- **算子拆分**：时频、源定位、连接性三类计算互不依赖，可分别扩展。
- **部署**：发送端、处理端、可视化端分别容器化，拓扑由 compose 文件描述。

## 实测结果（毕业设计环境）

| 项目 | 结果 |
| --- | --- |
| 数据规模 | 32 通道 / 1000 Hz 流式 EEG |
| 端到端延迟 | < 80 ms |
| 吞吐 | 120 MB/s |
| 容错率 | 99.2% |
| 数据读写效率 | 2.3× |
| 内存占用 | −45% |
| 并行算子加速 | 2.3–6.3× |

## 说明

- 仓库不含数据集与个人材料；数据需按 OpenNeuro 的许可自行获取。
- 代码用于研究与教学场景，未做生产级安全加固。
