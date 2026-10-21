#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG API服务器模块 - 处理HTTP API请求

此模块确保eventlet的monkey_patch在所有导入之前执行
"""

# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import logging
import threading
import time
import hashlib
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import json
from functools import lru_cache
import queue

# 导入Flask相关模块 - 确保在创建应用上下文之前导入
from flask import Flask, render_template, request, jsonify, Response, current_app
from flask_socketio import SocketIO, emit

# 配置日志
# 使用日志目录初始化器确保日志目录和文件正确创建
from utils.log_directory_initializer import LogDirectoryInitializer

# 初始化日志目录
log_initializer = LogDirectoryInitializer()
log_dir = log_initializer.initialize()

if not log_dir:
    # 如果所有尝试都失败，使用默认目录
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"警告: 使用默认日志目录: {log_dir}")

# 确保API服务器日志文件存在
api_log_path = log_initializer.ensure_log_file_exists("api_server.log")
if not api_log_path:
    api_log_path = os.path.join(log_dir, "api_server.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)





class EEGAPIServer:
    """EEG API服务器 - 处理HTTP API请求"""

    def __init__(self, app, visualizer):
        """初始化API服务器

        Args:
            app: Flask应用实例
            visualizer: EEG可视化器实例
        """
        self.app = app
        self.visualizer = visualizer
        self.api_cache = {}
        self.api_cache_ttl = 0.3  # 缓存有效期（秒）
        self.api_cache_size_limit = 100  # 最大缓存数量

        # 注册路由
        self._register_routes()


    def _register_routes(self):
        """注册API路由"""
        # 主页路由已在flask_server.py中定义，这里不再重复注册
        # 避免'View function mapping is overwriting an existing endpoint function: index'错误

        # 获取数据源列表
        @self.app.route("/api/data_sources")
        def get_data_sources():
            try:
                source_ids = self.visualizer.get_data_source_ids()
                active_source = None

                # 获取活动数据源
                for source_id in source_ids:
                    source_data = self.visualizer.get_data_source(source_id)
                    if (
                        source_data
                        and source_id == self.visualizer.base_visualizer.active_source
                    ):
                        active_source = source_id
                        break

                # 构建数据源信息
                sources_info = []
                for source_id in source_ids:
                    source_data = self.visualizer.get_data_source(source_id)
                    if source_data:
                        sources_info.append(
                            {
                                "id": source_id,
                                "channels": len(source_data["channel_names"]),
                                "samples": source_data["data"].shape[1],
                                "sampling_rate": source_data["sampling_rate"],
                                "duration": source_data["data"].shape[1]
                                / source_data["sampling_rate"],
                                "is_active": source_id == active_source,
                                "metadata": source_data["metadata"],
                            }
                        )

                return jsonify(
                    {
                        "status": "success",
                        "data": {
                            "sources": sources_info,
                            "active_source": active_source,
                        },
                    }
                )
            except Exception as e:
                
                return (
                    jsonify(
                        {"status": "error", "message": f"获取数据源列表出错: {str(e)}"}
                    ),
                    500,
                )

        # 设置活动数据源
        @self.app.route("/api/set_active_source")
        def set_active_source():
            try:
                source_id = request.args.get("source_id")
                if not source_id:
                    return (
                        jsonify({"status": "error", "message": "缺少source_id参数"}),
                        400,
                    )

                success = self.visualizer.set_active_source(source_id)
                if success:
                    return jsonify(
                        {
                            "status": "success",
                            "message": f"已设置活动数据源为 {source_id}",
                        }
                    )
                else:
                    return (
                        jsonify(
                            {"status": "error", "message": f"数据源 {source_id} 不存在"}
                        ),
                        404,
                    )
            except Exception as e:
            
                return (
                    jsonify(
                        {"status": "error", "message": f"设置活动数据源出错: {str(e)}"}
                    ),
                    500,
                )

        # 获取通道名称
        @self.app.route("/api/channel_names")
        def get_channel_names():
            try:
                source_id = request.args.get("source_id")
                source_data = self.visualizer.get_data_source(source_id)

                if not source_data:
                    return (
                        jsonify(
                            {"status": "error", "message": f"数据源 {source_id} 不存在"}
                        ),
                        404,
                    )

                # 获取通道名称和分组
                channel_names = source_data["channel_names"]
                channel_groups = {}

                for channel in channel_names:
                    group = self.visualizer.base_visualizer._get_channel_group(channel)
                    if group not in channel_groups:
                        channel_groups[group] = []
                    channel_groups[group].append(channel)

                return jsonify(
                    {
                        "status": "success",
                        "data": {
                            "channel_names": channel_names,
                            "channel_groups": channel_groups,
                        },
                    }
                )
            except Exception as e:
               
                return (
                    jsonify(
                        {"status": "error", "message": f"获取通道名称出错: {str(e)}"}
                    ),
                    500,
                )

        # 可视化API
        @self.app.route("/api/visualize", methods=["POST"])
        def visualize():
            try:
                # 获取请求参数
                params = request.json
                if not params:
                    return jsonify({"status": "error", "message": "缺少请求参数"}), 400

                # 获取可视化类型
                viz_type = params.get("type", "time_domain")
                source_id = params.get("source_id")

                # 生成缓存键
                cache_key = self._generate_cache_key("visualize", **params)

                # 检查缓存
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    return jsonify({"status": "success", "data": cached_result})

                # 根据可视化类型调用相应的方法
                if viz_type == "time_domain":
                    # 时域可视化
                    start_time = params.get("start_time", 0)
                    duration = params.get("duration", 5.0)
                    baseline_correction = params.get("baseline_correction", True)
                    baseline_window = params.get("baseline_window", (-0.2, 0))
                    voltage_scale = params.get("voltage_scale")
                    time_markers = params.get("time_markers")
                    sliding_step = params.get("sliding_step", 1.0)
                    selected_channels = params.get("selected_channels")
                    show_ms_precision = params.get("show_ms_precision", True)
                    show_artifacts = params.get("show_artifacts", True)
                    show_erp_average = params.get("show_erp_average", False)
                    figsize = params.get("figsize", (12, 8))

                    img_base64 = self.visualizer.plot_time_domain(
                        start_time=start_time,
                        duration=duration,
                        baseline_correction=baseline_correction,
                        baseline_window=baseline_window,
                        voltage_scale=voltage_scale,
                        time_markers=time_markers,
                        sliding_step=sliding_step,
                        selected_channels=selected_channels,
                        show_ms_precision=show_ms_precision,
                        show_artifacts=show_artifacts,
                        show_erp_average=show_erp_average,
                        figsize=figsize,
                        source_id=source_id,
                    )

                elif viz_type == "frequency_domain":
                    # 频域可视化
                    freq_range = params.get("freq_range", (0, 50))
                    selected_channels = params.get("selected_channels")
                    show_bands = params.get("show_bands", True)
                    method = params.get("method", "welch")
                    log_scale = params.get("log_scale", False)
                    normalize = params.get("normalize", False)
                    figsize = params.get("figsize", (12, 8))

                    img_base64 = self.visualizer.plot_frequency_domain(
                        freq_range=freq_range,
                        selected_channels=selected_channels,
                        show_bands=show_bands,
                        method=method,
                        log_scale=log_scale,
                        normalize=normalize,
                        figsize=figsize,
                        source_id=source_id,
                    )

                elif viz_type == "topographic_map":
                    # 拓扑图可视化
                    time_window = params.get("time_window")
                    freq_band = params.get("freq_band")
                    selected_time = params.get("selected_time")
                    interpolation = params.get("interpolation", "cubic")
                    contours = params.get("contours", 6)
                    colormap = params.get("colormap", "RdBu_r")
                    show_names = params.get("show_names", True)
                    show_sensors = params.get("show_sensors", True)
                    figsize = params.get("figsize", (10, 8))

                    img_base64 = self.visualizer.plot_topographic_map(
                        time_window=time_window,
                        freq_band=freq_band,
                        selected_time=selected_time,
                        interpolation=interpolation,
                        contours=contours,
                        colormap=colormap,
                        show_names=show_names,
                        show_sensors=show_sensors,
                        figsize=figsize,
                        source_id=source_id,
                    )

                elif viz_type == "impedance_map":
                    # 阻抗图可视化
                    impedance_values = np.array(params.get("impedance_values", []))
                    channel_names = params.get("channel_names", [])
                    figsize = params.get("figsize", (10, 8))

                    if len(impedance_values) == 0 or len(channel_names) == 0:
                        return (
                            jsonify(
                                {"status": "error", "message": "缺少阻抗值或通道名称"}
                            ),
                            400,
                        )

                    img_base64 = self.visualizer.plot_impedance_map(
                        impedance_values=impedance_values,
                        channel_names=channel_names,
                        figsize=figsize,
                    )

                elif viz_type == "combined":
                    # 组合可视化
                    start_time = params.get("start_time", 0)
                    duration = params.get("duration", 5.0)
                    selected_channels = params.get("selected_channels")
                    freq_range = params.get("freq_range", (0, 50))
                    show_topography = params.get("show_topography", True)
                    figsize = params.get("figsize", (15, 10))

                    img_base64 = self.visualizer.plot_combined_visualization(
                        start_time=start_time,
                        duration=duration,
                        selected_channels=selected_channels,
                        freq_range=freq_range,
                        show_topography=show_topography,
                        figsize=figsize,
                        source_id=source_id,
                    )

                else:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": f"不支持的可视化类型: {viz_type}",
                            }
                        ),
                        400,
                    )

                # 缓存结果
                self._cache_result(cache_key, img_base64)

                return jsonify({"status": "success", "data": img_base64})

            except Exception as e:
           
                return (
                    jsonify({"status": "error", "message": f"可视化出错: {str(e)}"}),
                    500,
                )

    def _generate_cache_key(self, func_name: str, **kwargs) -> str:
        """生成缓存键

        Args:
            func_name: 函数名称
            **kwargs: 参数

        Returns:
            str: 缓存键
        """
        # 创建一个包含所有参数的字典
        key_dict = {"func": func_name}
        key_dict.update(kwargs)

        # 将字典转换为字符串并生成哈希值
        try:
            key_str = str(sorted([(k, str(v)) for k, v in key_dict.items()]))
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception as e:
        
            return hashlib.md5(f"{func_name}_{time.time()}".encode()).hexdigest()

    def _cache_result(self, key: str, result: str) -> None:
        """缓存结果

        Args:
            key: 缓存键
            result: 结果
        """
        # 添加到缓存，记录时间戳
        self.api_cache[key] = {"result": result, "timestamp": time.time()}

        # 如果缓存超过大小限制，删除最旧的项
        if len(self.api_cache) > self.api_cache_size_limit:
            oldest_key = min(
                self.api_cache.keys(), key=lambda k: self.api_cache[k]["timestamp"]
            )
            del self.api_cache[oldest_key]

    def _get_cached_result(self, key: str) -> Optional[str]:
        """获取缓存结果

        Args:
            key: 缓存键

        Returns:
            Optional[str]: 缓存结果，如果不存在或已过期则为None
        """
        if key in self.api_cache:
            cache_entry = self.api_cache[key]
            # 检查缓存是否过期
            if time.time() - cache_entry["timestamp"] < self.api_cache_ttl:

                return cache_entry["result"]
            else:
                # 删除过期缓存
                del self.api_cache[key]
               
        return None
