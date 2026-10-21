#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG发送器和接收器连接测试运行脚本

此脚本提供了一个简单的方式来运行测试，无需直接使用pytest命令。
"""

import os
import sys
import pytest

def main():
    """运行所有测试"""
    print("\n===== 开始测试EEG发送器和接收器连接 =====\n")
    
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 运行测试
    test_file = os.path.join(current_dir, 'test_sender_receiver.py')
    result = pytest.main(['-xvs', test_file])
    
    if result == 0:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败，请检查上述错误信息")
    
    return result

if __name__ == "__main__":
    sys.exit(main())