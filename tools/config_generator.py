#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置生成器工具

帮助生成不同场景的命令行配置。
"""

def print_config(name: str, command: str, description: str):
    """打印配置信息"""
    print(f"\n{'=' * 70}")
    print(f"📝 {name}")
    print(f"{'=' * 70}")
    print(f"\n描述: {description}\n")
    print("命令:")
    print(f"  {command}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🛠️  Innovation Resolution 配置生成器")
    print("=" * 70)
    print("\n选择一个预设配置:\n")
    
    configs = {
        "1": {
            "name": "快速测试",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method kmeans --n-clusters 300 --skip-visualization --skip-eval --quiet",
            "description": "用于快速测试，跳过耗时步骤"
        },
        "2": {
            "name": "完整分析",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method hdbscan --min-cluster-size 3 --top-n 20 --output-dir ./detailed_results --verbose --print-summary",
            "description": "完整的分析流程，包含详细输出"
        },
        "3": {
            "name": "HDBSCAN 调优",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method hdbscan --min-cluster-size 4 --metric euclidean --output-dir ./hdbscan_tuned",
            "description": "使用欧氏距离和较大的最小聚类大小"
        },
        "4": {
            "name": "K-Means 聚类",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method kmeans --n-clusters 400 --output-dir ./kmeans_results",
            "description": "使用 K-Means 算法，400 个聚类"
        },
        "5": {
            "name": "生产环境",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --cache-path ./prod_cache.json --top-n 15 --max-iter 1500 --output-dir ./prod_results --skip-visualization --quiet",
            "description": "优化的生产环境配置"
        },
        "6": {
            "name": "调试模式",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --steps load cluster --no-cache --verbose",
            "description": "只执行加载和聚类，用于调试"
        },
        "7": {
            "name": "重新分析",
            "command": "PYTHONPATH=src python -m innovation_platform.innovation_resolution --steps analyze visualize export --output-dir ./reanalysis",
            "description": "跳过数据加载和聚类，只重新分析"
        },
        "8": {
            "name": "比较实验",
            "command": """# HDBSCAN
PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method hdbscan --output-dir ./exp_hdbscan

# K-Means  
PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method kmeans --n-clusters 400 --output-dir ./exp_kmeans

# Agglomerative
PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method agglomerative --n-clusters 400 --output-dir ./exp_agglomerative""",
            "description": "运行多个不同聚类方法的实验"
        }
    }
    
    # 打印选项
    for key, config in configs.items():
        print(f"  {key}. {config['name']}")
    
    print(f"\n  0. 自定义配置")
    print(f"  q. 退出\n")
    
    # 获取用户输入
    choice = input("请选择 (1-8, 0, q): ").strip()
    
    if choice == 'q':
        print("\n再见！\n")
        return
    
    if choice in configs:
        config = configs[choice]
        print_config(config['name'], config['command'], config['description'])
        
        # 询问是否执行
        execute = input("是否立即执行此命令? (y/n): ").strip().lower()
        if execute == 'y':
            print("\n执行中...\n")
            import os
            os.system(config['command'].split('\n')[0])  # 只执行第一行命令
    
    elif choice == '0':
        print("\n" + "=" * 70)
        print("🔧 自定义配置生成器")
        print("=" * 70)
        print()
        
        # 聚类方法
        print("1. 选择聚类方法:")
        print("   1) hdbscan (默认)")
        print("   2) kmeans")
        print("   3) agglomerative")
        print("   4) spectral")
        method_choice = input("   选择 (1-4, 默认 1): ").strip() or "1"
        
        methods = {
            "1": ("hdbscan", "--min-cluster-size 2"),
            "2": ("kmeans", "--n-clusters 450"),
            "3": ("agglomerative", "--n-clusters 450"),
            "4": ("spectral", "--n-clusters 450")
        }
        method, method_params = methods.get(method_choice, methods["1"])
        
        # 输出目录
        output_dir = input("\n2. 输出目录 (默认 ./results): ").strip() or "./results"
        
        # 缓存
        cache = input("\n3. 启用缓存? (y/n, 默认 y): ").strip().lower() != 'n'
        cache_param = "" if cache else "--no-cache"
        
        # 详细输出
        verbose = input("\n4. 详细输出? (y/n, 默认 n): ").strip().lower() == 'y'
        verbose_param = "--verbose" if verbose else ""
        
        # 跳过步骤
        skip_viz = input("\n5. 跳过可视化? (y/n, 默认 n): ").strip().lower() == 'y'
        skip_viz_param = "--skip-visualization" if skip_viz else ""
        
        skip_eval = input("6. 跳过评估? (y/n, 默认 n): ").strip().lower() == 'y'
        skip_eval_param = "--skip-eval" if skip_eval else ""
        
        # 生成命令
        command_parts = [
            "PYTHONPATH=src python -m innovation_platform.innovation_resolution",
            f"--clustering-method {method}",
            method_params,
            f"--output-dir {output_dir}",
            cache_param,
            verbose_param,
            skip_viz_param,
            skip_eval_param
        ]
        
        command = " ".join(filter(None, command_parts))
        
        print_config("自定义配置", command, "根据您的选择生成的配置")
        
        # 询问是否执行
        execute = input("是否立即执行此命令? (y/n): ").strip().lower()
        if execute == 'y':
            print("\n执行中...\n")
            import os
            os.system(command)
    
    else:
        print("\n无效的选择。\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消。\n")
    except Exception as e:
        print(f"\n错误: {e}\n")
