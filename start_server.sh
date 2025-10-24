#!/bin/bash

# 创新网络可视化 - HTTP服务器启动脚本
# 
# 使用方法:
#   chmod +x start_server.sh
#   ./start_server.sh [端口号]
#
# 默认端口: 8000

PORT=${1:-8000}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"

echo "=========================================="
echo "🚀 启动创新网络可视化服务器"
echo "=========================================="
echo ""
echo "📁 服务目录: $RESULTS_DIR"
echo "🔌 端口号: $PORT"
echo ""

# 检查results目录是否存在
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 错误: $RESULTS_DIR 目录不存在"
    echo "   请先运行 PYTHONPATH=src python -m innovation_platform.innovation_resolution 生成结果文件"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  警告: 端口 $PORT 已被占用"
    echo "   尝试使用其他端口: ./start_server.sh 8080"
    exit 1
fi

# 启动服务器
cd "$RESULTS_DIR"
echo "✅ 服务器启动成功！"
echo ""
echo "📊 访问可视化页面："
echo "   🌐 3D交互式网络图: http://localhost:$PORT/innovation_network_tufte_3D.html"
echo "   📈 创新统计图:      http://localhost:$PORT/innovation_stats_tufte.png"
echo "   👥 顶级组织图:      http://localhost:$PORT/top_organizations_tufte.png"
echo ""
echo "⏹️  按 Ctrl+C 停止服务器"
echo "=========================================="
echo ""

python -m http.server $PORT
