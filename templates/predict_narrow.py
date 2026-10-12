"""单图预测入口：训练完成后，修改配置区并运行

    python templates/predict_narrow.py

这个文件调用 src/inspect_parameters.py，输出类别顺序、各类分数和预测类别。
"""

# 处理配置中的图片和模型文件路径。
from pathlib import Path
# 调用已有的预测脚本。
import subprocess
# 获取当前 Python 解释器，确保使用同一环境。
import sys

# 根据模板位置找到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ===== 每次预测只需检查这里 =====
# 填写训练时使用的实验名。
EXPERIMENT = "experiment_01"  # 须与训练模板中的名称相同。
# 改为实际图片路径；可以是项目外的绝对路径。
# 替换为实际要预测的图片路径。
IMAGE_PATH = PROJECT_ROOT / "narrow_road_dataset" / "images" / "待预测图片.jpg"
# 从与 EXPERIMENT 同名的训练结果目录读取模型。
# 指向该实验训练得到的最佳模型。
MODEL_PATH = PROJECT_ROOT / "outputs" / "narrow_road" / EXPERIMENT / "best_model.pth"
# ===== 配置区结束 =====


# 检查文件并调用正式的预测入口。
def main():
    # 先确认待预测图片存在。
    if not IMAGE_PATH.is_file():
        # 图片路径错误时给出修改提示。
        raise SystemExit(f"请先在配置区填写真实图片路径：{IMAGE_PATH}")
    # 确认训练产出的模型文件存在。
    if not MODEL_PATH.is_file():
        # 缺少模型时提示先完成训练。
        raise SystemExit(f"找不到模型：{MODEL_PATH}。请先完成对应实验的训练。")
    # 使用当前虚拟环境运行预测脚本，避免调用系统 Python。
    # 组合预测命令、图片路径和模型路径。
    command = [
        # 用当前虚拟环境运行正式预测脚本。
        sys.executable, str(PROJECT_ROOT / "src" / "inspect_parameters.py"),
        # 把图片和最佳模型传给预测脚本。
        str(IMAGE_PATH), "--model", str(MODEL_PATH),
    ]
    # 执行预测；失败时把错误传回终端。
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


# 直接运行模板时才执行预测。
if __name__ == "__main__":
    # 启动预测主流程。
    main()
