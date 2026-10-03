import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# 复用训练时的模型结构；train.py 的入口判断会阻止导入时启动训练。
from train import create_model

# 从当前脚本的位置找到项目根目录，不依赖运行命令时所在的目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best_model.pth"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "sample_scene.png"


def main():
    """查看并加载已保存的参数，再预测一张图片的类别。"""

    # argparse 读取命令行参数；nargs="?" 表示图片路径可以省略。
    parser = argparse.ArgumentParser(description="查看模型参数并预测一张场景图片")
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="图片路径；相对路径以当前工作目录为起点，默认使用项目中的 sample_scene.png",
    )
    args = parser.parse_args()
    # 展开路径中的 ~，再转换为绝对路径，方便确认实际读取的图片。
    image_path = args.image_path.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"没有找到图片文件：{image_path}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"没有找到模型参数文件：{MODEL_PATH}")

    # map_location="cpu" 把张量加载到 CPU，本步不需要显卡。
    # weights_only=True 使用受限加载方式，适合读取保存的 state_dict。
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # 新格式同时保存参数和类别顺序，分别取出后再使用。
    if "model_state_dict" in checkpoint:
        parameters = checkpoint["model_state_dict"]
        class_names = checkpoint["class_names"]
    else:
        # 兼容本项目前面保存的纯参数文件，仅沿用当时训练日志中的类别顺序。
        parameters = checkpoint
        class_names = ["city", "indoor", "nature"]
        print("读取旧格式：使用本项目原有类别顺序；其他数据集请重新训练保存新格式。")

    print("参数文件：", MODEL_PATH)
    # 每一项由名称和张量组成，例如 0.weight 对应第 0 层的权重。
    for name, value in parameters.items():
        print(name, value.shape)

    # 本项目第 4 层是分类层，权重的行数就是输出类别数量。
    # 从权重形状取得数量，再检查它与类别名称数量是否一致。
    number_of_classes = parameters["4.weight"].shape[0]
    if len(class_names) != number_of_classes:
        raise ValueError("类别名称数量与模型输出数量不一致")
    model = create_model(number_of_classes=number_of_classes)

    # 新模型最初使用随机参数；这里用文件中的参数替换它们。
    # strict=True 要求参数名称完整匹配，形状不匹配也会报错。
    model.load_state_dict(parameters, strict=True)
    # 加载参数不会自动切换模式，预测前需要切换到评估模式。
    model.eval()
    print("参数已成功加载到模型，模型已切换到评估模式。")

    # 图片预处理必须与训练时一致：调整为 64×64，再转换为张量。
    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    with Image.open(image_path) as image:
        # 与训练时 ImageFolder 的读取方式一致，统一为三个颜色通道。
        image_tensor = image_transform(image.convert("RGB"))

    # 单张图片是 [3, 64, 64]；增加批次维度后变成 [1, 3, 64, 64]。
    image_batch = image_tensor.unsqueeze(0)
    # eval() 切换模型模式，no_grad() 则关闭梯度记录，预测时两者配合使用。
    with torch.no_grad():
        scores = model(image_batch)
        # 沿类别维度把原始分数转换为概率，每张图片的各类别概率之和为 1。
        # 这里只用于展示预测结果，训练时交叉熵仍接收原始分数。
        probabilities = torch.softmax(scores, dim=1)
    # 沿类别维度取最高分的下标，再取出唯一一张图片的预测编号。
    predicted_index = scores.argmax(dim=1).item()
    print("预测图片：", image_path)
    print("输入形状：", image_batch.shape)
    print("类别顺序：", class_names)
    print("原始类别分数：", scores[0].tolist())
    # 下标 0 取出唯一一张图片，按训练时的类别顺序显示概率。
    # 这些数值表示模型的预测倾向，不保证等于实际判断正确的概率。
    for class_name, probability in zip(class_names, probabilities[0].tolist()):
        print(f"{class_name} 的预测概率：{probability:.2%}")
    print("预测类别：", class_names[predicted_index])


# 直接运行这个文件才查看参数并预测；不会启动训练，也不会修改参数文件。
if __name__ == "__main__":
    main()
