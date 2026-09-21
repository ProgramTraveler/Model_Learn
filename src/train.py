from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# __file__ 表示当前 train.py 文件的位置。
# parent.parent 从 src/train.py 返回到项目根目录 scene_model。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "learning_scene_dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"


def create_data_loaders(batch_size: int = 8):
    """读取训练集和验证集，并创建分批加载器。"""

    # 所有图片统一调整为 64×64，再转换为 PyTorch Tensor。
    image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # ImageFolder 根据子文件夹名称自动生成类别标签。
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=image_transform,
    )
    validation_dataset = datasets.ImageFolder(
        root=VALIDATION_DIR,
        transform=image_transform,
    )

    # 训练集和验证集必须使用完全相同的类别编号。
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError("训练集与验证集的类别不一致")

    # 训练数据需要打乱；验证数据保持固定顺序。
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, train_dataset.classes


def create_model(number_of_classes: int):
    """创建一个用于 64×64 彩色图片分类的简单 CNN。"""

    model = nn.Sequential(
        # 输入形状：[批次数量, 3, 64, 64]
        # 使用 8 个卷积核，输出 8 张特征图。
        nn.Conv2d(
            in_channels=3,  # 有三个颜色通道
            out_channels=8,  # 有 8 个卷积核，产生 8 个特征图
            kernel_size=3,  # 每个卷积核观察 3x3 局部区域
            padding=1,  # 上下左右临时补一圈 0
        ),

        # 把负数变为 0，为模型加入非线性能力。
        nn.ReLU(),

        # 高度和宽度从 64×64 缩小为 32×32。
        nn.MaxPool2d(kernel_size=2),

        # 保留批次维度，把每张图片的特征展开成一排数字。
        # 从下标 1 开始，把后面的维度全部合并。
        nn.Flatten(start_dim=1),

        # 每张图片有 8×32×32 个特征。
        # 最终输出的分数数量等于数据集的类别数量。
        nn.Linear(
            in_features=8 * 32 * 32,
            out_features=number_of_classes,
        ),
    )

    return model


def main():
    """程序入口：检查环境并验证数据能否正确读取。"""

    print("PyTorch 版本：", torch.__version__)
    print("项目目录：", PROJECT_ROOT)

    # 提前检查目录，可以给出比 ImageFolder 更容易理解的错误信息。
    if not TRAIN_DIR.is_dir() or not VALIDATION_DIR.is_dir():
        raise FileNotFoundError(f"没有找到数据集目录：{DATASET_ROOT}")

    train_loader, validation_loader, class_names = create_data_loaders()

    print("类别名称：", class_names)
    print("训练批次数量：", len(train_loader))
    print("验证批次数量：", len(validation_loader))

    # 读取一个训练批次，确认图片和标签的形状。
    batch_images, batch_labels = next(iter(train_loader))
    print("一个批次的图片形状：", batch_images.shape)
    print("一个批次的标签形状：", batch_labels.shape)

    # 类别数量由 ImageFolder 读取到的文件夹数量决定。
    model = create_model(number_of_classes=len(class_names))

    # 目前只测试前向传播，不需要记录梯度。
    with torch.no_grad():
        batch_scores = model(batch_images)

    print("模型结构：")
    print(model)
    print("模型输出的类别分数形状：", batch_scores.shape)


# 只有直接运行 python src/train.py 时才调用 main()。
# 以后其他文件导入 train.py 时，不会自动开始训练。
if __name__ == "__main__":
    main()
