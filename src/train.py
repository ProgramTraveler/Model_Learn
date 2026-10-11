import csv
from pathlib import Path

# pyplot 用来把每轮记录的指标画成曲线。
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# __file__ 表示当前 train.py 文件的位置。
# parent.parent 从 src/train.py 返回到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "learning_scene_dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"


class NarrowRoadDataset(Dataset):
    """按 CSV 标注读取图片与二分类标签；编号含义待团队确认。"""

    def __init__(self, csv_path, image_root, transform=None):
        # 图片路径以这个目录为起点；在读取样本时使用。
        self.image_root = Path(image_root)
        # 保存预处理方法，后续读取图片时使用。
        self.transform = transform

        # CSV 示例列名为 image_path、label，最终按标注团队格式适配。
        # utf-8-sig 也能处理带 BOM 的 UTF-8 文件。
        with open(csv_path, newline="", encoding="utf-8-sig") as file:
            # 每行变成一个字典；CSV 中的 label 目前仍是字符串。
            self.records = list(csv.DictReader(file))

    def __len__(self):
        # 标注表中的记录数量就是数据集的样本数量。
        return len(self.records)


    def __getitem__(self, index):
        # index 从 0 开始；每条记录包含 image_path 和 label。
        record = self.records[index]
        image_path = self.image_root / record["image_path"]
        # CSV 的标签是字符串；交叉熵需要整数类别编号。
        label = int(record["label"])
        if label not in (0, 1):
            raise ValueError(f"二分类标签必须为 0 或 1：{image_path}，实际为 {label}")

        with Image.open(image_path) as source:
            # 统一为三个颜色通道，避免灰度图或透明图片的通道数量不一致。
            image = source.convert("RGB")
        if self.transform is not None:
            # 传入 create_image_transform() 时，输出为 [3, 64, 64] 张量。
            image = self.transform(image)

        return image, label


def create_image_transform():
    """训练和预测共用预处理，确保输入格式一致。"""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


def create_data_loaders(batch_size: int = 8):
    """读取训练集和验证集，并创建分批加载器。"""

    image_transform = create_image_transform()

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
    """程序入口：训练、验证、保存最佳模型并绘制曲线。"""

    # 固定 PyTorch 随机种子，便于在相同环境和流程下重复比较实验。
    # 必须放在模型初始化和数据打乱之前；不要在每轮训练中重新设置。
    random_seed = 42
    torch.manual_seed(random_seed)
    print("随机种子：", random_seed)

    print("PyTorch 版本：", torch.__version__)
    print("项目目录：", PROJECT_ROOT)

    # 提前检查目录，可以给出比 ImageFolder 更容易理解的错误信息。
    if not TRAIN_DIR.is_dir() or not VALIDATION_DIR.is_dir():
        raise FileNotFoundError(f"没有找到数据集目录：{DATASET_ROOT}")

    train_loader, validation_loader, class_names = create_data_loaders()

    print("类别名称：", class_names)
    print("训练批次数量：", len(train_loader))
    print("验证批次数量：", len(validation_loader))

    # 类别数量来自数据集；二分类数据接入后，这里会自动输出两个分数。
    model = create_model(number_of_classes=len(class_names))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 交叉熵接收原始类别分数，不需要提前调用 softmax。
    loss_function = nn.CrossEntropyLoss()
    print("模型结构：")
    print(model)

    # epoch 表示完整遍历一次训练集；这里让模型连续学习 10 轮。
    number_of_epochs = 10

    # 在轮次循环外创建列表，让各轮的指标一直保留在本次运行的内存中。
    training_loss_history = []  # 按轮次记录平均训练 loss。
    validation_loss_history = []  # 按轮次记录平均验证 loss。
    validation_accuracy_history = []  # 按轮次记录验证准确率。

    # 在循环外保留最佳指标；准确率从 -1 开始，确保首轮可以更新。
    best_validation_accuracy = -1.0
    # 这里记录最佳准确率对应的 loss，不是所有轮次中最低的 loss。
    best_validation_loss = float("inf")
    # 轮次从 1 开始，0 表示目前还没有完成任何一轮验证。
    best_epoch = 0
    # 最佳参数保存到项目根目录；之后出现更好结果时覆盖同一个文件。
    best_model_path = PROJECT_ROOT / "best_model.pth"

    # range 包含起点、不包含终点：range(1, 11) 依次产生 1 到 10。
    # 每次外层循环都执行下面缩进的完整流程：训练一轮，再验证一次。
    # 模型在循环外创建，因此下一轮会接着使用上一轮更新后的参数。
    for epoch_index in range(1, number_of_epochs + 1):
        # 显示当前轮次和总轮数，例如“第 1/10 轮”。
        print(f"\n第 {epoch_index}/{number_of_epochs} 轮")

        # 每一轮都先切换到训练模式，再遍历全部训练批次。
        model.train()

        # 累加这一轮所有样本的 loss，最后计算整轮的平均 loss。
        total_training_loss = 0.0
        total_training_samples = 0

        for training_images, training_labels in train_loader:
            # 每个批次开始前清除上一批次留下的梯度。
            optimizer.zero_grad()

            # 预测当前批次，并与正确标签比较。
            training_scores = model(training_images)
            training_loss = loss_function(training_scores, training_labels)

            # 先计算梯度，再用梯度更新模型参数。
            training_loss.backward()
            optimizer.step()

            # 把批次平均 loss 换算成总和，再累计样本数量。
            current_batch_size = training_labels.size(0)
            total_training_loss += training_loss.item() * current_batch_size
            total_training_samples += current_batch_size

        # 计算这一轮全部训练样本的平均 loss。
        average_training_loss = (total_training_loss / total_training_samples)

        # 训练结束后切换到验证模式，只检查模型而不更新参数。
        model.eval()

        # 记录验证集的 loss 总和、样本数量和预测正确数量。
        total_validation_loss = 0.0
        total_validation_samples = 0
        total_correct_predictions = 0

        # 验证阶段不需要计算梯度，可以减少内存和计算量。
        with torch.no_grad():
            for validation_images, validation_labels in validation_loader:
                # 使用当前模型预测这一批验证图片。
                validation_scores = model(validation_images)
                validation_loss = loss_function(
                    validation_scores,
                    validation_labels,
                )

                # 把当前批次的平均 loss 换算为 loss 总和。
                current_batch_size = validation_labels.size(0)
                total_validation_loss += (validation_loss.item() *
                                          current_batch_size)
                total_validation_samples += current_batch_size

                # 最大类别分数所在的下标就是模型的预测类别。
                validation_predictions = validation_scores.argmax(dim=1)

                # 比较预测类别与正确标签，累计预测正确数量。
                total_correct_predictions += (
                    validation_predictions == validation_labels).sum().item()

        # 计算这一轮的平均验证 loss 和验证准确率。
        average_validation_loss = (total_validation_loss /
                                   total_validation_samples)
        validation_accuracy = (total_correct_predictions /
                               total_validation_samples)

        # 优先比较准确率；准确率相同时，验证 loss 更低才算更好。
        is_better = (validation_accuracy > best_validation_accuracy
                     or (validation_accuracy == best_validation_accuracy
                         and average_validation_loss < best_validation_loss))
        # 两个指标必须一起更新，确保它们来自同一轮。
        if is_better:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = average_validation_loss
            # 只在指标变得更好时更新轮次，保持轮次与最佳指标对应。
            best_epoch = epoch_index
            # state_dict() 收集模型的参数和缓冲区，不包含模型结构。
            # 必须在当前最佳轮次立即写入文件，后续训练仍会继续改变参数。
            # 用字典一起保存参数和类别顺序，预测时才能正确解释类别编号。
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                }, best_model_path)
            print(f"已保存第 {best_epoch} 轮的最佳模型参数：{best_model_path}")

        # 每轮验证结束后追加一次，列表中的第一个数对应第 1 轮。
        training_loss_history.append(average_training_loss)
        validation_loss_history.append(average_validation_loss)
        validation_accuracy_history.append(validation_accuracy)

        print(f"平均训练 loss：{average_training_loss:.4f}")
        print(f"平均验证 loss：{average_validation_loss:.4f}")
        print(f"验证准确率：{validation_accuracy:.2%}")

    # 退出 epoch 循环后只执行一次，查看全部轮次的训练 loss。
    print("\n各轮训练 loss：", training_loss_history)
    # 列表中的相同位置对应同一轮，可以比较训练和验证 loss。
    print("各轮验证 loss：", validation_loss_history)
    # 直接打印列表时准确率仍是小数，例如 0.8 表示 80%。
    print("各轮验证准确率：", validation_accuracy_history)
    print(f"最佳指标所在轮次：第 {best_epoch} 轮")
    print(f"最佳验证准确率：{best_validation_accuracy:.2%}")
    print(f"最佳准确率对应的验证 loss：{best_validation_loss:.4f}")

    # 横轴依次是第 1 轮到第 10 轮，与历史列表中的记录一一对应。
    epoch_numbers = range(1, number_of_epochs + 1)
    # 创建画布，宽 8 英寸、高 5 英寸。
    plt.figure(figsize=(8, 5))
    # 将每轮训练 loss 连成曲线，圆点标出每轮的实际数值。
    plt.plot(epoch_numbers,
             training_loss_history,
             marker="o",
             label="Train loss")
    # 在同一张图上绘制验证 loss，方便与训练 loss 比较。
    plt.plot(epoch_numbers,
             validation_loss_history,
             marker="o",
             label="Validation loss")
    # 使用英文坐标标签，避免本机缺少中文字体时显示方框。
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # 横轴只显示实际训练轮次，并用图例区分两条曲线。
    plt.xticks(epoch_numbers)
    plt.legend()
    # 自动调整边距，避免坐标标签被裁掉。
    plt.tight_layout()
    # 将曲线保存到项目根目录，重新运行时会覆盖这张图。
    chart_path = PROJECT_ROOT / "training_loss.png"
    plt.savefig(chart_path)
    # 保存后关闭画布，释放绘图占用的资源。
    plt.close()
    # 显示图片路径，方便训练结束后打开查看。
    print("训练曲线已保存：", chart_path)


# 只有直接运行 python src/train.py 时才调用 main()。
# 以后其他文件导入 train.py 时，不会自动开始训练。
if __name__ == "__main__":
    main()
