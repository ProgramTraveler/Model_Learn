from pathlib import Path

# pyplot 用来把每轮记录的指标画成曲线。
import matplotlib.pyplot as plt
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

    # SGD 根据参数的梯度更新模型；学习率决定每次更新的步长。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 清除可能保留的旧梯度。本例只运行一次，仍按标准训练顺序书写。
    optimizer.zero_grad()

    # 保存更新前的分类层权重，稍后检查整层是否发生变化。
    weights_before = model[4].weight.detach().clone()

    # 为反向传播保留计算过程，因此这里不能使用 torch.no_grad()。
    batch_scores = model(batch_images)

    # 交叉熵损失比较原始类别分数与正确标签。
    # 它会在内部处理概率，因此这里不要先调用 softmax。
    loss_function = nn.CrossEntropyLoss()
    batch_loss = loss_function(batch_scores, batch_labels)

    print("模型结构：")
    print(model)
    print("模型输出的类别分数形状：", batch_scores.shape)
    print("这个批次的正确标签：", batch_labels)
    print("这个批次的平均 loss：", batch_loss.item())

    # 下标 0 指同一个批次中的第一张图片。
    # 它的三个分数和正确标签必须按相同下标配对。
    first_scores = batch_scores[0]
    first_label = batch_labels[0].item()
    first_prediction = first_scores.argmax().item()

    print("第一张图片的类别分数：", first_scores)
    print("第一张图片的预测类别：", class_names[first_prediction])
    print("第一张图片的正确类别：", class_names[first_label])

    # 从 loss 反向计算每个可训练参数的梯度。
    # 这一步只计算调整方向；还没有改变模型参数。
    batch_loss.backward()

    # 第 0 层是卷积层，第 4 层是最终分类层。
    print("卷积层权重的梯度形状：", model[0].weight.grad.shape)
    print("分类层权重的梯度形状：", model[4].weight.grad.shape)

    # 根据刚算出的梯度更新全部可训练参数；当前只更新一次。
    optimizer.step()

    # 单个权重可能因对应梯度为 0 而不变，所以比较整层最大变化量。
    largest_weight_change = (model[4].weight.detach() -
                             weights_before).abs().max().item()

    print("分类层权重的最大变化量：", largest_weight_change)

    # 用更新后的模型重新预测同一批图片，以便比较更新前后的 loss。
    # 这里只检查结果，不需要记录梯度。
    with torch.no_grad():
        updated_scores = model(batch_images)
        updated_loss = loss_function(updated_scores, batch_labels)

    print("更新前 loss：", batch_loss.item())
    print("更新后 loss：", updated_loss.item())

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
        is_better = (
            validation_accuracy > best_validation_accuracy
            or (
                validation_accuracy == best_validation_accuracy
                and average_validation_loss < best_validation_loss
            )
        )
        # 两个指标必须一起更新，确保它们来自同一轮。
        if is_better:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = average_validation_loss
            # 只在指标变得更好时更新轮次，保持轮次与最佳指标对应。
            best_epoch = epoch_index
            # state_dict() 收集模型的参数和缓冲区，不包含模型结构。
            # 必须在当前最佳轮次立即写入文件，后续训练仍会继续改变参数。
            # 用字典一起保存参数和类别顺序，预测时才能正确解释类别编号。
            torch.save({
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
    plt.plot(epoch_numbers, training_loss_history, marker="o", label="Train loss")
    # 在同一张图上绘制验证 loss，方便与训练 loss 比较。
    plt.plot(epoch_numbers, validation_loss_history, marker="o", label="Validation loss")
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
