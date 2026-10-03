# PyTorch 场景图像分类训练模板

本模板根据本项目 `src/train.py` 整理，供以后构建类似的场景识别模型时参考。适用于**一张图片对应一个类别**的监督分类任务，例如把场景图片分成室内、街道、树林。

这里固定的是训练流程。数据、类别和模型结构需要根据任务调整；它不是适合所有场景的通用最优模型，也不适用于直接输出物体位置的目标检测任务。

## 1. 使用方式与数据准备

代码按保存到项目的 `src/train.py` 设计，因此 `Path(__file__).resolve().parent.parent` 指向项目根目录。当前学习文件保留原样；下面是从现有文件提炼出的可复用版本。

模板沿用现有 CNN、CPU 计算、SGD 和 10 轮训练，省略了正式训练前“单批次更新、检查梯度、比较更新前后权重”的教学演示。这样不会在 10 轮训练之前额外更新一次参数。

数据目录示例（类别名称可以更换）：

```text
scene_model/
├── src/
│   └── train.py
└── learning_scene_dataset/
    ├── train/
    │   ├── indoor/
    │   ├── street/
    │   └── forest/
    └── validation/
        ├── indoor/
        ├── street/
        └── forest/
```

每个类别文件夹放入该类别的图片，训练集和验证集都必须有图片，并使用相同的类别文件夹名称。`ImageFolder` 自动建立类别编号；不要自行假设编号顺序，以返回的 `class_names` 为准。

训练与验证应使用分开的图片。同一张图片及其近似副本不要同时放入两边，否则验证结果可能过于乐观。

## 2. 完整模板代码

代码各阶段都附有中文注释。已有环境需要安装 `torch`、`torchvision` 和 `matplotlib`；本项目可以沿用现有虚拟环境。

```python
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
    """创建模型，并进行多轮训练和验证。"""

    # 开始训练前确认数据集目录存在。
    if not TRAIN_DIR.is_dir() or not VALIDATION_DIR.is_dir():
        raise FileNotFoundError(f"没有找到数据集目录：{DATASET_ROOT}")

    # 每批读取 8 张图片；类别名称由数据集子文件夹自动确定。
    train_loader, validation_loader, class_names = create_data_loaders(batch_size=8)
    print("类别名称：", class_names)

    # 模型只创建一次，各轮继续使用上一轮更新后的参数。
    model = create_model(number_of_classes=len(class_names))

    # 交叉熵接收原始类别分数和整数标签，不要提前调用 softmax。
    loss_function = nn.CrossEntropyLoss()

    # 优化器只创建一次，绑定当前模型参数，使用学习率 0.01。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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


# 直接运行本文件时开始训练；导入本文件时不自动执行。
if __name__ == "__main__":
    main()
```

## 3. 换一个识别场景时，修改哪里

| 内容 | 当前设置 | 如何调整 |
| --- | --- | --- |
| 数据路径 | `learning_scene_dataset` | 修改 `DATASET_ROOT`，指向新数据集 |
| 类别 | 根据子文件夹自动读取 | 更换两份数据集的类别文件夹，输出类别数自动跟随 |
| 批次大小 | `batch_size=8` | 在 `main()` 创建加载器时修改 |
| 训练轮数 | `number_of_epochs = 10` | 根据训练和验证表现调整 |
| 学习率 | `lr=0.01` | 在创建 SGD 时调整，过大可能导致训练不稳定 |
| 输入图片尺寸 | `64×64` | 修改 `Resize`，并同步修改全连接层的 `in_features` |
| 模型结构 | 一层卷积，8 个特征通道 | 在 `create_model()` 中调整，并检查后续层的输入形状 |

当前形状变化为：`[批次, 3, 64, 64] → [批次, 8, 64, 64] → [批次, 8, 32, 32] → [批次, 8192] → [批次, 类别数]`。因此不能只改图片大小而不检查 `8 * 32 * 32`。

## 4. 固定流程与放置位置

| 位置 | 操作 | 原因 |
| --- | --- | --- |
| epoch 循环之前 | 创建加载器、模型、损失函数和优化器 | 模型参数持续学习，不每轮重新初始化 |
| epoch 循环之前 | 初始化历史列表、最佳指标及保存路径 | 跨轮次保留记录 |
| 每轮训练开始 | `model.train()`，清零训练统计 | 从上一轮验证模式切回训练模式，只统计本轮 |
| 每个训练批次 | `zero_grad → 预测 → loss → backward → step` | 清除旧梯度，计算本批梯度，再更新参数 |
| 每个训练批次末尾 | 累计 `平均 loss × 实际批次大小` 和样本数 | 正确处理最后一个批次数量不足的情况 |
| 训练批次循环结束 | 计算整轮平均训练 loss | 总 loss 除以总样本数 |
| 每轮验证开始 | `model.eval()`，清零验证统计 | 检查本轮训练后的模型表现 |
| 验证代码块 | `torch.no_grad()`，遍历验证集 | 预测并统计指标，不记录反向传播所需过程 |
| 每轮验证结束 | 计算平均验证 loss 和准确率，追加到历史列表 | 观察并保留本轮验证结果 |
| 每轮指标计算之后 | 判断 `is_better`，更新最佳指标和轮次，并保存参数 | 准确率优先，同分比较 loss；完全相同则保留较早轮次 |
| epoch 循环之后 | 输出最佳指标并保存 loss 曲线 | 汇总本次训练结果 |

### 今天学到的关键区别

- **权重与梯度**：权重保留学习成果；梯度指导本次更新。`zero_grad()` 清除梯度，不清除权重。
- **反向传播与更新**：`backward()` 计算梯度；普通 SGD 的 `step()` 按“新参数 = 旧参数 − 学习率 × 梯度”更新参数。
- **模式与梯度记录**：`eval()` 切换某些层的行为，不会自动关闭梯度记录；`no_grad()` 关闭梯度记录，不会切换模型模式。当前 CNN 的各层在 train/eval 模式下行为相同，但加入 Dropout 或 BatchNorm 后就有区别。
- **验证阶段**：仍可以预测和计算 loss，但不调用 `backward()` 和 `step()`。`no_grad()` 是省去不必要记录的做法，不是验证能运行的硬性条件。
- **梯度累积**：本模板每批更新一次，所以每批清除梯度。有意累积梯度时，需要另行设计多个批次后再更新的流程，并合理缩放 loss。
- **训练 loss**：累计各批次更新前算出的 loss，反映一轮中的学习过程；验证 loss 使用这一轮训练结束后的同一套参数。
- **准确率**：`argmax(dim=1)` 选出每张图片分数最高的类别，再用“预测正确数量 / 总样本数”计算准确率。

## 5. 运行与结果

如果将模板用于 `src/train.py`，在项目根目录运行：

```bash
# 激活本项目已有的虚拟环境。
source .venv/bin/activate

# 启动训练，每轮输出训练 loss、验证 loss 和验证准确率。
python src/train.py
```

loss 越低通常说明预测分数与正确标签越匹配，但不保证每轮都下降；验证准确率反映当前验证集的分类结果。较小验证集上的 100% 准确率不代表面对所有新图片都能正确识别。

运行后，项目根目录会产生以下文件：

- `best_model.pth`：字典中的 `model_state_dict` 保存最佳模型参数和缓冲区，`class_names` 保存训练时的类别顺序。准确率更高，或准确率相同且 loss 更低时覆盖保存；重新运行训练也会覆盖该文件。
- `training_loss.png`：各轮训练与验证 loss 曲线，重新运行时覆盖。

最佳准确率、对应 loss 和轮次会在终端输出。历史指标列表仅保留在本次运行的内存中；图片展示的是 loss 曲线，不包含准确率曲线。

`.pth` 是二进制文件，包含参数和类别名称，但不包含模型结构或优化器状态。加载时需要创建相同结构的模型，再把字典中的 `model_state_dict` 传给 `load_state_dict()`。当前项目的 `src/inspect_parameters.py` 已实现参数查看、加载及单张图片预测，可在项目根目录运行：

```bash
# 默认预测 sample_scene.png。
python src/inspect_parameters.py

# 也可以指定图片路径。
python src/inspect_parameters.py "/完整路径/你的图片.jpg"
```

预测脚本从新格式文件的 `class_names` 读取类别顺序。旧的纯参数文件仍可读取，但兼容分支仅适用于本项目原有的 `['city', 'indoor', 'nature']` 顺序；其他数据集应重新训练并保存新格式。预测时的模型结构和图片预处理仍必须与训练时一致。
