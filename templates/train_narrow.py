"""机器人窄道二分类的完整训练模板。

使用顺序：放好图片和两份 CSV → 填写下方配置区 → 运行
    python templates/train_narrow.py

这份文件自己完成读取、训练、验证和保存；无需同时运行 src/train.py。
"""

# 读取标注 CSV 文件。
import csv
# 把每轮训练结果保存为 JSON。
import json
# 检查学习率是否为有限数字。
import math
# 使用 Path 处理跨目录文件路径。
from pathlib import Path

# 导入绘图库并设置无窗口绘图模式。
import matplotlib

# 在终端运行时直接保存图片，不弹出绘图窗口。
matplotlib.use("Agg")
# 使用 pyplot 绘制训练损失曲线。
import matplotlib.pyplot as plt
# 导入 PyTorch，创建张量并训练模型。
import torch
# 读取并检查图片文件。
from PIL import Image
# 使用 PyTorch 的神经网络层和损失函数。
from torch import nn
# Dataset 定义样本；DataLoader 自动组成批次。
from torch.utils.data import DataLoader, Dataset
# 把图片统一缩放并转换为模型输入张量。
from torchvision import transforms

# 从 templates/ 回到项目根目录；在 VS Code 或终端从其他目录启动也能找到文件。
# 根据本文件位置找到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ===== 1. 配置区：收到标注数据后先修改这里 =====
# 预期结构：narrow_road_dataset/train.csv、validation.csv、images/。
# 填写标注团队交付的数据目录。
DATA_DIR = PROJECT_ROOT / "narrow_road_dataset"
# 训练集标注表路径；表中需要 image_path、label 两列。
TRAIN_CSV = DATA_DIR / "train.csv"
# 验证集标注表路径；不能与训练集重复。
VALIDATION_CSV = DATA_DIR / "validation.csv"
# CSV 的 image_path 是相对这个目录的路径，例如 images/001.jpg。
# CSV 内图片相对路径的起点目录。
IMAGE_ROOT = DATA_DIR

# 第一个对应 CSV 的 label 0，第二个对应 label 1。必须按团队定义填写。
# 例如 0=非窄道、1=窄道：LABEL_0="non_narrow"；LABEL_1="narrow"。
# 按标注团队定义填写 0 的类别名。
LABEL_0 = ""
# 按标注团队定义填写 1 的类别名。
LABEL_1 = ""

# 本次实验的文件夹名；换参数训练时改名。
EXPERIMENT = "experiment_01"  # 新实验改名，以免覆盖旧结果。
# 这个目录会产生 best_model.pth、history.json、training_loss.png。
# 模型、指标和曲线统一保存到此目录。
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "narrow_road" / EXPERIMENT
# 训练轮数；每轮会遍历整个训练集。
EPOCHS = 10  # 训练集完整遍历的次数。
# 每批图片数；内存不足可以调小。
BATCH_SIZE = 8  # 一次处理的图片数；内存不足时调小。
# 参数更新幅度；初学时可先保持默认。
LEARNING_RATE = 0.01  # 训练步长；首次运行先保持默认。
# 固定随机种子，便于比较两次实验。
SEED = 42  # 固定随机起点，便于重复比较实验。

# ===== 配置区结束。以下依次是数据、模型、训练、验证、保存 =====

# 2. 数据：读取 CSV 和图片，生成训练/验证批次。


# 定义 CSV 图片数据集，供 DataLoader 按索引读取。
class NarrowRoadDataset(Dataset):
    """CSV 必须有 image_path、label；label 是 0 或 1，可选 sequence_id。"""

    # 创建数据集时，读取并校验整份标注表。
    def __init__(self, csv_path, image_root, transform=None):
        # 把 CSV 路径转换为绝对路径。
        self.csv_path = Path(csv_path).expanduser().resolve()
        # 把图片根目录转换为绝对路径。
        self.image_root = Path(image_root).expanduser().resolve()
        # 保存稍后处理图片时要用的转换步骤。
        self.transform = transform
        # 用列表保存通过检查的样本信息。
        self.records = []
        # 只读取标注表；实际图片在 __getitem__ 中按批次加载。
        # 打开 CSV；utf-8-sig 兼容带 BOM 的文件。
        with self.csv_path.open(newline="", encoding="utf-8-sig") as file:
            # 按列名读取每一行标注。
            reader = csv.DictReader(file)
            # 确认表头具有必需的图片路径和标签列。
            if not {"image_path", "label"}.issubset(reader.fieldnames or []):
                # 表头不符合约定时立即报错。
                raise ValueError(f"{self.csv_path} 必须包含 image_path、label 两列")
            # 记录是否提供采集序列编号。
            has_sequence = "sequence_id" in reader.fieldnames
            # 记录已出现的图片路径，防止同一表内重复。
            seen = set()
            # 逐行检查数据；行号从表头后的第 2 行开始。
            for line, row in enumerate(reader, start=2):
                # 生成报错定位信息，指出文件和行号。
                location = f"{self.csv_path}:{line}"
                # 读取并清理图片相对路径。
                name = (row.get("image_path") or "").strip()
                # 读取并清理 0/1 标签。
                value = (row.get("label") or "").strip()
                # 拒绝空路径和非二分类标签。
                if not name or value not in {"0", "1"}:
                    # 报告当前错误标注行。
                    raise ValueError(f"{location} 图片路径不能为空，label 必须为 0 或 1")
                # 用 IMAGE_ROOT + CSV 中的相对路径定位图片。
                # 将相对图片路径拼到 IMAGE_ROOT 下，再取绝对路径。
                image_path = (self.image_root /
                              Path(name).expanduser()).resolve()
                # 确认标注对应的图片存在。
                if not image_path.is_file():
                    # 报告缺失图片及其 CSV 行号。
                    raise FileNotFoundError(f"{location} 找不到图片：{image_path}")
                # 检查当前图片是否在本表出现过。
                if image_path in seen:
                    # 重复图片会导致样本权重失真，直接报错。
                    raise ValueError(f"{location} 图片重复：{image_path}")
                # 把当前图片加入已见集合。
                seen.add(image_path)
                # 提前验证文件可解码，避免训练到中途才发现损坏图片。
                # 打开图片以验证文件是否可读取。
                with Image.open(image_path) as image:
                    # 检查图片编码是否损坏。
                    image.verify()
                # 同一段视频/同一次采集的图片可共用一个 sequence_id。
                # 读取采集序列编号，未提供则为空。
                sequence = (row.get("sequence_id") or "").strip()
                # 若表中使用 sequence_id，则每行都必须填写。
                if has_sequence and not sequence:
                    # 报告漏填序列编号的行。
                    raise ValueError(f"{location} sequence_id 不能为空")
                # 保存这张图片的路径、数字标签和序列编号。
                self.records.append({
                    # 记录图片的绝对路径。
                    "image_path": image_path,
                    # 把 CSV 中的文本标签转成整数。
                    "label": int(value),
                    # 记录采集序列编号。
                    "sequence_id": sequence
                })
        # 确保标注表至少包含一张图片。
        if not self.records:
            # 空数据集无法训练，提前报错。
            raise ValueError(f"标注表为空：{self.csv_path}")

    # DataLoader 用这个方法获取样本总数。
    def __len__(self):
        # 返回通过检查的图片数量。
        return len(self.records)

    # DataLoader 用索引取出一张图片及其标签。
    def __getitem__(self, index):
        # DataLoader 需要某张图时才调用：返回 (图片张量, 整数标签)。
        # 按索引找到这张图片的标注信息。
        record = self.records[index]
        # 读取当前图片。
        with Image.open(record["image_path"]) as source:
            # 统一转为三通道 RGB。
            image = source.convert("RGB")
        # 若设置了预处理，就在这里应用。
        if self.transform is not None:
            # 缩放图片并转换为张量。
            image = self.transform(image)
        # 返回模型输入和对应的 0/1 答案。
        return image, record["label"]


# 定义训练和预测需要一致的图片预处理。
def create_image_transform():
    """训练与预测统一输入：RGB、64×64、浮点张量。"""
    # 把多个图片转换步骤组合起来。
    return transforms.Compose(
        # 把任意原图统一缩放为 64×64。
        [
            transforms.Resize((64, 64)),
            # 把像素转为 [0,1] 范围的浮点张量。
            transforms.ToTensor()
        ])


# 创建训练和验证批次，并检查数据划分。
def create_data_loaders():
    # 取得统一的图片预处理步骤。
    transform = create_image_transform()
    # 读取并校验训练集。
    train = NarrowRoadDataset(TRAIN_CSV, IMAGE_ROOT, transform)
    # 读取并校验验证集。
    validation = NarrowRoadDataset(VALIDATION_CSV, IMAGE_ROOT, transform)
    # 训练和验证的图片、采集序列必须分开；发现重叠就停止。
    # 收集训练集的图片路径。
    train_paths = {r["image_path"] for r in train.records}
    # 检查两份数据是否含相同图片。
    if train_paths & {r["image_path"] for r in validation.records}:
        # 发现图片泄漏时停止训练。
        raise ValueError("训练和验证 CSV 包含相同图片")
    # 收集训练集非空的采集序列编号。
    train_sequences = {
        r["sequence_id"]
        for r in train.records if r["sequence_id"]
    }
    # 收集验证集非空的采集序列编号。
    validation_sequences = {
        r["sequence_id"]
        for r in validation.records if r["sequence_id"]
    }
    # 检查两份 CSV 是否一致地提供序列编号。
    if bool(train_sequences) != bool(validation_sequences):
        # 序列编号使用不一致时停止。
        raise ValueError("两份 CSV 应同时提供 sequence_id，或同时不提供")
    # 检查同一采集序列是否跨训练集和验证集。
    if train_sequences & validation_sequences:
        # 防止相邻视频帧进入不同数据集。
        raise ValueError("训练和验证 CSV 包含相同采集序列")
    # 分别检查两份数据集的类别。
    for name, dataset in [("训练集", train), ("验证集", validation)]:
        # 确认每份数据都含 0 和 1 两类。
        if {r["label"] for r in dataset.records} != {0, 1}:
            # 单类数据集无法正确评估二分类。
            raise ValueError(f"{name} 必须同时包含 0、1 两类")
    # 批量大小必须为正整数。
    if BATCH_SIZE < 1:
        # 报告无效批量大小。
        raise ValueError("BATCH_SIZE 必须大于 0")
    # 训练打乱图片顺序；验证保持固定顺序。
    # 训练时打乱顺序；验证时保持固定顺序。
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False))


# 3. 模型：64×64 RGB 输入，输出两个类别分数。


# 创建输出指定类别数的简单 CNN。
def create_model(number_of_classes):
    """沿用已学过的 CNN；输出数量由类别数决定。"""
    # 按照列表顺序串联网络层。
    return nn.Sequential(
        # 卷积层从 RGB 图片提取 8 组特征。
        nn.Conv2d(3, 8, kernel_size=3, padding=1),  # [N,3,64,64] → [N,8,64,64]
        # 加入非线性激活。
        nn.ReLU(),
        # 池化后宽和高各减半。
        nn.MaxPool2d(2),  # → [N,8,32,32]
        # 把每张图的特征图展开成一维。
        nn.Flatten(start_dim=1),  # → [N,8192]
        # 输出每个类别的原始分数。
        nn.Linear(8 * 32 * 32, number_of_classes),  # 原始类别分数
    )


# 4. 训练和验证：训练时更新参数，验证时统计指标。


# 遍历一次数据集；有优化器时更新参数。
def run_epoch(model, loader, loss_function, optimizer=None):
    """传入优化器时训练；否则只验证，返回平均 loss 和混淆矩阵。"""
    # 传入 optimizer 表示训练；不传则只计算验证结果，不更新参数。
    # 根据优化器是否存在判断当前阶段。
    training = optimizer is not None
    # 切换模型的训练或验证模式。
    model.train(training)
    # 累计每张图片的损失之和。
    total_loss = 0.0
    # 创建混淆矩阵：行是真实类别，列是预测类别。
    matrix = torch.zeros(model[-1].out_features,
                         model[-1].out_features,
                         dtype=torch.long)
    # 训练时记录梯度；验证时关闭以节省资源。
    with torch.set_grad_enabled(training):
        # 逐批取得图片张量和真实标签。
        for images, labels in loader:
            # 仅训练阶段执行下面的参数更新操作。
            if training:
                # 清空上一批留下的梯度。
                optimizer.zero_grad()
            # images: [批次, 3, 64, 64]；scores: [批次, 2]。
            # 让模型输出每张图片的两类分数。
            scores = model(images)
            # 根据真实标签计算本批损失。
            loss = loss_function(scores, labels)
            # 检查损失是否出现 NaN 或无穷大。
            if not torch.isfinite(loss):
                # 损失异常时停止，避免保存无效模型。
                raise ValueError("loss 出现非有限值，请检查数据或降低学习率")
            # 仅训练阶段执行下面的参数更新操作。
            if training:
                # 只有训练阶段才计算梯度并更新模型。
                # 反向传播，计算每个参数的梯度。
                loss.backward()
                # 根据梯度更新模型参数。
                optimizer.step()
            # 把本批平均损失换算为本批总损失并累加。
            total_loss += loss.item() * labels.size(0)
            # 选分数最高的类别作为预测结果。
            predictions = scores.argmax(dim=1)
            # 行是真实类别，列是预测类别。
            # 把真实和预测类别组合成一维计数索引。
            indices = labels * matrix.size(0) + predictions
            # 统计本批各类预测数量并加到混淆矩阵。
            matrix += torch.bincount(
                indices, minlength=matrix.numel()).reshape(matrix.shape)
    # 返回整套数据的平均 loss 和混淆矩阵（行真实、列预测）。
    # 返回全数据集平均损失及混淆矩阵。
    return total_loss / len(loader.dataset), matrix


# 从混淆矩阵计算每一类的 precision 和 recall。
def describe_metrics(matrix, class_names):
    # 分别统计两个类别，尤其关注 narrow 的 recall（窄道被找出的比例）。
    # 准备存放按类别名称索引的指标。
    result = {}
    # 依次处理标签 0、标签 1。
    for index, name in enumerate(class_names):
        # 正确识别为本类的图片数。
        correct = matrix[index, index].item()
        # 模型预测为本类的图片总数。
        predicted = matrix[:, index].sum().item()
        # 实际属于本类的图片总数。
        actual = matrix[index].sum().item()
        # 保存本类精确率、召回率和样本数。
        result[name] = {
            # 精确率：预测为本类的图片中有多少是真的。
            "precision": correct / predicted if predicted else 0.0,
            # 召回率：真实本类图片中找出了多少。
            "recall": correct / actual if actual else 0.0,
            # 该类别在验证集中的实际图片数。
            "support": actual
        }
    # 返回所有类别的指标字典。
    return result


# 5. 保存：每轮记录指标，保留验证表现最好的模型和损失曲线。
# 主流程：检查配置、训练模型、保存结果。
def main():
    # 按数字标签 0、1 的顺序建立类别名称列表。
    class_names = [LABEL_0, LABEL_1]
    # 确认两类名称都已按标注定义填写。
    if set(class_names) != {"non_narrow", "narrow"}:
        # 标签映射未填时给出操作提示。
        raise SystemExit("先按标注团队的 0/1 定义填写 LABEL_0 和 LABEL_1")
    # 实验名必须是单个非空目录名。
    if not EXPERIMENT.strip() or Path(EXPERIMENT).name != EXPERIMENT:
        # 报告不合法的实验名。
        raise SystemExit("EXPERIMENT 必须是单个非空文件夹名")
    # 确认轮数与学习率是有效正数。
    if EPOCHS < 1 or not math.isfinite(LEARNING_RATE) or LEARNING_RATE <= 0:
        # 报告无效训练参数。
        raise SystemExit("EPOCHS 和 LEARNING_RATE 必须大于 0")
    # 先检查填写的配置，再读取数据、创建模型；错误尽量在训练前报告。
    # 固定 PyTorch 的随机数起点。
    torch.manual_seed(SEED)
    # 读取训练集和验证集并生成批次。
    train_loader, validation_loader = create_data_loaders()
    # 确定训练产物的绝对保存目录。
    output_dir = OUTPUT_DIR.expanduser().resolve()
    # 自动创建结果目录及缺失的上级目录。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建输出 0/1 两类分数的网络。
    model = create_model(number_of_classes=2)
    # 用 SGD 按学习率更新网络参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # 使用交叉熵衡量二分类预测误差。
    loss_function = nn.CrossEntropyLoss()
    # 保存每轮训练和验证的统计结果。
    history = []
    # 初始化最佳指标，保证第一轮可以保存模型。
    best_accuracy, best_loss = -1.0, float("inf")
    # 在终端显示 0、1 分别代表的类别。
    print("类别映射：", dict(enumerate(class_names)))
    # 在终端显示两份数据集的图片数量。
    print("训练 / 验证图片数：", len(train_loader.dataset),
          len(validation_loader.dataset))

    # 每轮：先训练一次，再用验证集检查一次。
    # 重复指定轮数；每轮都训练并验证。
    for epoch in range(1, EPOCHS + 1):
        # 训练一轮；这里不使用训练集混淆矩阵。
        train_loss, _ = run_epoch(model, train_loader, loss_function,
                                  optimizer)
        # 在验证集上计算损失和混淆矩阵。
        validation_loss, matrix = run_epoch(model, validation_loader,
                                            loss_function)
        # 准确率等于预测正确数除以总图片数。
        accuracy = matrix.diag().sum().item() / matrix.sum().item()
        # 计算各类别的精确率和召回率。
        metrics = describe_metrics(matrix, class_names)
        # 汇总当前轮所有需要保存的统计数据。
        record = {
            # 当前训练轮数。
            "epoch": epoch,
            # 训练集平均损失。
            "train_loss": train_loss,
            # 验证集平均损失。
            "validation_loss": validation_loss,
            # 验证集总体准确率。
            "validation_accuracy": accuracy,
            # 每一类的精确率、召回率和图片数。
            "class_metrics": metrics,
            # 把混淆矩阵转换成可写入 JSON 的列表。
            "confusion_matrix": matrix.tolist()
        }
        # 把当前轮指标加入训练历史。
        history.append(record)
        # 准确率更高时保存；准确率相同则选验证 loss 更低的一轮。
        # 优先比较准确率；平局时比较验证损失。
        if accuracy > best_accuracy or (accuracy == best_accuracy
                                        and validation_loss < best_loss):
            # 更新目前最好的验证结果。
            best_accuracy, best_loss = accuracy, validation_loss
            # 把当前最佳模型及复现所需信息写入文件。
            torch.save(
                {
                    # 保存模型学到的权重。
                    "model_state_dict": model.state_dict(),
                    # 保存 0/1 对应的类别名供预测时使用。
                    "class_names": class_names,
                    # 保存模型要求的输入宽高。
                    "image_size": 64,
                    # 记录当前使用的网络结构版本。
                    "architecture": "simple_cnn_v1",
                    # 当前训练轮数。
                    "epoch": epoch,
                    # 同时保存这轮验证指标。
                    "metrics": record,
                    # 记录本次实验的随机种子。
                    "seed": SEED
                },
                output_dir / "best_model.pth")
        # 每轮把指标写到磁盘，方便训练后查看和比较。
        # 覆盖写入截至当前轮的完整训练历史。
        (output_dir / "history.json").write_text(json.dumps(history, indent=2),
                                                 encoding="utf-8")
        # 在终端报告本轮损失、准确率和窄道召回率。
        print(f"第 {epoch}/{EPOCHS} 轮：train loss={train_loss:.4f}，"
              f"validation loss={validation_loss:.4f}，accuracy={accuracy:.2%}，"
              f"narrow recall={metrics['narrow']['recall']:.2%}")

    # 所有轮次结束后，保存训练和验证 loss 曲线。
    # 创建 8×5 英寸的曲线图。
    plt.figure(figsize=(8, 5))
    # 分别绘制训练损失和验证损失。
    for key, label in [("train_loss", "Train loss"),
                       ("validation_loss", "Validation loss")]:
        # 横轴为轮数，纵轴为当前损失。
        plt.plot([r["epoch"] for r in history], [r[key] for r in history],
                 marker="o",
                 label=label)
    # 给横轴标注训练轮数。
    plt.xlabel("Epoch")
    # 给纵轴标注损失值。
    plt.ylabel("Loss")
    # 显示两条曲线的图例。
    plt.legend()
    # 调整边距，避免文字被裁切。
    plt.tight_layout()
    # 把曲线保存为 PNG 文件。
    plt.savefig(output_dir / "training_loss.png")
    # 关闭图像，释放绘图资源。
    plt.close()
    # 输出所有轮次中最高的验证准确率。
    print("最佳准确率：", f"{best_accuracy:.2%}")
    # 提示模型和统计文件的保存位置。
    print("结果目录：", output_dir)


# 直接运行此文件时才启动训练；被导入时不会自动训练。
if __name__ == "__main__":
    # 执行主流程。
    main()
