import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 只保存曲线，终端运行不需要图形窗口。
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "learning_scene_dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VALIDATION_DIR = DATASET_ROOT / "validation"


class NarrowRoadDataset(Dataset):
    """CSV 必须有 image_path、label；label 是 0 或 1，可选 sequence_id。"""

    def __init__(self, csv_path, image_root, transform=None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.transform = transform
        self.records = []
        with self.csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not {"image_path", "label"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{self.csv_path} 必须包含 image_path、label 两列")
            has_sequence = "sequence_id" in reader.fieldnames
            seen = set()
            for line, row in enumerate(reader, start=2):
                location = f"{self.csv_path}:{line}"
                name = (row.get("image_path") or "").strip()
                value = (row.get("label") or "").strip()
                if not name or value not in {"0", "1"}:
                    raise ValueError(f"{location} 图片路径不能为空，label 必须为 0 或 1")
                image_path = (self.image_root /
                              Path(name).expanduser()).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"{location} 找不到图片：{image_path}")
                if image_path in seen:
                    raise ValueError(f"{location} 图片重复：{image_path}")
                seen.add(image_path)
                # 提前验证文件可解码，避免训练到中途才发现损坏图片。
                with Image.open(image_path) as image:
                    image.verify()
                sequence = (row.get("sequence_id") or "").strip()
                if has_sequence and not sequence:
                    raise ValueError(f"{location} sequence_id 不能为空")
                self.records.append({
                    "image_path": image_path,
                    "label": int(value),
                    "sequence_id": sequence
                })
        if not self.records:
            raise ValueError(f"标注表为空：{self.csv_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record["label"]


def create_image_transform():
    """训练与预测统一输入：RGB、64×64、浮点张量。"""
    return transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor()])


def create_data_loaders(batch_size=8,
                        *,
                        train_csv=None,
                        validation_csv=None,
                        image_root=None,
                        class_names=None):
    transform = create_image_transform()
    options = (train_csv, validation_csv, image_root, class_names)
    if any(value is not None for value in options):
        if any(value is None for value in options):
            raise ValueError("CSV 模式需要训练表、验证表、图片根目录和类别顺序")
        if isinstance(class_names, str):
            raise ValueError("class_names 应是两个名称组成的列表")
        class_names = list(class_names)
        if (len(class_names) != 2 or len(set(class_names)) != 2
                or any(not isinstance(n, str) or not n.strip()
                       for n in class_names)):
            raise ValueError("class_names 必须按 label 0、1 提供两个不同的非空名称")
        train = NarrowRoadDataset(train_csv, image_root, transform)
        validation = NarrowRoadDataset(validation_csv, image_root, transform)
        # 防止同一图片或同一采集序列出现在训练和验证两边。
        train_paths = {r["image_path"] for r in train.records}
        if train_paths & {r["image_path"] for r in validation.records}:
            raise ValueError("训练集和验证集包含相同图片路径，请重新划分")
        train_sequences = {
            r["sequence_id"]
            for r in train.records if r["sequence_id"]
        }
        validation_sequences = {
            r["sequence_id"]
            for r in validation.records if r["sequence_id"]
        }
        if bool(train_sequences) != bool(validation_sequences):
            raise ValueError("两份 CSV 应同时提供 sequence_id，或同时不提供")
        if train_sequences & validation_sequences:
            raise ValueError("训练集和验证集包含相同 sequence_id，请按采集序列划分")
        for name, dataset in [("训练集", train), ("验证集", validation)]:
            if {r["label"] for r in dataset.records} != {0, 1}:
                raise ValueError(f"{name} 必须包含两个类别的图片")
    else:
        train = datasets.ImageFolder(TRAIN_DIR, transform=transform)
        validation = datasets.ImageFolder(VALIDATION_DIR, transform=transform)
        if train.class_to_idx != validation.class_to_idx:
            raise ValueError("训练和验证的类别映射不一致")
        class_names = train.classes
    if batch_size < 1 or not len(train) or not len(validation):
        raise ValueError("批次大小必须大于 0，数据集不能为空")
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(validation, batch_size=batch_size,
                       shuffle=False), class_names)


def create_model(number_of_classes):
    """沿用已学过的 CNN；输出数量由类别数决定。"""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),  # [N,3,64,64] → [N,8,64,64]
        nn.ReLU(),
        nn.MaxPool2d(2),  # → [N,8,32,32]
        nn.Flatten(start_dim=1),  # → [N,8192]
        nn.Linear(8 * 32 * 32, number_of_classes),  # 原始类别分数
    )


def run_epoch(model, loader, loss_function, optimizer=None):
    """传入优化器时训练；否则只验证，返回平均 loss 和混淆矩阵。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    matrix = torch.zeros(model[-1].out_features,
                         model[-1].out_features,
                         dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad()
            scores = model(images)
            loss = loss_function(scores, labels)
            if not torch.isfinite(loss):
                raise ValueError("loss 出现非有限值，请检查数据或降低学习率")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predictions = scores.argmax(dim=1)
            # 行是真实类别，列是预测类别。
            indices = labels * matrix.size(0) + predictions
            matrix += torch.bincount(
                indices, minlength=matrix.numel()).reshape(matrix.shape)
    return total_loss / len(loader.dataset), matrix


def describe_metrics(matrix, class_names):
    result = {}
    for index, name in enumerate(class_names):
        correct = matrix[index, index].item()
        predicted = matrix[:, index].sum().item()
        actual = matrix[index].sum().item()
        result[name] = {
            "precision": correct / predicted if predicted else 0.0,
            "recall": correct / actual if actual else 0.0,
            "support": actual
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="场景示例 / 机器人窄道 CSV 二分类训练（CPU）")
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--class-names",
                        nargs=2,
                        choices=["non_narrow", "narrow"],
                        help="按 label 0、1 的顺序提供名称，必须与团队标注一致")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="默认 outputs/narrow_road 或 outputs/scene_model；重新运行会覆盖同目录产物")
    args = parser.parse_args()
    options = (args.train_csv, args.validation_csv, args.image_root,
               args.class_names)
    csv_mode = any(value is not None for value in options)
    if csv_mode and (any(value is None for value in options)
                     or set(args.class_names) != {"narrow", "non_narrow"}):
        parser.error("窄道模式须同时提供两份 CSV、图片根目录和两个不同类别名称")
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(
            args.lr) or args.lr <= 0:
        parser.error("epochs、batch-size 和 lr 必须大于 0，lr 必须是有限数值")
    torch.manual_seed(args.seed)
    try:
        train_loader, validation_loader, class_names = create_data_loaders(
            args.batch_size,
            train_csv=args.train_csv,
            validation_csv=args.validation_csv,
            image_root=args.image_root,
            class_names=args.class_names)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output_dir = (
        args.output_dir or PROJECT_ROOT / "outputs" /
        ("narrow_road" if csv_mode else "scene_model")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("模式：", "窄道二分类" if csv_mode else "原三分类示例")
    print("类别映射：", dict(enumerate(class_names)))
    print("训练 / 验证样本数：", len(train_loader.dataset),
          len(validation_loader.dataset))
    model = create_model(len(class_names))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy, best_loss = -1.0, float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_function,
                                  optimizer)
        validation_loss, matrix = run_epoch(model, validation_loader,
                                            loss_function)
        accuracy = matrix.diag().sum().item() / matrix.sum().item()
        metrics = describe_metrics(matrix, class_names)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy,
            "class_metrics": metrics,
            "confusion_matrix": matrix.tolist()
        }
        history.append(record)
        if accuracy > best_accuracy or (accuracy == best_accuracy
                                        and validation_loss < best_loss):
            best_accuracy, best_loss = accuracy, validation_loss
            # 参数、标签映射、预处理信息和最佳轮次一起保存，预测时无需猜测。
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": 64,
                    "architecture": "simple_cnn_v1",
                    "epoch": epoch,
                    "metrics": record,
                    "seed": args.seed
                }, output_dir / "best_model.pth")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2),
                                                 encoding="utf-8")
        print(f"第 {epoch}/{args.epochs} 轮：train loss={train_loss:.4f}，"
              f"validation loss={validation_loss:.4f}，accuracy={accuracy:.2%}")
        for name, value in metrics.items():
            print(
                f"  {name}：precision={value['precision']:.2%}，recall={value['recall']:.2%}"
            )
    plt.figure(figsize=(8, 5))
    for key, label in [("train_loss", "Train loss"),
                       ("validation_loss", "Validation loss")]:
        plt.plot([r["epoch"] for r in history], [r[key] for r in history],
                 marker="o",
                 label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png")
    plt.close()
    print("最佳验证准确率：", f"{best_accuracy:.2%}")
    print("结果目录：", output_dir)


if __name__ == "__main__":
    main()
