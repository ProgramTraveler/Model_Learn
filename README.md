# Model Learn

这是一个用于记录 PyTorch 与图像分类学习过程的项目。

当前目标是从基础概念开始，逐步完成一个能够识别特定场景图片的分类模型。项目使用 VS Code 开发，主要代码位于 `src/train.py`。

## 当前学习内容

- 使用 `ImageFolder` 根据文件夹名称识别图片类别
- 使用 `DataLoader` 分批加载图片与标签
- 使用 `Conv2d`、`ReLU` 和 `MaxPool2d` 提取图片特征
- 使用 `Flatten` 和 `Linear` 输出各类别的原始分数
- 使用 `CrossEntropyLoss` 计算分类 loss
- 使用反向传播和 SGD 更新模型参数
- 计算一轮训练的平均 loss
- 使用验证集计算平均验证 loss

## 项目结构

```text
scene_model/
├── learning_scene_dataset/   # 场景图片数据集
│   ├── train/                # 训练集
│   └── validation/           # 验证集
├── src/
│   ├── train.py              # 当前训练与学习代码
│   └── learn_img/            # 学习过程中生成的示意图
├── .gitignore
└── README.md
```

## 运行方式

进入项目目录并激活虚拟环境：

```bash
cd /Users/zhuanz/Documents/learn/scene_model
source .venv/bin/activate
```

运行训练程序：

```bash
python src/train.py
```

## 学习进度

### 2026-09-28：验证准确率与多轮训练

已完成：

- 逐行理解验证循环以及 `model.eval()` 和 `torch.no_grad()`
- 理解验证批次平均 loss 与整个验证集平均 loss 的计算方式
- 使用 `argmax(dim=1)` 从类别分数中取得预测类别
- 比较预测类别与正确标签，并计算验证准确率
- 使用外层 epoch 循环连续训练和验证 10 轮
- 每一轮输出平均训练 loss、平均验证 loss 和验证准确率

当前结果：一次运行中，验证准确率从第 1 轮的 `33.33%` 提升到第 10 轮的 `100.00%`，平均验证 loss 从 `0.9945` 降至 `0.0850`。

需要注意：当前验证集较小，验证准确率达到 `100%` 不代表模型面对所有新图片时也能达到相同效果。

下一步：理解外层 epoch 循环，并学习记录每轮指标和绘制训练曲线。

### 2026-09-23：训练与验证

已完成：

- 理解一个训练批次中的图片、标签和类别分数
- 理解 `CrossEntropyLoss` 的基本计算方式
- 完成单批次的前向传播、反向传播和参数更新
- 遍历完整训练集，并计算一轮的平均训练 loss
- 使用 `model.eval()` 和 `torch.no_grad()` 验证模型
- 计算整个验证集的平均验证 loss

当前结果：训练流程和验证流程均可正常运行。

下一步：逐行理解验证循环，然后计算验证准确率。

## 后续记录模板

每次学习结束后，可以按照下面的格式继续追加：

```markdown
### YYYY-MM-DD：本次主题

已完成：

- 学习内容一
- 学习内容二

遇到的问题：

- 问题与解决方式

下一步：下一次准备学习的内容。
```
