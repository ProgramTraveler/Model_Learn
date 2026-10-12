# Model Learn

这是一个用于记录 PyTorch 与图像分类学习过程的项目。

当前目标是逐步完成机器人前方通道“窄道 / 非窄道”的二分类模型。现已支持 CSV 标注的二分类训练与单图预测，同时保留原三分类环境示例。真实窄道数据由标注团队提供。项目使用 VS Code 开发，主要代码位于 `src/train.py`。

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
Model_Learn/
├── learning_scene_dataset/   # 场景图片数据集
│   ├── train/                # 训练集
│   └── validation/           # 验证集
├── src/
│   ├── train.py              # 当前训练与学习代码
│   ├── inspect_parameters.py # 加载模型并预测图片
│   └── learn_img/            # 学习过程中生成的示意图
├── requirements.txt         # 已验证的依赖版本
├── .python-version          # Python 版本
├── TRAINING_TEMPLATE.md     # 可套用的操作模板
├── .gitignore
└── README.md
```

## 运行方式

本项目使用 Python 3.12 和项目虚拟环境 `.venv`。依赖版本记录在 `requirements.txt`。如需重新创建环境，请先准备 Python 3.12，再执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

进入项目目录并激活虚拟环境：

```bash
cd Model_Learn
source .venv/bin/activate
```

运行训练程序：

```bash
python src/train.py
python src/inspect_parameters.py
```

## 窄道二分类：直接运行

想直接套用数据交付、训练、预测和实验记录模板，请先看 [TRAINING_TEMPLATE.md](TRAINING_TEMPLATE.md)。 若想直接套用完整训练代码，可用 [训练代码模板](templates/train_narrow.py)；单图预测可用 [预测模板](templates/predict_narrow.py)。


CSV 格式暂定为 `image_path,label`，支持 UTF-8 或带 BOM 的 UTF-8，可选 `sequence_id`。
图片相对路径以 `--image-root` 为起点；标注必须是整数 0 或 1。

```csv
image_path,label,sequence_id
images/frame_001.jpg,1,run_001
images/frame_002.jpg,0,run_002
```

准备 `narrow_road_dataset/train.csv`、`validation.csv` 和图片目录。训练、验证各包含两个类别，按采集序列或地点划分，不能将同一段视频的相邻帧随机分到两边。程序拒绝跨集合的相同图片路径和相同 sequence_id；它不能自动识别不同文件名的近似图片。

以下命令**假设团队确认 0=非窄道、1=窄道**。如果编号相反，将 `--class-names` 改为 `narrow non_narrow`，训练和预测使用保存的顺序。

```bash
python src/train.py \
  --train-csv narrow_road_dataset/train.csv \
  --validation-csv narrow_road_dataset/validation.csv \
  --image-root narrow_road_dataset \
  --class-names non_narrow narrow \
  --epochs 10 --batch-size 8 --lr 0.01

python src/inspect_parameters.py narrow_road_dataset/images/new_image.jpg \
  --model outputs/narrow_road/best_model.pth
```

输出包含 `best_model.pth`、`history.json`、`training_loss.png`。混淆矩阵的行是真实类别，列是预测类别；类别顺序与保存的 class_names 一致。最佳模型仍按验证准确率优先、相同准确率比较 loss 选择。注意窄道类别 recall 反映漏判情况，softmax 数值不是经过校准的可靠置信度。

不带参数运行 `python src/train.py` 会训练原三分类示例，结果在 `outputs/scene_model/`，不会覆盖窄道模型。可使用 `--output-dir` 为不同实验指定目录；再次使用同一目录会覆盖结果。旧的根目录模型可用 `--model best_model.pth` 显式加载。当前仅使用 CPU 和 64×64 基础 CNN，作为学习基线；真实窄道效果需用独立采集数据验证。

`outputs/` 和 `narrow_road_dataset/` 默认被 Git 忽略。仓库不包含真实窄道数据，也不使用生成图片代替真实训练数据。

## 学习进度

### 2026-10-11：环境配置与窄道数据读取起点

已配置 Python 3.12 环境并验证训练与预测。已完成 CSV 校验、图片读取、二分类训练入口、混淆矩阵、各类别 precision / recall 和模型预测。已精简额外单批次演示、复用预处理，并移除预测脚本中的旧三分类标签兜底。后续学习先修改代码，再逐段解读。


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
