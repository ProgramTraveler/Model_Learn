# 窄道二分类：照着做的模板

这份手册对应当前的 `src/train.py` 和 `src/inspect_parameters.py`。照顺序完成即可；目前无需改模型代码。所有命令从 `Model_Learn` 项目根目录运行。

## 最省事的用法：改配置区，运行 Python 模板

如果不想每次输入整条命令，直接使用仓库中的两个代码模板：

1. 打开 [`templates/train_narrow.py`](templates/train_narrow.py)，填写 `LABEL_0`、`LABEL_1`，核对数据路径和实验名称；运行 `python templates/train_narrow.py`。这是一份包含数据读取、模型、训练、验证和保存的完整代码模板。
2. 打开 [`templates/predict_narrow.py`](templates/predict_narrow.py)，填写实验名称和待预测图片路径；运行 `python templates/predict_narrow.py`。

训练模板本身包含完整流程，可直接阅读和修改；预测模板调用 `src/inspect_parameters.py`。下面保留命令行版训练方式，供你按需要选择。

## 模板 1：向标注团队确认交付内容

把下面这段发给标注团队，收到答复后填写方括号：

> 任务：根据机器人前向相机图片，判断机器人前方通道是否狭窄。请提供原图、训练集 CSV 和验证集 CSV。CSV 至少包含 `image_path,label` 两列，label 只能是 0 或 1。请确认 **0 = [类别]、1 = [类别]**，以及窄道的标注标准：[机器人宽度、两侧余量、观察区域、遮挡时的处理方式]。如图片来自视频或连续采集，请在两份 CSV 中都加 `sequence_id`，相同采集序列只放在同一个集合。请说明每类在训练集、验证集中的图片数量。

实际收到的数据若不是这种 CSV 格式，先按交付格式调整读取代码，不要只改文件后缀。

## 模板 2：放置数据

把交付文件整理为：

```text
Model_Learn/
└── narrow_road_dataset/
    ├── train.csv
    ├── validation.csv
    └── images/
        ├── image_001.jpg
        ├── image_002.jpg
        └── ...
```

CSV 中 `image_path` 从 `narrow_road_dataset/` 开始计算，例如：

```csv
image_path,label,sequence_id
images/image_001.jpg,1,run_001
images/image_002.jpg,0,run_002
```

`sequence_id` 是可选列；如果使用，两份 CSV 都应提供。每份 CSV 都要有 0 和 1 两类图片。训练集与验证集不要共享图片，也不要把同一视频的相邻帧分到两边。数据目录默认被 Git 忽略。

## 模板 3：训练

打开 VS Code 终端，进入项目目录并激活已创建的环境：

```bash
cd /Users/help/code_data/Model_Learn
source .venv/bin/activate
```

确认团队给出的标签映射，然后**只选下面一条**。不要猜测 0 和 1 的含义。

团队确认 **0=非窄道、1=窄道** 时：

```bash
python src/train.py \
  --train-csv narrow_road_dataset/train.csv \
  --validation-csv narrow_road_dataset/validation.csv \
  --image-root narrow_road_dataset \
  --class-names non_narrow narrow \
  --epochs 10 --batch-size 8 --lr 0.01 \
  --output-dir outputs/narrow_road/experiment_01
```

团队确认 **0=窄道、1=非窄道** 时：

```bash
python src/train.py \
  --train-csv narrow_road_dataset/train.csv \
  --validation-csv narrow_road_dataset/validation.csv \
  --image-root narrow_road_dataset \
  --class-names narrow non_narrow \
  --epochs 10 --batch-size 8 --lr 0.01 \
  --output-dir outputs/narrow_road/experiment_01
```

换一批数据或参数时，将 `experiment_01` 改成 `experiment_02` 等。重用同一个输出目录会覆盖其中的模型、历史记录和曲线。训练命令中 `--class-names` 的第一个名称对应 label 0，第二个对应 label 1。

## 模板 4：查看训练结果

完成后打开刚才指定的输出目录：

| 文件 | 用途 |
| --- | --- |
| `best_model.pth` | 预测时加载的模型；按验证准确率优先、同分时按验证 loss 选出 |
| `history.json` | 每一轮的训练 loss、验证 loss、准确率、各类 precision/recall 和混淆矩阵 |
| `training_loss.png` | 训练与验证 loss 曲线 |

先查看 `history.json` 各轮记录，尤其看 `class_metrics.narrow.recall`：它低时，验证集中的窄道漏判较多。`confusion_matrix` 的行是真实类别、列是预测类别，顺序与命令中的 `--class-names` 一致。结果来自当前验证集，不代表机器人在新场景中的真实表现。

## 模板 5：预测一张新图片

将新图片放在本机任意可读取的位置，并替换下面的图片路径。模型路径须与训练时的 `--output-dir` 一致：

```bash
python src/inspect_parameters.py /完整路径/待预测图片.jpg \
  --model outputs/narrow_road/experiment_01/best_model.pth
```

终端会显示类别顺序、各类分数与预测类别。此预测只用于离线检查；当前代码不直接连接机器人控制系统。

## 模板 6：常见问题直接对照

| 看到的情况 | 检查动作 |
| --- | --- |
| `No module named ...` | 在项目目录执行 `source .venv/bin/activate`，再运行命令 |
| `必须包含 image_path、label` | 检查 CSV 第一行列名，拼写须完全一致 |
| `找不到图片` | 用 `--image-root` 加上 CSV 的 `image_path` 核对实际路径 |
| `label 必须为 0 或 1` | 请团队确认并统一标签值，不要私自猜测或重映射 |
| `训练集和验证集包含相同...` | 按图片或采集序列重新划分数据，保留独立验证集 |
| 预测显示的类别反了 | 核对团队的 0/1 定义与训练时 `--class-names` 的顺序；修正后重新训练 |
| 运行 `python src/train.py` 得到三分类 | 窄道训练须使用模板 3 的完整 CSV 命令 |
| 找不到模型文件 | 检查训练输出目录，并让 `--model` 指向其中的 `best_model.pth` |

## 每次实验的记录模板

复制下面这段，填入真实值，方便比较实验：

```text
实验名称：experiment_01
数据来源与采集日期：
机器人及相机：
窄道标注标准：
标签映射：0 = ______；1 = ______
训练 CSV / 图片数：
验证 CSV / 图片数：
是否按采集序列隔离：
训练命令：
模型输出目录：
最佳 epoch / 验证准确率：
narrow precision / recall：
典型漏判图片：
下一步动作：
```

当前 `64×64` CNN 是学习基线。真实数据到位后，先按上述流程完成一次可复现的离线实验，再用独立场景检查效果。
