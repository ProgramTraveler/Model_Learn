# JupyterLab 启动与关闭

项目目录：

```text
/Users/zhuanz/Documents/learn/scene_model
```

## 启动 JupyterLab

打开 iTerm，然后进入项目目录：

```bash
cd /Users/zhuanz/Documents/learn/scene_model
```

激活 Python 虚拟环境：

```bash
source .venv/bin/activate
```

激活成功后，终端提示符通常会出现 `(.venv)`。

检查当前 Python 是否属于本项目：

```bash
python -c "import sys; print(sys.executable)"
```

预期路径：

```text
/Users/zhuanz/Documents/learn/scene_model/.venv/bin/python
```

启动 JupyterLab：

```bash
jupyter lab
```

如果不希望自动打开默认浏览器：

```bash
jupyter lab --no-browser
```

然后复制终端中包含 `?token=...` 的完整地址到浏览器，例如：

```text
http://localhost:8888/lab?token=...
```

> JupyterLab 运行期间，启动它的终端会一直被占用，这是正常现象。请保持该终端窗口开启。

## 查看正在运行的服务

另开一个 iTerm 标签页，激活虚拟环境后运行：

```bash
jupyter server list
```

## 关闭 JupyterLab

1. 保存 Notebook。
2. 回到运行 JupyterLab 的终端。
3. 按 `Control + C`。
4. 如果终端询问是否关闭服务，输入 `y` 并回车。

退出虚拟环境：

```bash
deactivate
```

也可以在另一个已激活虚拟环境的终端中停止 8888 端口的服务：

```bash
jupyter server stop 8888
```

