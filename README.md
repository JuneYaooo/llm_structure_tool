[[中文版](https://github.com/JuneYaooo/llm_structure_tool/blob/main/README.md)] [[English](https://github.com/JuneYaooo/llm_structure_tool/blob/main/README_en.md)]

# 病历结构化工具（持续更新中）

该工具是一个基于 PULSE 模型的结构化工具，旨在帮助用户处理和分析文本数据。它提供了以下功能，适用于医疗场景的结构化使用：

- **单选**

![单选案例](img/2.jpg)

- **多选**

![多选案例](img/3.jpg)

- **提取**

![提取案例](img/1.jpg)

### 安装

首先，克隆本项目到本地计算机：

```
git clone https://github.com/JuneYaooo/llm_structure_tool.git
```

#### 使用 pip 安装

确保您的计算机上已安装以下依赖项：

- Python 3.9
- pip 包管理器

进入项目目录并安装必要的依赖项：

```
cd llm_structure_tool
pip install -r requirements.txt
```

#### 使用 conda 安装

确保您的计算机上已安装以下依赖项：

- Anaconda 或 Miniconda

进入项目目录并创建一个新的 conda 环境：

```
cd llm_structure_tool
conda env create -f environment.yml
```

激活新创建的环境：

```
conda activate llm_structure
```

然后运行前端demo：

```
python app.py
```

## 使用方法
结构化工具将在终端上提供一个简单的交互界面。您可以根据提示输入相关信息，选择要执行的功能。

### 测试

输入一段话，设定规则，进行单选、多选或提取

**示例：**

字段类型：提取

字段名：肾上腺肿物大小

原文：CT检查示左肾上腺区见大小约5.5 cm×5.7 cm不均匀低密度肿块，边界清楚，增强扫描实性成分中度强化，内见无强化低密度，静脉期明显强化。CT诊断：考虑左肾上腺区肿瘤。B超检查示左肾上腺区见4.6 cm×4.2 cm的低回声区，边界清，有包膜，提示左肾上腺实质性占位声像。


输入不相关的字段，如胃部肿物大小，结果为“未提及”
![提取案例-对比1](img/4.jpg)

输入相关的字段，如肾上腺肿物大小，结果为“约5.5 cm×5.7 cm”
![提取案例-对比2](img/5.jpg)


## 贡献

如果您对该项目感兴趣，欢迎贡献您的代码和改进建议。您可以通过以下方式参与：

1. 提交问题和建议到本项目的 Issue 页面。
2. Fork 本项目并提交您的改进建议，我们将会审查并合并合适的改动。
