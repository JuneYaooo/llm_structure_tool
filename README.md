[[中文版](https://github.com/JuneYaooo/medical_kb_chatbot/blob/main/README.md)] [[English](https://github.com/JuneYaooo/medical_kb_chatbot/blob/main/README_en.md)]

# 病历结构化工具（under construction）
（代码待上传..）

该工具是一个基于 PULSE 模型的结构化工具，旨在帮助用户处理和分析文本数据。它提供了以下功能，适用于医疗场景的结构化使用：

- 单选
- 多选
- 提取

## 安装

要安装该工具，请按照以下步骤操作：

1. 克隆或下载工具的代码仓库。

2. 安装所需的依赖项。建议使用虚拟环境来避免与其他项目的冲突。

   ```shell
   pip install -r requirements.txt
   ```

3. 运行工具。

   ```shell
   python app.py
   ```

## 使用方法
结构化工具将在终端上提供一个简单的交互界面。您可以根据提示输入相关信息，选择要执行的功能。

### 测试

输入一段话，设定规则，进行单选、多选或提取

### 训练

根据要求填写配置文件，进行训练

### 预测

根据要求填写配置文件，选择合适的Lora模型进行预测

### 评估

放入真实标签和预测出的文件，进行预测


## 贡献

如果您对该项目感兴趣，欢迎贡献您的代码和改进建议。您可以通过以下方式参与：

1. 提交问题和建议到本项目的 Issue 页面。
2. Fork 本项目并提交您的改进建议，我们将会审查并合并合适的改动。
