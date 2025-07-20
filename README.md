# Predictive-and-Analytical-Agents-for-Stock-Prices-and-Trends 股票价格与趋势预测分析智能体

### 免责声明：本项目由大连理工大学软件学院张桐宇同学毕业设计改编而来，不作任何商业用途，侵权联删，学长勿怪，狗头保命
<img width="474" height="296" alt="image" src="https://github.com/user-attachments/assets/3aee5c39-4902-4643-9bc6-1aeabcf07d79" />

### 著作声明：本项目系大连理工大学软件学院李得成、王宝康、廉文煜同学倾力奉献

本项目针对不同行业股票行为模式的异质性，设计了融合多源信息并能自适应选择最优预测策略的智能化解决方案。项目首先构建覆盖五大行业的多元化预测模型库，包含LSTM、GRU、CNN、Transformer的深度学习模型及随机森林、XGBoost的传统机器学习模型。通过对各行业历史数据进行细致特征工程和性能评估，筛选出各行业最优预测模型。将这些模型封装为标准化Web API服务后利用Dify应用开发平台，集成大语言模型（DeepSeek-r1），构建了一个智能分析框架。该框架通过解析用户自然语言查询，调用后端API获取量化预测，并融合输入的新闻文本，由LLM进行深度语义理解与影响评估，作出综合判断的智能化分析报告。


本项目实现了一个面向特定行业的股票价格预测分析智能体，通过以下创新点提升预测准确性：

行业特化模型选择：针对五大行业（科技/医药/金融/能源/消费）筛选最优预测模型

多模型融合：集成LSTM、GRU、CNN、Transformer等6种深度学习模型

智能决策支持：结合量化预测与新闻文本分析生成投资建议

端到端工作流：通过Dify平台实现自然语言交互到决策输出的完整流程


<img width="789" height="424" alt="屏幕截图 2025-07-20 122606" src="https://github.com/user-attachments/assets/b4034f22-d5a4-4d4e-9b65-c20e9a89be7f" />


## 一个简短的使用说明：
首先自部署Dify到本地（操作文档见 https://docs.dify.ai/zh-hans/getting-started/install-self-hosted/docker-compose ），接入一个开源大模型（ https://docs.dify.ai/zh-hans/guides/model-configuration/readme ），然后导入 股票价格分析智能体.yaml 文件，如果不想使用大模型新闻分析功能以上可以不做。

然后运行 BestModel.py 文件，这个运行的时间可能会有点长，再运行 api.py 文件，访问其运行返回的网址，即可使用本项目基础功能。

附运行截图：
<img width="2556" height="1272" alt="676499d3b8882d24d9753ddce1b53251" src="https://github.com/user-attachments/assets/188cccee-4377-4539-ad45-c7061fc28e5d" />

## 前端说明：
前端可以直接输入股票代码并选择具体行业类别，采用哪个模型进行预测与行业类别强相关，两个蓝色按钮：“开始预测”是直接调用picture文件夹下生成好的价格预测图，训练模型是调用模型进行训练然后将价格预测图保存到picture文件夹下。

大模型对话组件由Dify平台提供，如果没配置好Dify平台可能没法使用。
