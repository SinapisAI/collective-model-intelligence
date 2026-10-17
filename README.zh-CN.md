![模型群体智能（Collective Model Intelligence）：模型能力刻画与协调。](docs/assets/readme-banner.svg)

# 模型群体智能

**Toward Collective Model Intelligence: Characterizing and Coordinating Models**

[**技术报告**](tech-report.pdf) · [实验结果](docs/experimental-results.md) · [体验 Mio](https://sinapis.ai) · [开源数据集](https://huggingface.co/datasets/SinapisAI/RobustRouteBench) · [English](README.md)

## Mio 1.0 Lite

**Mio（Models in One）** 将能力互补的模型组织起来，根据任务需求和剩余预算选择、协调模型调用。**Mio 1.0 Lite** 将这一思路用于来自中国的高性价比模型，让它们协同完成任务。

综合九项公开基准测试（等权平均），Mio 1.0 Lite 以 **0.0621 元/请求** 的平均费用，取得 **77.15/100** 的平均得分，在 13 个对比方法中同时实现最高平均分和最低平均费用。相较平均得分最高的基线，得分提高 **6.35 分**，平均费用降低 **94.0%**。[查看完整评测结果 →](docs/experimental-results.md#mio-10-lite-public-benchmarks)

![Mio 1.0 Lite：来自中国的高性价比模型协作。九项公开基准测试的平均得分、平均费用，以及三项任务对比。](docs/assets/mio-overview.png)

## 从单个模型到模型群体智能

**模型群体智能（Collective Model Intelligence）** 旨在理解、复用和协调独立开发的模型所积累的能力，在不断演进的模型群体中提升任务效果与资源效率。核心是发挥模型之间的互补性：专长模型可以补足缺失能力，不同模型也可以随任务需求和剩余成本、时延预算的变化，在不同阶段参与协作。

![模型群体智能复用不断扩展的模型群体中的互补能力；模型群体扩展与经验驱动的群体演进属于未来研究方向。](docs/assets/collective-vision.png)

## 系统架构概览

![模型群体智能框架：Model Eval、Mio、Model Space 与 Mos 连接应用、外部模型 API 和自托管运行环境。](docs/assets/system-overview.png)

- **Model Eval**：结合查询级分数估计与跨域预测，构建模型能力画像。
- **Mio（Models in One）**：组织能力互补的模型，根据任务状态与剩余预算进行选择和协作。
- **Mos（Model Serving OS）**：调度和执行评测与应用请求，管理自托管模型的部署与文件分发。

**Model Space** 提供共享的模型身份与资源引用，贯通能力证据、候选池和执行过程。

## RobustRouteBench

RobustRouteBench 包含 **35,072 个实例**，分为 General Text（63 个候选模型）、Multimodal（45 个）和 Chinese（10 个），其中训练集 **14,590 条**、验证集 **1,500 条**、测试集 **18,982 条**。测试集直接对应报告的评测样本，无需再次筛选。

使用 **Python 3.10+**，无需模型 API Key。

```sh
git clone https://github.com/SinapisAI/collective-model-intelligence.git
cd collective-model-intelligence
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,orbit]'
python examples/quickstart.py --scenario text  # 训练并评测 validation
```

其他场景使用 `--scenario multimodal` 或 `--scenario chinese`。

在正式测试集上，按报告的 nominal-budget nAUC 回放 **13 个方法的 65 份预测**：

```sh
python -m examples.reproduce_report --mode replay --output runs/report-replay
```

[数据集与评测协议](benchmark/README.md) · [复现指南](docs/reproducibility.md) · [公开对比方法](methods/README.md)

## 引用与许可

```bibtex
@techreport{sinapisai2026collectivemodelintelligence,
  title       = {Toward Collective Model Intelligence:
                 Characterizing and Coordinating Models},
  author      = {{SinapisAI Team}},
  institution = {SinapisAI},
  year        = {2026},
  url         = {https://github.com/SinapisAI/collective-model-intelligence}
}
```

引用信息见 [CITATION.cff](CITATION.cff)。代码采用 [Apache-2.0](LICENSE)，ORBIT 衍生部分保留 [MIT](methods/orbit/LICENSE)；数据集遵循[上游条款](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/blob/03bd3da2c3bb4a998f73d285c8853c721697e3cb/LICENSES.md)，报告与内嵌标识保留各自署名。
