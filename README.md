![Collective Model Intelligence. Characterizing and Coordinating Models.](docs/assets/readme-banner.svg)

# Collective Model Intelligence

**Toward Collective Model Intelligence: Characterizing and Coordinating Models**

[**Tech Report (arXiv)**](https://arxiv.org/abs/2610.24739) · [Results](docs/experimental-results.md) · [Try Mio](https://sinapis.ai) · [Open-Source Dataset](https://huggingface.co/datasets/SinapisAI/RobustRouteBench) · [中文](README.zh-CN.md)

## Mio 1.0 Lite

**Mio (Models in One)** organizes complementary models to work together, choosing which model to call as task needs and remaining budgets change. **Mio 1.0 Lite** applies this approach to cost-efficient models from China.

Across nine equally weighted public benchmarks, Mio 1.0 Lite achieves an average score of **77.15/100** at **0.0621 CNY/request**, the highest average score and lowest average cost among 13 compared methods. Compared with the strongest baseline by average score, Mio scores **6.35 points higher** at **94.0% lower average cost**. [Full benchmark results →](docs/experimental-results.md#mio-10-lite-public-benchmarks)

![Mio 1.0 Lite: collaboration among cost-efficient models from China. Average score and cost across nine public benchmarks, with three selected task comparisons.](docs/assets/mio-overview.png)

## From individual models to collective intelligence

**Collective Model Intelligence** aims to improve task outcomes and resource efficiency by understanding, reusing, and coordinating capabilities across an evolving population of independently developed models. The key is complementarity: a specialist can supply a missing capability, and different models can contribute at different stages as task needs and remaining cost and latency budgets change.

![Collective Model Intelligence reuses complementary capabilities from an expanding model universe; model growth and experience-driven collective evolution are future research directions.](docs/assets/collective-vision.png)

## System Architecture Overview

![Collective Model Intelligence Framework: Model Eval, Mio, Model Space, and Mos connect applications to external model APIs and self-managed runtimes.](docs/assets/system-overview.png)

- **Model Eval** builds capability profiles through query-level score estimation and cross-domain prediction.
- **Mio (Models in One)** organizes complementary models and selects or coordinates them using task state and remaining budgets.
- **Mos (Model Serving OS)** schedules and executes evaluation and application calls, and manages deployment and model-file distribution for self-managed models.

**Model Space** provides shared model identities and resource references, linking capability evidence, candidate pools, and execution.

## RobustRouteBench

RobustRouteBench contains **35,072 instances** across General Text (63 candidate models), Multimodal (45), and Chinese (10): **14,590 training**, **1,500 validation**, and **18,982 test requests**. The test split directly matches the report's evaluation population.

Use **Python 3.10+**. The examples need no model API key.

```sh
git clone https://github.com/SinapisAI/collective-model-intelligence.git
cd collective-model-intelligence
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,orbit]'
python examples/quickstart.py --scenario text  # train and evaluate on validation
```

Other scenarios: `--scenario multimodal` or `--scenario chinese`.

Replay the **65 prediction runs across 13 methods** on the official test split using the report's nominal-budget nAUC:

```sh
python -m examples.reproduce_report --mode replay --output runs/report-replay
```

[Dataset and protocol](benchmark/README.md) · [Reproduction guide](docs/reproducibility.md) · [Comparison methods](methods/README.md)

## Citation and license

```bibtex
@misc{sinapisaiteam2026collectivemodelintelligencecharacterizing,
  title={Toward Collective Model Intelligence: Characterizing and Coordinating Models},
  author={{SinapisAI Team} and Jian-Dong Liu and Qun Dai and Zhendong Du and Guannan Lai and Hao-Yu
          Liu and Yi-Fan Ma and Zi-Yu Mao and Tengzhong Sun and Xin-Yi Zhang and Zi-Chen Zhao and
          Gelin Bian and Zongyuan Cai and Ju Cang and Zhangqin Chen and Shilei Gu and Guoding Ji and
          Taojian Jiang and Hao-Xuan Ma and Fan Wu and Lin-Xing Wu and Tao Yu and Xiao Zhang and
          Tianyi Zhou and Rong Gu and Ming Li and Zhi-Hao Tan and Han-Jia Ye and Yang Yu and
          De-Chuan Zhan},
  year={2026},
  eprint={2610.24739},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.24739},
}
```

Citation metadata: [CITATION.cff](CITATION.cff). Code uses [Apache-2.0](LICENSE), with [MIT-licensed ORBIT components](methods/orbit/LICENSE). The dataset follows its [upstream terms](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/blob/03bd3da2c3bb4a998f73d285c8853c721697e3cb/LICENSES.md); report materials and embedded logos retain their attribution.
