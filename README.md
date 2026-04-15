# LLM-Enhanced Log Anomaly Detection: A Comprehensive Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2604.12218)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive benchmark study evaluating Large Language Models, fine-tuned transformers, and traditional methods for automated system log anomaly detection.

## Key Findings

- **Fine-tuned DeBERTa-v3** achieves the highest accuracy (F1: 95.3-98.9%)
- **GPT-4 zero-shot** shows remarkable performance (F1: 81.2-88.3%) without training data
- Our **Structured Log Context Prompting (SLCP)** improves LLM zero-shot performance by 2.9-3.1%
- Traditional **Drain + Random Forest** remains competitive (F1: 86.4-95.1%) with minimal latency

## Datasets

All datasets are publicly available from [LogHub](https://github.com/logpai/loghub):
- **HDFS**: Hadoop Distributed File System (11M+ log messages)
- **BGL**: Blue Gene/L supercomputer (4.7M+ messages)
- **Thunderbird**: Sandia National Labs (211M+ messages)
- **Spirit**: Spirit supercomputer (272M+ messages)

## Quick Start

```bash
# Clone
git clone https://github.com/dishapatel/llm-log-anomaly-benchmark.git
cd llm-log-anomaly-benchmark

# Install dependencies
pip install -r requirements.txt

# Run traditional ML experiments
python src/run_experiments.py

# Run transformer fine-tuning
python src/run_transformers.py

# Generate figures
python src/benchmark.py --generate-figures
```

## Methods Evaluated

| Category | Methods |
|----------|---------|
| Traditional | Drain + {LR, RF, SVM, IF} |
| Fine-tuned Transformers | BERT, RoBERTa, DeBERTa-v3 |
| Prompt-based LLMs | GPT-3.5, GPT-4, LLaMA-3 (zero/few-shot + SLCP) |

## Results

| Method | HDFS | BGL | Thunderbird | Spirit |
|--------|------|-----|-------------|--------|
| Drain + RF | 95.1 | 91.2 | 88.6 | 86.4 |
| DeBERTa-v3 | **98.9** | **97.4** | **96.1** | **95.3** |
| GPT-4 + SLCP (5-shot) | 93.8 | 91.5 | 89.7 | 87.9 |

## Citation

```bibtex
@article{patel2026llm,
  title={LLM-Enhanced Log Anomaly Detection: A Comprehensive Benchmark of Large Language Models for Automated System Diagnostics},
  author={Patel, Disha},
  journal={arXiv preprint arXiv:2604.12218},
  year={2026}
}
```

## License

MIT License
