# Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality

We construct **CodeChat**, a dataset derived from WildChat containing **587,568 real-world developer–LLM conversations** and **1,724,902 LLM-generated code snippets** across **more than 20 programming languages**.

- **Dataset:** [CodeChat V2.0](https://huggingface.co/datasets/Suzhen/CodeChat-V2.0)
- **Paper:** [arXiv:2509.10402](https://arxiv.org/abs/2509.10402)

## Research Questions

- **RQ1 — Common Topics:** What topics do developers most commonly prompt when interacting with LLMs?
- **RQ2 — Generated Code Quality:** How high is the quality of LLM-generated code within coding conversations?

## Access the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("Suzhen/CodeChat-V2.0")
```

## Repository Structure

| Folder | Contents |
|---|---|
| `Dataset_statistics/` | Dataset statistics, language distributions, and supporting data. |
| `RQ1/` | Common topics and interaction patterns: BERTopic training, topic plots, prompt-gap analysis, and Scott–Knott grouping. |
| `RQ2/` | Generated-code quality: static-analysis results, changes across turns, and follow-up analysis. |

Internal result-folder and output-file names retain their earlier numbering so existing script paths remain unchanged.

## Requirements

- **Python 3.10+** with `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `tiktoken`
- **BERTopic / UMAP** (RQ1 topic modeling) — see `umap_env.yml`
- **R** (Scott–Knott turn-per-topic analysis and some plots): `RQ1/2_analyzing/3_plot_groups_turn.R`
- **Linter toolchains (RQ2):** Pylint (Python), ESLint + Node.js (JavaScript), Cppcheck (C++), PMD (Java), Roslyn / .NET SDK (C#). The published analysis results are under `RQ2/rq3_results/`.
- **C4 clone detector:** used for task-continuity analysis in the paper; see the C4 reference for the tool. The published package includes the resulting analysis data under `RQ2/rq3_results/`.
