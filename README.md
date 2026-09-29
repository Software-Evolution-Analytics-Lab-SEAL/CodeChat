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

## Paper-to-artifact map

The paper has two research questions, organized here as `RQ1_topics` and `RQ2_code_quality`. Some output filenames retain earlier RQ numbering; the mapping below identifies their corresponding results in the final paper.

| Paper result | Package file |
|---|---|
| **Fig. 3** | `0_data_processing/rq0_results/RQ0_turns.png` |
| **Fig. 4** | `0_data_processing/rq0_results/RQ0_lan.png` |
| **Fig. 5(a)** | `RQ1_topics/rq1_results/RQ2_TopicsFullturn_v2.png` |
| **Fig. 5(b)** | `RQ1_topics/rq1_results/RQ2_TurnsInTopicsFullturn.png` |
| **Fig. 6** | `RQ1_topics/4_prompt_gaps/results/RQ2_WhyLongTurn.png` |
| **Table II** | `RQ1_topics/rq1_results/RQ1_PromptGaps.tex` (definitions, adapted from prior work) |
| **Fig. 7** | `RQ2_code_quality/rq2_results/RQ3_3SingleIssueCompare.png` |
| **Table III** | `RQ2_code_quality/rq2_results/RQ3_linter_percent_tab.tex` |
| **Table IV** | `RQ2_code_quality/rq2_results/RQ3_WhyErrorReduce.tex` |

## Requirements

- **Python 3.10+** with `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `tiktoken`
- **BERTopic / UMAP** (RQ1 topic modeling) — see `umap_env.yml`
- **R** (Scott–Knott turn-per-topic analysis and some plots): `RQ1_topics/2_analyzing/0_4plot_turn_distri_shifted.R`
- **Linter toolchains (RQ2):** Pylint (Python), ESLint + Node.js (JavaScript), Cppcheck (C++), PMD (Java), Roslyn / .NET SDK (C#). Each linter subfolder ships its config (`.pylintrc`, `package.json`, etc.); run `npm install` / `dotnet restore` before use.
- **C4 clone detector** for `RQ2_code_quality/2_clone_detection` (see the C4 reference in the paper).
