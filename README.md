# Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality

Replication package for the CASCON 2026 paper.

**Authors:** Suzhen Zhong, Ying Zou, and Bram Adams — Queen’s University, Canada.

- **Dataset:** [CodeChat V2.0](https://huggingface.co/datasets/Suzhen/CodeChat-V2.0)
- **Paper:** [arXiv:2509.10402](https://arxiv.org/abs/2509.10402)
- **Repository:** [CodeChat](https://github.com/Software-Evolution-Analytics-Lab-SEAL/CodeChat)

## Study Overview

Large language models (LLMs) support conversational coding assistance, including code generation, technical questions, and iterative problem solving. This study examines the topics users discuss with LLMs, how engagement varies across topics, and how generated-code quality evolves across conversational turns.

We construct **CodeChat**, a dataset derived from WildChat containing **587,568 real-world developer–LLM conversations** and **1,724,902 LLM-generated code snippets** across **more than 20 programming languages**. We identify common topics and interaction patterns, then assess generated code across Python, JavaScript, C++, Java, and C#. The dataset used in the paper is available as **CodeChat V2.0**.

## Research Questions

- **RQ1 — Common Topics:** What topics do developers most commonly prompt when interacting with LLMs?
- **RQ2 — Generated Code Quality:** How high is the quality of LLM-generated code within coding conversations?

## Dataset Overview

The paper studies **CodeChat V2.0**, derived from WildChat. The statistics below match the CASCON 2026 camera-ready paper (Table I).

| Field         | Value                                                    |
|---------------|----------------------------------------------------------|
| Conversations | 587,568 (18.4% of WildChat)                              |
| Turns         | 1.1 million conversational turns                         |
| Code snippets | 1,724,902 LLM-generated code snippets                    |
| Users         | 426,062 unique hashed IP addresses (proxy for users)     |
| Period        | April 2023 – July 2025 (28 months)                       |
| Source        | WildChat; 3.2 million conversations in the source collection described in the paper |
| Languages     | 20+ (Python, JavaScript, Java, C++, C#, etc.)            |

We retain a conversation when at least one LLM response contains a code block enclosed in triple backticks followed by a programming-language identifier. A conversational turn consists of one user prompt and one LLM response. In the paper, “developer” refers to the user participating in a code-related conversation.

## Analysis and Main Findings

- **Topics and engagement (RQ1):** We apply BERTopic to English initial prompts, use Scott–Knott grouping to compare turn counts across topics, and manually analyze prompt-gap sequences in the topic with the highest mean turn count. Web design and development (9.6%) and machine learning model training and AI bot deployment (8.7%) are the most frequent topics. AI-augmented business tools and strategy automation has the highest mean turn count (3.40); repeated use-case changes are the most frequent three-gap pattern within that topic.
- **Generated-code quality (RQ2):** We use Pylint, ESLint, Cppcheck, PMD, and Roslyn to assess Python, JavaScript, C++, Java, and C# code, respectively. C4 code-clone detection identifies task sequences across adjacent responses. The longitudinal analysis compares Turns 1–5 within task sequences containing at least five turns. Static-analysis issue prevalence does not consistently decrease with additional turns.
- **Syntax-error resolution:** Among 383 sampled transitions from responses with detected syntax errors to responses without them, “Point out mistake then request fix” is the most frequent follow-up prompt category (23.0%). This describes its frequency among successful resolutions, rather than comparative effectiveness across all correction attempts.

These findings suggest that conversational assistants should track evolving developer intent and monitor code quality across turns. The quality analysis concerns issues detected by static analyzers; it does not establish overall functional correctness.

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
