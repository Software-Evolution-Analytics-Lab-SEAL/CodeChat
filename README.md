---
license: odc-by
---

<p align="center">
  <img src="https://huggingface.co/datasets/Suzhen/CodeChat/resolve/main/CodeChat_LOGO.png" width="320">
</p>

# CodeChat: Developer–LLM Conversations Dataset

Empirical Study of Developer–LLM Interactions and Generated Code Quality

[![arXiv](https://img.shields.io/badge/arXiv-2509.10402-b31b1b.svg)](https://arxiv.org/abs/2509.10402)
[![GitHub](https://img.shields.io/badge/GitHub-CodeChat-blue?logo=github)](https://github.com/Software-Evolution-Analytics-Lab-SEAL/CodeChat)

---

- **Paper:** [https://arxiv.org/abs/2509.10402](https://arxiv.org/abs/2509.10402)  
- **GitHub:** [https://github.com/Software-Evolution-Analytics-Lab-SEAL/CodeChat](https://github.com/Software-Evolution-Analytics-Lab-SEAL/CodeChat)


## Abstract

**CodeChat** is a large-scale dataset comprising 82,845 real-world developer–LLM conversations, containing 368,506 code snippets generated across more than 20 programming languages, derived from the WildChat dataset. The dataset enables empirical analysis of how developers interact with LLMs during real coding workflows.

---

## Dataset Overview

| Field      | Value                                         |
|------------|-----------------------------------------------|
| Records    | 82,845 conversations                          |
| Code       | 368,506 code snippets                         |
| Languages  | 20+ (Python, JavaScript, Java, C++, C#, etc.) |
| Source     | WildChat                                      |
| Format     | JSON (multi-turn conversations)               |
| License    | ODC-BY (Open Data Commons Attribution)        |

---

## Data Structure

Each entry in CodeChat is a full developer–LLM conversation, with turn-by-turn dialogue and associated code snippets.

---


## How to Use
```python
from datasets import load_dataset

ds = load_dataset("Suzhen/CodeChat")
print(ds['train'][0])


# To load the new v2.0 version:
ds_v2 = load_dataset("Suzhen/CodeChat", "v2.0"
```


---

## Citation

If you use this paper, please cite:

```bibtex
@misc{zhong2025developerllmconversationsempiricalstudy,
      title={Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality}, 
      author={Suzhen Zhong and Ying Zou and Bram Adams},
      year={2025},
      eprint={2509.10402},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2509.10402}, 
}
```

---

## Reference Paper (WildChat Dataset)

[![arXiv](https://img.shields.io/badge/arXiv-2405.01470-b31b1b.svg)](https://arxiv.org/abs/2405.01470)
[![HuggingFace](https://img.shields.io/badge/HF-dataset-orange?logo=huggingface)](https://huggingface.co/datasets/allenai/WildChat)

- **Paper:** [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470)
- **Dataset:** [https://huggingface.co/datasets/allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat)

## License

This dataset is licensed under the [Open Data Commons Attribution License (ODC-BY)](https://opendatacommons.org/licenses/by/).
