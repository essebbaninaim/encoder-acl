# Official repo for the paper "Modelling Complex Semantics Relation with Contrastively Fine-Tuned Relational Encoders" in ACL 2025

This repository contains the official codebase and configuration files for reproducing the experiments and results from our ACL 2025 paper.

## Getting Started

To run the main pipeline:
```python run.py [path_to_config.json]```

Make sure to provide the appropriate configuration file path. Example config files are available in the config/ directory.


## LLM-Based Evaluation
We provide scripts for evaluating relational encoders using large language models on analogy and relation classification tasks.


Analogy Evaluation

```
cd utils/analogie
python analogie_llm.py
```

Relation Classification Evaluation

```
cd utils/relation
python relation_llm.py
```

