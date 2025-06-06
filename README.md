# Official repository for the paper "Modelling Complex Semantics Relation with Contrastively Fine-Tuned Relational Encoders" in ACL 2025

This repository contains the official codebase and configuration files for reproducing the experiments and results from our ACL 2025 paper.

The dataset can be found in ```./output/dataset/llama3/```
## Getting Started

To run the main pipeline:
```python run.py [path_to_config.json]```

Make sure to provide the appropriate configuration file path. Example config files are available in the config/ directory.

Below is a sample configuration to help you get started:
```
{
  "output" : "./output/encoder/multibert/FacebookAI_roberta-base/llama3/prompt6",
  "encodeur" : {
    "model_name": "multibert",
    "based_encoder": "FacebookAI/roberta-base",
    "batch_size": 256,
    "lr": 1e-5,
    "max_length": 64,
    "temperature": 0.03,
    "dataset": "./output/dataset/llama3/merged.csv",

    "templates": ["One property of [HEAD] is to be the <mask> of [TAIL] .","Usually, we are [TAIL] <mask> [HEAD] .","In term of science, [HEAD] is the <mask> of [TAIL]"]
  }
}
```

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

