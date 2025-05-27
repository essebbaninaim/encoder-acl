# Encoder


Creating config files:
```cd config/encoder```

```python make_conf.py```

Run encoders xp with all differents
```cd ../../src/```

 ```find ../config/encoder -type f -name "*.json" -exec sbatch -A <account> run.sh {} \;```

 Run LLM Evaluation

```cd utils/analogie && sbatch -A <account> analogie_llm.sh```

```cd utils/relation && sbatch -A <account> relation_llm.sh```

