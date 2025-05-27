import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'encodeur', 'models'))

import torch
import torch.nn.functional as F
from datasets import load_dataset

from utils.Utils import load_encodeur, getSentence

from tqdm.auto import tqdm

tqdm.pandas()
#from relbert import RelBERT


def encode(head, tail, encodeur, config):
    res = []
    for t in config["encodeur"]["templates"]:
        res.append([getSentence({"head" : head, "tail" : tail}, t)])
        
    with torch.no_grad():
        return encodeur.forward(res)


def analog_test(config):
    encodeur_path = os.path.join("../..",config["output"],"encoder_end.pt")

    encodeur, _ = load_encodeur(encodeur_path, config)

    resultat = ""
    resultat_tab = []
    for dataset_name in ['u2', 'u4', 'bats', 'google', 'scan',
                         'nell_relational_similarity',
                         't_rex_relational_similarity', 'conceptnet_relational_similarity']:

        dataset = load_dataset("relbert/analogy_questions", dataset_name,trust_remote_code=True, download_mode="reuse_cache_if_exists")[
            "test" if dataset_name != "semeval2012_relational_similarity" else "validation"]

        accruracy_per_prefix={}
        prefix_groups = {}
        for item in dataset:
            prefix = item["prefix"]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(item)
                                         

        n_good = 0
        n_tested = 0
        for prefix, items in prefix_groups.items():
            n_good_prefix = 0
            n_tested_prefix = 0
            loop = tqdm(items, desc=f"Testing : {dataset_name} - Prefix: {prefix}")

            for item in loop:
                answer = item["answer"]

                embedding = encode(item["stem"][0], item["stem"][1], encodeur, config)
                cos_sim = []
                for candidate in item["choice"]:
                    candidate_embedding = encode(candidate[0], candidate[1], encodeur, config)
                    cos_sim.append(F.cosine_similarity(embedding, candidate_embedding).item())
                
                max_index = cos_sim.index(max(cos_sim))

                if max_index == answer:
                    n_good += 1
                    n_good_prefix += 1

                n_tested += 1
                n_tested_prefix += 1
                loop.set_postfix(accuracy=n_good / n_tested)

            accuracy_prefix = n_good_prefix / n_tested_prefix
            accruracy_per_prefix[prefix] = round(accuracy_prefix * 100, 2)

        dataset_accuracy = n_good / n_tested if n_tested > 0 else 0
        resultat += f"{dataset_name} -> {dataset_accuracy:.2f}\n"
        resultat_tab.append(round(dataset_accuracy * 100, 2))

        print(f"Results for {dataset_name} : {dataset_accuracy}")
        for prefix, accuracy in accruracy_per_prefix.items():
            print(f"    {prefix}: {accuracy}")
            resultat += f"  {prefix} : {accuracy}\n"

    resultat += f"{resultat_tab}"
    return resultat

