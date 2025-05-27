import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import builtins
import json
import os
import subprocess
import sys
import threading
import time

import requests
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer


def getSentence(row, template):
    return template.replace("[HEAD]", row['head']).replace("[TAIL]", row['tail'])


def is_plural(word, nlp):
    doc = nlp(word)
    for token in doc:
        if token.tag_ in ["NNS", "NNPS"]:
            return True
    return False


def triplet_is_plural(triplet, nlp):
    for word in triplet:
        if not is_plural(word, nlp):
            return False
    return True


def get_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def extract_mask(batch, mask_token_index):
    res = []
    for i in range(batch.shape[0]):
        res.append(batch[i][mask_token_index[i]])
    return torch.stack(res)


def nvidia_smi_task():
    file_name = 'nvidia_smi_output.txt'
    with open(file_name, 'w') as file:
        file.write("--- START TRACKING ---")

    while True:
        with open(file_name, 'a') as file:
            subprocess.run(["nvidia-smi"], stdout=file)
        time.sleep(10)
def get_nvidia_smi_output():
    # Pour avoir la sortie de nvidia smi
    nvidia_smi_thread = threading.Thread(target=nvidia_smi_task)
    nvidia_smi_thread.start()


def fixCluster():
    # Pour fix les utilisations simulatanées sur le cluster
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    os.environ['NO_PROXY'] = 'huggingface.co'

    # Pour fix les logs stuck a cause de hf ("Some weights of the model checkpoint  ...")
    original_print = builtins.print

    def custom_print(*args, **kwargs):
        kwargs["flush"] = True
        original_print(*args, **kwargs)

    builtins.print = custom_print





def read_config(path=None):
    if path is None:
        if len(sys.argv) >= 2:
            path = sys.argv[1]
        else:
            raise Exception("Un argument doit etre fournie. Il s'agit du fichier de configuation")
    with open(path, 'r') as config_file:
        return json.load(config_file), path


def get_encodeur(config):
    from src.encodeur.models.MultiBERT import MultiBERT
    from src.encodeur.models.SemProp import SemProp

    tokenizer = AutoTokenizer.from_pretrained(config['encodeur']['based_encoder'], local_files_only=True)

    model_name = config['encodeur']['model_name']
    if model_name =='multibert':
        model = MultiBERT(tokenizer, temperature=config['encodeur']['temperature'], based_encoder=config['encodeur']['based_encoder'],nb_model=len(config['encodeur']["templates"]), max_length=config['encodeur']["max_length"])
    elif model_name == 'semprop':
        model = SemProp(tokenizer, temperature=config['encodeur']['temperature'], based_encoder=config['encodeur']['based_encoder'], max_length=config['encodeur']["max_length"])
    else:
        raise Exception("Unrecognized model")
    
    return model, tokenizer

def load_encodeur(path, config):
    print("Path : ", path)
    model, tokenizer = get_encodeur(config)

    if torch.cuda.device_count() == 3:
        model.load_state_dict(torch.load(path, weights_only=False))
    else:
        model.load_state_dict(torch.load(path, map_location='cuda:0', weights_only=False))
    model.eval()
    return model, tokenizer


def my_split(ligne, separateurs):
    if len(ligne.split(' ')) > 7:
        return []
    res = []
    w = ""
    for c in ligne:
        if c in separateurs:
            if w != "":
                res.append(w)
                w = ""
        else:
            w += c
    if w != "":
        res.append(w)
    res = [element.strip(' 0123456789.,;:-*|') for element in res if element.strip()]
    return res


def extract_triplet(ligne):
    ligne = ligne.lstrip(' 0123456789.,;:-*|•')
    res = my_split(ligne, ",;:-")
    if len(res) != 3:
        return None
    for i in range(len(res)):
        res[i] = res[i].lower()
        for c in res[i]:
            if c.isspace() or c.isalpha():
                continue
            else:
                return None
    return res


def extract_triplet_v2(ligne):
    try:
        ligne = ligne.lstrip(' 0123456789.,;:-*|')

        head = ligne[ligne.index("between ") + 8: ligne.index(" and")]
        tail = ligne[ligne.index(" and") + 4: ligne.index(" is")]
        relation = ligne[ligne.index(":") + 2:].rstrip(".")

        words_head, words_tail, words_relation = set(head.split()), set(tail.split()), set(relation.split())

        if len(words_head | words_tail | words_relation) != len(words_head) + len(words_tail) + len(words_relation):
            return None

        max = 25
        if len(head) < max and len(tail) < max and len(relation) < max:
            return [head, relation, tail]

        return None
    except Exception as e:
        # print("Erreur lors de l'extraction de la phrase : ", ligne, " erreur : ", e)
        return None


def extract_mentions(output):
    res = []
    for ligne in output.split("\n"):
        if re.match(r"^[1-5]\.", ligne):
            res.append(ligne[3:])
    if len(res) == 5:
        return res
    return None
