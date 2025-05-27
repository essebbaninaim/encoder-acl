import sys
from pathlib import Path
import os

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.Utils import getSentence, load_encodeur


import torch
from tqdm import tqdm
from datasets import load_dataset
#from relbert import RelBERT



def train_rel_cls(config):
    if config is not None:
        encodeur_path = os.path.join("../..",config["output"],"encoder_end.pt")
        encodeur, _ = load_encodeur(encodeur_path, config)
    #else :
    #    encodeur = RelBERT('relbert/relbert-roberta-large')
        
    res = ""
    metrics = {
        "accuracy" : [],
        'f1w' : [],
        'f1m' : []
    }

    for dataset_name in ['BLESS', 'CogALexV', 'EVALution', 'K&H+N', 'ROOT09']:
        res += f"{dataset_name} : \n"
        print("Loading dataset >", dataset_name)
        dataset = load_dataset("relbert/lexical_relation_classification", dataset_name)


        train_data = dataset['train']
        test_data = dataset['test']

        all_relations = list(set(train_data['relation']))

        def get_embeddings(data, batch_size=32):
            embeddings = []
            labels = []

            for i in tqdm(range(0, len(data), batch_size), desc="Encoding"):
                batch_data = data[i:i + batch_size]

                for i in range(len(batch_data)):
                    #if isinstance(encodeur,RelBERT):
                    #    h = batch_data['head'][i]
                    #    t = batch_data['tail'][i]
                    #    e = torch.tensor(encodeur.get_embedding([h,t])).unsqueeze(0)
                    #    embeddings.append(e)
                    res = []
                    for t in config["encodeur"]["templates"]:
                        res.append([getSentence({"head" : batch_data["head"][i], "tail" : batch_data["tail"][i]}, t)])
                    with torch.no_grad():
                        embeddings.append(encodeur.forward(res))
                    labels.append(all_relations.index(batch_data["relation"][i]))
                
            embeddings = torch.cat(embeddings,dim=0).cpu().numpy()
            return embeddings, labels

        train_embedding, train_labels = get_embeddings(train_data)
        test_embedding, test_labels = get_embeddings(test_data)

        hidden_dim = 150
        lr = 0.0001

        print("Fitting MLP >")
        model = MLPClassifier(hidden_layer_sizes=hidden_dim, learning_rate='constant', learning_rate_init=lr,batch_size=64)
        model.fit(train_embedding, train_labels)

        print("Predicting >")
        pred = model.predict(test_embedding)

        f1_w = float(f1_score(test_labels, pred, average='weighted'))
        f1_micro = float(f1_score(test_labels, pred, average='micro'))
        n_good = 0
        for i in range(len(pred)):
            p = pred[i]
            reel = test_labels[i]
            if p == reel:
                n_good += 1
        acc = n_good / len(pred)

        f1_w = round(f1_w * 100,2)
        f1_micro = round(f1_micro * 100,2)
        acc = round(acc * 100,2)

        metrics['accuracy'].append(acc)
        metrics['f1m'].append(f1_micro)
        metrics['f1w'].append(f1_w)

        info = f"\t Accuracy : {acc}\n\t f1 weighted : {f1_w}\n\t f1 micro : {f1_micro}\n"
        print(info)
        res += info
    
    res+= str(metrics)
    return res

if __name__ == "__main__":
    train_rel_cls(None)