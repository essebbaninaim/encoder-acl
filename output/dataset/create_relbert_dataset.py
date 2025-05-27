import pandas as pd
import json

# default : 3021



train_csv = pd.read_csv("./llama3/merged.csv", delimiter='\t', header=None, names=["head", "relation", "tail"], dtype=str).astype(str)

relation_counts = train_csv['relation'].value_counts()
relation_to_remove = relation_counts[relation_counts == 1].index
#relation_to_remove = relation_counts[(relation_counts >= 500) | (relation_counts == 1)].index
train_csv = train_csv[~train_csv['relation'].isin(relation_to_remove)]

res = []

for relation in train_csv['relation'].unique():
    positives=[]
    negatives=[]
    
    pos_data = train_csv[train_csv['relation']==relation]
    neg_data = train_csv[train_csv['relation']!=relation]

    for i, (_, item) in enumerate(pos_data.iterrows()):
        if i >= 8:
            break
        positives.append([item['head'], item['tail']])

        sample = neg_data.sample(n=1).iloc[0]
        negatives.append([sample['head'], sample['tail']])
    
    obj = {
        'relation_type' : relation,
        'positives' : positives,
        'negatives' : negatives
    }
    if len(positives)==0==[] or len(negatives)==0:
        print("wsh t nul")
        exit()
    res.append(obj)

print(len(res))
with open("dataset.json", "w") as fichier_json:
    json.dump(res, fichier_json, indent=4)