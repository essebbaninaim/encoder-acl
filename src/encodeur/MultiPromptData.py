import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.Utils import getSentence


class MultiPromptData(Dataset):
    def __init__(self, dataset_path, templates=None):
        super(MultiPromptData, self).__init__()
        self.TEMPLATES = templates

        self.train_csv = pd.read_csv(dataset_path, delimiter='\t',
                                     header=None, names=["head", "relation", "tail"], dtype=str).astype(str)
        self.train_csv = self.train_csv.sample(frac=1).reset_index(drop=True)

        self.relation_groups = self.train_csv.groupby('relation')

    
    def __len__(self):
        return len(self.train_csv)


    def get_json(self, row):
        res = []
        for t in self.TEMPLATES:
            res.append(getSentence(row, t))
        return res
    
    def __getitem__(self, index):
        row_anchor = self.train_csv.iloc[index]
            
        return {
            "anchor" : self.get_json(row_anchor),
            "positive" : self.get_json(self.relation_groups.get_group(row_anchor['relation']).sample(n=1).iloc[0]),
            #"label": row_anchor['relation_id']
        }
