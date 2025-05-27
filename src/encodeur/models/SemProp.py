from utils.Utils import extract_mask
from torch import nn
from transformers import AutoModelForMaskedLM
import torch
from pytorch_metric_learning import losses


class SemProp(nn.Module):
    def __init__(self, tokenizer, temperature=0.05, based_encoder="bert-base-uncased", max_length=64):
        super(SemProp, self).__init__()
        self.tokenizer = tokenizer
        self.MAX_LENGTH = max_length

        self.head_encodeur = AutoModelForMaskedLM.from_pretrained(based_encoder, device_map="cuda", local_files_only=True)
        self.head_encodeur = nn.DataParallel(self.head_encodeur)
        
        self.tail_encodeur = AutoModelForMaskedLM.from_pretrained(based_encoder, device_map="cuda", local_files_only=True)
        self.tail_encodeur = nn.DataParallel(self.tail_encodeur)

        self.loss_fn = losses.NTXentLoss(temperature=temperature)

    def get_emb(self, sentence, encodeur):
        encoding = self.tokenizer(
            sentence, 
            max_length=self.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        inputs_ids = encoding['input_ids'].to("cuda")
        am = encoding['attention_mask'].to("cuda")
        
        out = encodeur(inputs_ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
        mask_token_index = (inputs_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        return extract_mask(out, mask_token_index)
    
    def forward(self, templates):
        assert len(templates) == 2
        
        head_emb = self.get_emb(templates[0], self.head_encodeur)
        tail_emb = self.get_emb(templates[1], self.tail_encodeur)

        return head_emb * tail_emb

    def compute_loss(self, batch):
        embedding = self(batch["anchor"])  
        pos_embedding = self(batch["positive"])

        emb_all = torch.cat([embedding, pos_embedding], dim=0)
        labels = torch.arange(embedding.size(0))
        labels = torch.cat([labels, labels], dim=0)

        return self.loss_fn(emb_all, labels)
