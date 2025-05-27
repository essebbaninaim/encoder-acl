from utils.Utils import extract_mask
from torch import nn
from torch.nn.functional import normalize
from transformers import AutoModelForMaskedLM
import torch
from pytorch_metric_learning import losses


class MultiBERT(nn.Module):
    def __init__(self, tokenizer, temperature=0.05, based_encoder="bert-base-uncased", nb_model=5, max_length=64):
        super(MultiBERT, self).__init__()
        self.tokenizer = tokenizer
        self.MAX_LENGTH = max_length
        self.nb_model = nb_model

        self.encodeurs = nn.ModuleList()
        for _ in range(nb_model): 
            encodeur = AutoModelForMaskedLM.from_pretrained(based_encoder, device_map="cuda", local_files_only=True)
            encodeur = nn.DataParallel(encodeur)
            self.encodeurs.append(encodeur)
        
        self.weights_layer = nn.Linear(nb_model, nb_model, bias=False).to("cuda")

        self.loss_fn = losses.NTXentLoss(temperature=temperature)

    def forward(self, templates):
        assert len(templates) == len(self.encodeurs)

        embeddings = []
        for i in range(len(templates)):
            sentence = templates[i]

            encoding = self.tokenizer(
                sentence, 
                max_length=self.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            inputs_ids = encoding['input_ids'].to("cuda")
            am = encoding['attention_mask'].to("cuda")
            
            out = self.encodeurs[i](inputs_ids, attention_mask=am, output_hidden_states=True).hidden_states[-1]
            mask_token_index = (inputs_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

            emb = extract_mask(out, mask_token_index)
            embeddings.append(emb)

        
        embeddings = torch.stack(embeddings, dim=0).transpose(0, 1)  # (batch_size, nb_model, hidden_size)
        embeddings = normalize(embeddings, p=2, dim=-1)

        if self.nb_model == 1:
            embeddings = embeddings.squeeze(1) # passer de (batch_size, 1, hidden_size) a (batch_size, hidden_size)
            return embeddings
        
        # Calcul des pondérations
        weights = self.weights_layer(torch.ones(1, self.nb_model).to("cuda"))  
        weights = torch.sigmoid(weights).view(-1)  

        weighted_embeddings = embeddings * weights.unsqueeze(0).unsqueeze(-1) 

        final_embedding = weighted_embeddings[:, 0, :]
        for i in range(1, embeddings.size(1)):
            final_embedding = torch.cat((final_embedding, embeddings[:, i, :]), dim=-1)

        return final_embedding

    def compute_loss(self, batch):
        embedding = self(batch["anchor"])  
        pos_embedding = self(batch["positive"])

        emb_all = torch.cat([embedding, pos_embedding], dim=0)
        labels = torch.arange(embedding.size(0))
        labels = torch.cat([labels, labels], dim=0)

        return self.loss_fn(emb_all, labels)
