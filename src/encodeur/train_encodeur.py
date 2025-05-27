import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.Utils import  read_config, get_encodeur

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import shutil

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader



from MultiPromptData import MultiPromptData

import matplotlib.pyplot as plt
import copy


def train_encodeur(config, config_path):
    print(f"{torch.cuda.device_count()} GPUs found")

    based_encoder = config['encodeur']['based_encoder']
    dataset = os.path.join("../..", config['encodeur']['dataset'])
    templates=config['encodeur']["templates"]
    batch_size = config['encodeur']['batch_size']
    output = os.path.join("../..", config['output'])
    lr = config['encodeur']['lr']

    print("Runnnig with >")
    print(config)
    
    print("Dataloader >")
    dataloader = DataLoader(
        dataset=MultiPromptData(dataset, templates=templates),
        batch_size=batch_size,
        num_workers=2 ,
        pin_memory=True
    ) 


    print("Model based on ",based_encoder, " >")
    model, tokenizer = get_encodeur(config)


    print("Fine tuning du modèle >")
    os.makedirs(output, exist_ok=True)
    shutil.copy(config_path, output)

    optimizer = AdamW(model.parameters(), lr=lr)
    patience = 30

    all_loss = []
    graph_loss = []
    x = []
    best_loss = float('inf')
    batch_no_imporove = 0
    nb_batch = 0
    early_stop = False
    epoch = 0
    save_model = None

    model.train()
    while True:
        if early_stop or epoch == 10:
            print("Early stopping triggered at epoch : ", epoch)
            break
        print(f"Epoch : {epoch + 1}")

        running_loss = 0.0
        loop = tqdm(dataloader, desc="Training")
        for batch in loop:
            optimizer.zero_grad()
            
            loss = model.compute_loss(batch)

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), batch_no_imporove=batch_no_imporove)
            loss.backward()
            optimizer.step()

            graph_loss.append(loss.item())
            x.append(len(graph_loss) / len(dataloader))

            nb_batch += 1
            if loss.item() < best_loss:
                best_loss = loss
                save_model = copy.deepcopy(model.state_dict())
                batch_no_imporove = 0
            else:
                batch_no_imporove += 1

            if batch_no_imporove == patience:
                early_stop = True
                break

        e_loss = running_loss / min(len(dataloader), nb_batch)
        print(f"Average Loss for Epoch {epoch + 1}: {e_loss:.8f}")
        all_loss.append(f"{e_loss:.3f}")
        epoch += 1

    # Sauvegarde
    print("Saving model >")
    save_model_path = os.path.join(output, f"encoder_end.pt")
    torch.save(save_model, save_model_path)
    print("Save at : ", save_model_path)
    model.eval()

    # Création du graphique
    print("Creating loss graph >")
    plt.figure(figsize=(10, 5))
    plt.plot(x, graph_loss, marker='o')
    plt.title('Evolution de la perte pendant l\'entraînement')
    plt.xlabel('Epoch')
    plt.ylabel('Perte moyenne')
    plt.grid(True)
    plt.savefig(os.path.join(output, f"loss_graph.png"))
    plt.close()
    return all_loss


if __name__ == "__main__":
    config, config_path = read_config()
    train_encodeur(config, config_path)


