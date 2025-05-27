import os
import pandas as pd

dataset_dir = '../../output/dataset'
available_files = [f for f in os.listdir(dataset_dir)]
for idx, filename in enumerate(available_files, 1):
    print(f"{idx}. {filename}")

user_choice = int(input(f"Select a folder: (1-{len(available_files)}): "))
if not (1 <= user_choice <= len(available_files)):
    raise Exception("Unvalid answer.")
selected_file = available_files[user_choice - 1]
dossier_csv = os.path.join(dataset_dir, selected_file)

dossier_csv = '../../output/dataset/llama3'

fichiers_csv = [f for f in os.listdir(dossier_csv) if f.endswith('.csv')]

donnees = []

for fichier in fichiers_csv:
    chemin_complet = os.path.join(dossier_csv, fichier)
    df = pd.read_csv(chemin_complet, delimiter='\t',
                                     header=None, names=["head", "relation", "tail"], dtype=str).astype(str)
    donnees.append(df)

concatenated_df = pd.concat(donnees)

donnees_fusionnees = concatenated_df.sample(frac=1).reset_index(drop=True)

donnees_fusionnees.to_csv(os.path.join(dossier_csv, 'merged.csv'), index=False,sep='\t',header=False)

print("Le fichier merged.csv a été créé avec succès.")
