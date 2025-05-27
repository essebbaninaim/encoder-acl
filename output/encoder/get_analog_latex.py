import os
import ast

MODEL_NAME_MAP={
    "multibert" : "MultiBERT",
    "semprop" : "Semantic Properties"
}

DATASET_MAP = {
    "llama3" : "Llama 3",
    "relbert" : "RelBERT",
    "relbert_and_llama" :"Llama 3 and RelBERT"
}

BASED_ENCODER_MAP={
    "FacebookAI_roberta-base"  : "roberta-base",
    "FacebookAI_roberta-large" :  "roberta-large",
    "google-bert_bert-base-uncased" : "bert-base",
    "google-bert_bert-large-uncased" :  "bert-large",
    "microsoft_deberta-v3-base" : "deberta-base",
    "microsoft_deberta-v3-large" : "deberta-large"
}


header=r"""\begin{table*}[]

\centering

{\smaller%
\begin{tabular}{lccccccccc}
    \toprule
    \textbf{} & \textbf{U2} & \textbf{U4} & \textbf{BATS} & \textbf{GOOGLE} & \textbf{SCAN} & \textbf{NELL} & \textbf{T-REX} & \textbf{CN} & \textbf{Average} \\
"""

footer=r"""    \bottomrule
\end{tabular}}
\caption{[MODEL_NAME] performance on analogy with [BASED_ENCODER]}
\end{table*}
"""

raw_start = r"""        \midrule
    \multicolumn{10}{c}{\textit{[DATASET_NAME]}} \\"""

for model_name in [f for f in os.listdir() if os.path.isdir(f)]:
    for based_encoder in os.listdir(model_name):
        print(header)
        for dataset in os.listdir(os.path.join(model_name,based_encoder)):   
            print(raw_start.replace("[DATASET_NAME]", DATASET_MAP[dataset]))

            for prompt in sorted(os.listdir(os.path.join(model_name, based_encoder, dataset)), key=lambda x: int(x.replace("prompt", ""))):
                path = os.path.join(model_name,based_encoder,dataset, prompt,"relbert_analog.txt")

                if os.path.exists(path):
                    with open(path) as file:
                        array = ast.literal_eval(file.readlines()[-1].strip())
                        avg = sum(array) / len(array)
                        avg = round(avg, 2)
                        array.append(avg)
                
                print(f"        {prompt} & {' & '.join(map(str, array))} \\\\")
                

                
        print(footer.replace("[MODEL_NAME]", MODEL_NAME_MAP[model_name]).replace("[BASED_ENCODER]", BASED_ENCODER_MAP[based_encoder]))
