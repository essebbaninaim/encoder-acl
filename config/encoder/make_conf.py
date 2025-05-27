import os

datasets = {
    "llama3" : "./output/dataset/llama3/merged.csv",
    "relbert" : "./output/dataset/others/relbert_dataset.csv",
    "relbert_and_llama" : "./output/dataset/others/relbert_llama3.csv"
}

config = """{
  "output" : "[OUTPUT]",
  "encodeur" : {
    "model_name": "[MODEL_NAME]",
    "based_encoder": "[BASED_ENCODER]",
    "batch_size": 256,
    "lr": 1e-5,
    "max_length": [MAX_LENGTH],
    "temperature": 0.03,
    "dataset": "[DATASET_PATH]",

    "templates": [TEMPLATES_SENTENCES]
  }
}
"""




bert_templates = {
    "prompt1" : ["The relationship between [HEAD] and [TAIL] is <mask> . "],
    "prompt2" : ["The word that best describe the relationship between [HEAD] and [TAIL] is <mask> . "],
    "prompt3" : ["People often use the word <mask> to describe the relationship between [HEAD] and [TAIL] ."],
    "prompt4" : ["One property of [HEAD] is to be the <mask> of [TAIL] ."],
    "prompt5" : ["One property of [HEAD] is to be the <mask> of [TAIL] .", "Usually, we are [TAIL] <mask> [HEAD] ."],
    "prompt6" : ["One property of [HEAD] is to be the <mask> of [TAIL] .", "Usually, we are [TAIL] <mask> [HEAD] .", "In term of science, [HEAD] is the <mask> of [TAIL]"]
}

semprop_templates = {
    "prompt1" : ["[HEAD] means <mask> .", "[TAIL] means <mask> ."],
}
berts = [
    ["google-bert/bert-base-uncased", "[MASK]"],
    ["google-bert/bert-large-uncased", "[MASK]"],
    ["FacebookAI/roberta-base", "<mask>"],
    ["FacebookAI/roberta-large", "<mask>"],
    ["microsoft/deberta-v3-base", "[MASK]"],
    ["microsoft/deberta-v3-large", "[MASK]"],
]


def create_dir(model_name, template, max_length="64"):
    for dataset_alias in datasets.keys():
        dataset_path = datasets[dataset_alias]

        for bert in berts:
            bert_model = bert[0]
            bert_model_safe = bert_model.replace("/","_")
            bert_mask = bert[1]

            for template_alias in template:
                template_sentences = template[template_alias]
                template_sentences_str="["
                for sentence in template_sentences:
                    template_sentences_str += f"\"{sentence.replace('<mask>', bert_mask)}\","
                template_sentences_str = template_sentences_str[:-1] + "]"

                path = os.path.join(model_name, bert_model_safe, dataset_alias, template_alias)

                output = os.path.join("./output","encoder",path)

                content = config.replace("[OUTPUT]", output).replace("[BASED_ENCODER]", bert_model).replace("[DATASET_PATH]", dataset_path).replace("[TEMPLATES_SENTENCES]", template_sentences_str).replace("[MODEL_NAME]",model_name).replace("[MAX_LENGTH]", max_length)
                
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".json", "w") as file:
                    file.write(content)



create_dir("multibert", bert_templates)
create_dir("semprop", semprop_templates)
