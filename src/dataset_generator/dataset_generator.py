import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from utils.Utils import get_nvidia_smi_output, read_config, extract_triplet, get_bnb, fixCluster, triplet_is_plural
import torch.multiprocessing as mp
import spacy

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['HF_HUB_OFFLINE'] = '1'

fixCluster()


def init_model(device, model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, device_map=device)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only=True, quantization_config=bnb_config)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def build_prompt(system_prompt, relation_obj):
    examples = "\n".join(f"* {example}" for example in relation_obj['examples'])
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

I am interested in {relation_obj['prompt_suffix']}. 
For example:
{examples}

Please generate a bullet list of 100 different examples. <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def process_relation(device_id, relation, nlp, config, triplets):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print("Relation : ", relation, " use device:", device)

    tokenizer, model = init_model(device, config['model_name'])
    
    prompt = build_prompt(config["system_prompt"], config["relation"][relation])

    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    existing_triplets_file = os.path.join(config['output_dir'], relation + ".csv")

    nb_generate = 0
    # Load existing triplets if available
    if os.path.exists(existing_triplets_file):
        with open(existing_triplets_file, 'r', encoding='utf-8') as existing_file:
            for line in existing_file:
                triplet_str = line.rstrip()
                triplets.append(triplet_str)
                nb_generate += 1

    # Generate new triplets and save
    with open(existing_triplets_file, 'a', encoding='utf-8') as csvfile:

        while nb_generate < config['nb_to_generate']:
            print("Relation : ", relation, " length: ", len(triplets))
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=config['max_length'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                pad_token_id=tokenizer.eos_token_id
            )

            decode_output = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for line in decode_output.split('\n'):
                triplet = extract_triplet(line)
                if triplet:
                    triplet_str = '\t'.join(triplet)
                    if triplet_str not in triplets and not triplet_is_plural(triplet, nlp):
                        csvfile.write(f"{triplet_str}\n")

                        triplets.append(triplet_str)
                        nb_generate += 1
            csvfile.flush()
            print(f"{relation} : {nb_generate}")


def worker(device_id, work_queue, nlp, config, triplets):
    while not work_queue.empty():
        try:
            relation = work_queue.get_nowait()
        except mp.Queue.Empty:
            break
        process_relation(device_id, relation, nlp, config, triplets)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    config, config_path = read_config()

    gpu_count = torch.cuda.device_count()

    config['output_dir'] = os.path.join("../..", config['output_dir'])
    os.makedirs(config['output_dir'], exist_ok=True)
    shutil.copy(config_path, config['output_dir'])

    nlp = spacy.load("en_core_web_sm")

    manager = mp.Manager()
    triplets = manager.list()

    work_queue = mp.Queue()
    for relation in config["relation"].keys():
        work_queue.put(relation)

    processes = []
    for device_id in range(gpu_count):
        p = mp.Process(target=worker, args=(device_id, work_queue, nlp, config, triplets))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done. ")
