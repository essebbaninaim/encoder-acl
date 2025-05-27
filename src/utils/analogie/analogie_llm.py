from transformers import pipeline
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
from openai import OpenAI
import os

prompt_item = """You are tasked with identifying the most analogous pair of words based on the given example. 
The example pair is: [HEAD] and [TAIL]. 

From the following list, choose the most fitting pair:

[CANDIDATE]
Only provide the number of the correct answer :"""

def create_prompt(head, tail, candidates):
    candidate_str=""
    for index, candidate in enumerate(candidates,start=1):
        candidate_str+= f"{index}. {candidate[0]} and {candidate[1]}\n"
    
    return prompt_item.replace("[HEAD]", head).replace("[TAIL]", tail).replace("[CANDIDATE]", candidate_str)

def get_res(pipe, inputs):
    if not isinstance(pipe, OpenAI):
        return pipe(inputs)[0]["generated_text"][-1]["content"]
    else:
        response = pipe.chat.completions.create(model="gpt-4o", messages=inputs, max_tokens=1)
        return response.choices[0].message.content[0]
    
def main(args):
    #load_dotenv()

    if args.llm == "gpt-4o":
        pipe = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        pipe = pipeline("text-generation", model=args.llm)

    for dataset_name in ['u2', 'u4', 'bats', 'google', 'scan',
                         'nell_relational_similarity',
                         't_rex_relational_similarity', 'conceptnet_relational_similarity']:

        dataset = load_dataset("relbert/analogy_questions", dataset_name, trust_remote_code=True, download_mode="reuse_cache_if_exists")
        
        examples = []
        for ex in dataset["validation"].select(range(args.k)):
            examples.append({"role": "user", "content": create_prompt(ex["stem"][0], ex["stem"][1], ex["choice"])})
            examples.append({"role": "assistant", "content": str(ex["answer"] + 1)})
 
        nb_good = 0
        nb_test = 0
        if args.llm == "gpt-4o":
            test_data = dataset["test"].select(range(100))
        else:
            test_data = dataset["test"].select(range(100))
            
        loop = tqdm(test_data, desc=f"Testing : {dataset_name}")
        for item in loop:
            answer = item["answer"]

            question = {"role": "user", "content": create_prompt(item["stem"][0], item["stem"][1], item["choice"])}
            inputs = examples + [question]

            llm_answer = get_res(pipe, inputs)

            try:
                if int(llm_answer.strip()) == int(answer) + 1:
                    nb_good += 1
            except:
                pass
            
            nb_test += 1
            loop.set_postfix(accuracy= round((nb_good / nb_test)*100, 2))

            if args.log:
                path = os.path.join("log", args.llm.replace("/","_"), str(args.k), dataset_name, f"{item['stem'][0]}_{item['stem'][1]}")
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "question.txt"), "w") as f:
                    f.write(str(inputs))

                with open(os.path.join(path, "res.txt"), "w") as f:
                    f.write(llm_answer)


        print(f"Accuracy with {dataset_name} : ", round((nb_good / nb_test) * 100, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analogie test with LLM")
    parser.add_argument("--k", type=int, default=0, help="k shot (default : 0)")
    parser.add_argument("--llm", type=str, default="unsloth/Llama-3.3-70B-Instruct-bnb-4bit", 
                        help="LLM (default : unsloth/Llama-3.3-70B-Instruct-bnb-4bit).")
    parser.add_argument('--log', action=argparse.BooleanOptionalAction)

    
    main(parser.parse_args())