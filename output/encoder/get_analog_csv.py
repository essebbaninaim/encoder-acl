import os
import ast

for model_name in [f for f in os.listdir() if os.path.isdir(f)]:
    print(model_name.upper())
    print()
    print(",,, U2, U4, BATS, GOOGLE, SCAN, NELL, T-REX, CN, AVERAGE")
    print()
    for based_encoder in os.listdir(model_name):
        for dataset in os.listdir(os.path.join(model_name,based_encoder)):          
              for prompt in  os.listdir(os.path.join(model_name,based_encoder, dataset)):         
                path = os.path.join(model_name,based_encoder,dataset, prompt,"relbert_analog.txt")

                if os.path.exists(path):
                    with open(path) as file:
                        array = ast.literal_eval(file.readlines()[-1].strip())
                        avg = sum(array) / len(array)
                        avg = round(avg, 2)
                        array.append(avg)
                        last_line = str(array)
                        tab = last_line[1:-1]
                    print(f"{based_encoder}, {dataset}, {prompt}, {tab}")
                else:
                    print(f"{based_encoder}, {dataset}, {prompt}")
        print()
    print("----")