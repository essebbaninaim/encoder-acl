import sys
import os

sys.path.extend([os.path.join(os.getcwd(), d) for d in ("encodeur", "cls", "utils")])

from encodeur.train_encodeur import train_encodeur
from relation_classification.rel_cls import train_rel_cls
from utils.Utils import read_config
from analog.analog_test import analog_test


if __name__ == "__main__":
    print("Set up")
    config, config_path = read_config()

    # TRAIN
    print("Start training")
    os.chdir("./encodeur")
    train_encodeur(config, os.path.join("..", config_path))

    # EVAL
    print("Start evaluating")
    res = analog_test(config)
    with open(os.path.join("../..", config["output"], "relbert_analog.txt"), "w") as file:
        file.write(res)

    res = train_rel_cls(config)
    with open(os.path.join("../..", config["output"], "relbert_cls.txt"), "w") as file:
        file.write(res)

    
