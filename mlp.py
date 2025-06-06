#!python

import argparse

from src.split import split
from src.train import train
from src.eval import eval


def all(args):
    split(args)
    args.file = "./modefile"
    train(args)
    eval(args)


def main():
    actions = {"split": split, "train": train, "eval": eval, "all": all}
    parser = argparse.ArgumentParser(prog="42mlp", description="42 Multilayer Perception")
    parser.add_argument("mode", choices=actions.keys())
    parser.add_argument("file")
    args = parser.parse_args()

    actions[args.mode](args)


if __name__ == "__main__":
    # try:
        main()
    # except Exception as error:
        # print(type(error).__name__ + ":", error)
