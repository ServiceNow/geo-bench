import multiprocessing
import time
import subprocess
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import defaultdict


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def generate_hp_experiments(args):

    cmd = (
        "python {task} "
        "--dataset {ds} --data_dir {dd} "
        "--backbone_type {bbt} "
        "--ckpt_path {cp} "
        "{ft} "
        "--lr {lr}"
        "--bb_lr {bblr}"
        "--weight_decay {wd}\n"
    )

    if args.dataset == "oscd":
        cmd = cmd.format_map(SafeDict(task="train/main_segmentation.py", ds="oscd", dd=args.data_dir))
    else:
        cmd = cmd.format_map(SafeDict(task="train/main_classification.py", ds=args.dataset, data_dir=args.data_dir))
    cmd = cmd.format_map(SafeDict(bbt=args.backbone_type, cp=args.ckpt_path))

    cmd1 = ""
    cmd2 = ""
    if "lp" in args.finetune:
        cmd1 = cmd.format_map(SafeDict(ft=""))
    if "ft" in args.finetune:
        cmd2 = cmd.format_map(SafeDict(ft="--finetune"))
    cmd = cmd1 + cmd2

    vals = args.lr.split(",")
    cmd_list = [cmd] * len(vals)
    for i in range(len(vals)):
        cmd_list[i] = cmd_list[i].format_map(SafeDict(lr=vals[i]))
    cmd = "".join(cmd_list)

    vals = args.bb_lr.split(",")
    cmd_list = [cmd] * len(vals)
    for i in range(len(vals)):
        cmd_list[i] = cmd_list[i].format_map(SafeDict(bblr=vals[i]))

    cmd = "".join(cmd_list)
    vals = args.weight_decay.split(",")
    cmd_list = [cmd] * len(vals)
    for i in range(len(vals)):
        cmd_list[i] = cmd_list[i].format_map(SafeDict(wd=vals[i]))

    cmd = "".join(cmd_list)
    return cmd


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mode", type=str, default="generate")
    parser.add_argument("--dataset", type=str, default="oscd")
    parser.add_argument("--data_dir", type=str, default="datasets/oscd")
    parser.add_argument("--backbone_type", type=str, default="imagenet")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/seco_resnet18_1m.ckpt")

    parser.add_argument("--finetune", type=str, default="lp,ft", help="lp for linear probing, ft for finetuning")
    parser.add_argument("--lr", type=str, default="0.001,0.0001")
    parser.add_argument("--bb_lr", type=str, default="0.001,0.0001")
    parser.add_argument("--weight_decay", type=str, default="0.0001")

    parser.add_argument("--out", type=str, default="hp_search.txt")

    args = parser.parse_args()

    if args.mode == "generate":

        cmd = generate_hp_experiments(args)

        with open(args.out, "w") as f:
            f.write(cmd.strip())
