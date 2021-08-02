import multiprocessing
import time
import subprocess
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product


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

    # parser.add_argument("--mode", type=str, default="generate")
    parser.add_argument("--dataset", type=str, default="oscd")
    parser.add_argument("--data_dir", type=str, default="datasets/oscd")
    parser.add_argument("--backbone_type", type=str, default="imagenet")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/seco_resnet18_1m.ckpt")

    parser.add_argument("--ft", type=str, default="ft,lp", help="")
    parser.add_argument("--lr", type=str, default="0.001,0.0001")
    parser.add_argument("--bb_lr", type=str, default="0.001,0.0001")
    parser.add_argument("--weight_decay", type=str, default="0.0001")

    parser.add_argument("--out", type=str, default="hp_search.txt")

    args = parser.parse_args()

    hps = ["lr", "bb_lr", "weight_decay", "ft"]
    other_keys = ["data_dir", "backbone_type", "ckpt_path", "dataset"]
    all_keys = hps + other_keys

    all_combos = product(*[args.__dict__[k].split(",") for k in all_keys])
    all_combos = [dict(zip(all_keys, combo)) for combo in all_combos]

    for i in range(len(all_combos)):

        if all_combos[i]["ft"] == "lp":
            all_combos[i]["ft"] = ""
        elif all_combos[i]["ft"] == "ft":
            all_combos[i]["ft"] = "--finetune"

        if all_combos[i]["dataset"] == "oscd":
            all_combos[i]["task"] = "segmentation"
        elif all_combos[i]["dataset"] in ["eurosat", "sat"]:
            all_combos[i]["task"] = "classification"

    cmd = (
        "python train/main_{task}.py --backbone_type {backbone_type} --ckpt_path {ckpt_path} --data_dir {data_dir} --dataset {dataset}"
        + "--lr {lr} --bb_lr {bb_lr} --weight_decay {weight_decay} {ft}"
    )

    all_cmds = [cmd.format_map(hp) for hp in all_combos]

    # Save all commands into a file (one per line)
    open(args.out, "w").write("\n".join(all_cmds))
