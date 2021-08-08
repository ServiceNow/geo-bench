from argparse import ArgumentParser
from itertools import product


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
