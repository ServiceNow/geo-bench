#!/bin/bash

PREFIX='eai job new --gpu 1 --mem 32 --image registry.console.elementai.com/snow.rg_climate_benchmark/base:SeCo-Baseline --data 1bc862ee-73d8-401a-bee6-942ba0580cae:/mnt -- bash -c' 
CD='cd /mnt/climate-change-benchmark'
ARGS='--max_epochs 1'

${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset sat --data_dir datasets/sat-6-full.mat --backbone_type random --lr 0.001 --exp 0 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset sat --data_dir datasets/sat-6-full.mat --backbone_type random --lr 0.0001 --exp 1 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset sat --data_dir datasets/sat-6-full.mat --backbone_type imagenet --lr 0.001 --exp 2 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset sat --data_dir datasets/sat-6-full.mat --backbone_type imagenet --lr 0.0001 --exp 3 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset sat --data_dir datasets/sat-6-full.mat --backbone_type pretrain --ckpt_path checkpoints/seco_resnet18_100k.ckpt --lr 0.001 --exp 4 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset sat --data_dir datasets/sat-6-full.mat --backbone_type pretrain --ckpt_path checkpoints/seco_resnet18_100k.ckpt --lr 0.0001 --exp 5 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset eurosat --data_dir datasets/eurosat  --backbone_type random --lr 0.001 --exp 6 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset eurosat --data_dir datasets/eurosat  --backbone_type random --lr 0.0001 --exp 7 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset eurosat --data_dir datasets/eurosat  --backbone_type imagenet --lr 0.001 --exp 8 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset eurosat --data_dir datasets/eurosat  --backbone_type imagenet --lr 0.0001 --exp 9 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset eurosat --data_dir datasets/eurosat  --backbone_type pretrain --ckpt_path checkpoints/seco_resnet18_100k.ckpt --lr 0.001 --exp 10 --max_epochs 1' 2>&1
${PREFIX} 'cd /mnt/climate-change-benchmark;python train/main_classification.py --dataset eurosat --data_dir datasets/eurosat  --backbone_type pretrain --ckpt_path checkpoints/seco_resnet18_100k.ckpt --lr 0.0001 --exp 11 --max_epochs 1' 2>&1
