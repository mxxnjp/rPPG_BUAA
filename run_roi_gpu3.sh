#!/bin/bash
set -e
cd ~/rPPG/BUAA_share
PY=~/miniconda3/envs/rppg/bin/python
COMMON="--window_size 150 --stride 75 --frame_depth 10 --batch_size 2 --epochs 10 --max_train_windows 120 --max_val_windows 80 --val_max_lux 4.0 --seed 42 --device cuda:3 --results_csv results_roi.csv --roi multi"

run() {
  model=$1; split=$2; trainf=$3; valf=$4
  echo "[GPU3] START model=$model split=$split roi=multi $(date)"
  $PY -u train.py --model $model $COMMON --split $split --train_subjects $trainf --val_subjects $valf > logs/${model}_split${split}_roimulti.log 2>&1
  echo "[GPU3] DONE  model=$model split=$split roi=multi $(date)"
}

run efficientphys B buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
run efficientphys_bigru A buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
run efficientphys_bigru C buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
echo GPU3_ROI_QUEUE_DONE
