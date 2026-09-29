#!/bin/bash
set -e
cd ~/rPPG/BUAA_share
PY=~/miniconda3/envs/rppg/bin/python
COMMON_BASE="--window_size 150 --stride 75 --frame_depth 10 --batch_size 2 --epochs 10 --max_train_windows 120 --max_val_windows 80 --val_max_lux 4.0 --device cuda:3 --results_csv results.csv"

run() {
  model=$1; split=$2; seed=$3; trainf=$4; valf=$5
  echo "[GPU3] START model=$model split=$split seed=$seed $(date)"
  $PY -u train.py --model $model $COMMON_BASE --seed $seed --split $split --train_subjects $trainf --val_subjects $valf > logs/${model}_split${split}_seed${seed}.log 2>&1
  echo "[GPU3] DONE  model=$model split=$split seed=$seed $(date)"
}

run efficientphys B 44 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
run efficientphys_gru A 44 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
run efficientphys_gru C 44 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
run efficientphys_bigru B 44 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
run efficientphys A 45 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
run efficientphys C 45 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
run efficientphys_gru B 45 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
run efficientphys_bigru A 45 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
run efficientphys_bigru C 45 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
run efficientphys B 46 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
run efficientphys_gru A 46 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
run efficientphys_gru C 46 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
run efficientphys_bigru B 46 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
echo GPU3_SEED44TO46_QUEUE_DONE
