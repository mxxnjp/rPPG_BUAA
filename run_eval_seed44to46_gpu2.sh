#!/bin/bash
set -e
cd ~/rPPG/BUAA_share
PY=~/miniconda3/envs/rppg/bin/python
COMMON="--window_size 150 --stride 75 --frame_depth 10 --batch_size 2 --max_train_windows 120 --max_val_windows 80 --val_max_lux 4.0 --device cuda:2 --results_csv metrics_eval.csv"

ev() {
  model=$1; split=$2; seed=$3; trainf=$4; valf=$5
  ckpt="checkpoint_buaa_${model}_split${split}_roifull_seed${seed}.pt"
  echo "[GPU2-EVAL] START model=$model split=$split seed=$seed $(date)"
  $PY -u evaluate.py --model $model --checkpoint $ckpt $COMMON --seed $seed --split $split --train_subjects $trainf --val_subjects $valf > logs/eval_${model}_split${split}_seed${seed}.log 2>&1
  echo "[GPU2-EVAL] DONE  model=$model split=$split seed=$seed $(date)"
}

ev efficientphys A 44 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
ev efficientphys C 44 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
ev efficientphys_gru B 44 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
ev efficientphys_bigru A 44 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
ev efficientphys_bigru C 44 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
ev efficientphys B 45 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
ev efficientphys_gru A 45 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
ev efficientphys_gru C 45 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
ev efficientphys_bigru B 45 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
ev efficientphys A 46 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
ev efficientphys C 46 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
ev efficientphys_gru B 46 buaa_experiment/train_subjects_B.txt buaa_experiment/val_subjects_B.txt
ev efficientphys_bigru A 46 buaa_experiment/train_subjects.txt buaa_experiment/val_subjects.txt
ev efficientphys_bigru C 46 buaa_experiment/train_subjects_C.txt buaa_experiment/val_subjects_C.txt
echo GPU2_EVAL_SEED44TO46_QUEUE_DONE
