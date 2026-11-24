#!/usr/bin/env bash
set -euo pipefail

project_dir=/mnt/cn600_1tb/ZY/landslide
python_bin=/mnt/cn600_1tb/miniconda3/envs/yolo-obb/bin/python
base_model="$project_dir/yolov8/ckpt/yolov8n.pt"
data_dir="$project_dir/dataset/landslide_yolo_gpt_clean_2026-08-26"
data_yaml="$data_dir/data_domain_balanced.yaml"
real_test_yaml="$data_dir/data_real_test.yaml"
run_name=landslide_yolov8n_gpt_clean_2026-08-26

cd "$project_dir"
"$python_bin" script_yolo/build_domain_balanced_train_list.py --data-dir "$data_dir"

"$python_bin" script_yolo/train.py \
  --data "$data_yaml" \
  --model "$base_model" \
  --epochs 100 \
  --imgsz 640 \
  --batch 1 \
  --workers 4 \
  --patience 20 \
  --optimizer AdamW \
  --lr0 0.0005 \
  --lrf 0.1 \
  --warmup-epochs 1 \
  --mosaic 0.2 \
  --translate 0.05 \
  --scale 0.25 \
  --close-mosaic 10 \
  --save-period 5 \
  --device 0 \
  --name "$run_name" \
  --evaluate-test \
  --evaluate-real-test "$real_test_yaml"
