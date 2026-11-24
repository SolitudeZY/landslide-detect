#!/usr/bin/env bash
set -euo pipefail

project_dir=/mnt/cn600_1tb/ZY/landslide
python_bin=/mnt/cn600_1tb/miniconda3/envs/yolo-obb/bin/python
run_name=landslide_yolov8n_screened_2026-08-21_finetune
run_dir="$project_dir/script_yolo/runs/train/$run_name"
old_model="$project_dir/script_yolo/runs/train/landslide_yolov8n_screened_2026-08-19/weights/best.pt"
new_model="$run_dir/weights/best.pt"
eval_dir="$project_dir/script_yolo/runs/eval"

cd "$project_dir"

"$python_bin" script_yolo/train.py \
  --model "$old_model" \
  --epochs 100 \
  --imgsz 640 \
  --batch 2 \
  --workers 4 \
  --patience 20 \
  --device 0 \
  --name "$run_name" \
  --evaluate-test

"$python_bin" script_yolo/evaluate_negative_subset.py \
  --model "$new_model" \
  --split all \
  --source negative_20260821 \
  --device cpu \
  --output "$eval_dir/new_model_all_232_new_negatives.json"

"$python_bin" script_yolo/evaluate_positive_subset.py \
  --model "$new_model" \
  --split all \
  --source new_positive_20260821 \
  --device cpu \
  --output "$eval_dir/new_model_all_115_new_positives.json"

"$python_bin" script_yolo/evaluate_negative_subset.py \
  --model "$new_model" \
  --split test \
  --source negative_20260821 \
  --device cpu \
  --output "$eval_dir/new_model_test_23_new_negatives.json"

"$python_bin" script_yolo/evaluate_positive_subset.py \
  --model "$new_model" \
  --split test \
  --source new_positive_20260821 \
  --device cpu \
  --output "$eval_dir/new_model_test_8_new_positives.json"

"$python_bin" script_yolo/compare_regression_reports.py \
  --old-negative "$eval_dir/baseline_old_model_all_232_new_negatives.json" \
  --new-negative "$eval_dir/new_model_all_232_new_negatives.json" \
  --old-positive "$eval_dir/baseline_old_model_all_115_new_positives.json" \
  --new-positive "$eval_dir/new_model_all_115_new_positives.json" \
  --threshold 0.35 \
  --output "$eval_dir/old_vs_new_regression_at_035.json"
