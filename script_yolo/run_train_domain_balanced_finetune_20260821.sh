#!/usr/bin/env bash
set -euo pipefail

project_dir=/mnt/cn600_1tb/ZY/landslide
python_bin=/mnt/cn600_1tb/miniconda3/envs/yolo-obb/bin/python
old_model="$project_dir/script_yolo/runs/train/landslide_yolov8n_screened_2026-08-19/weights/best.pt"
data_yaml="$project_dir/dataset/landslide_yolo_screened_2026-08-21/data_domain_balanced.yaml"
run_name=landslide_yolov8n_domain_balanced_finetune_2026-08-21
run_dir="$project_dir/script_yolo/runs/train/$run_name"
eval_dir="$project_dir/script_yolo/runs/eval"

cd "$project_dir"
"$python_bin" script_yolo/build_domain_balanced_train_list.py

"$python_bin" script_yolo/train.py \
  --data "$data_yaml" \
  --model "$old_model" \
  --epochs 80 \
  --imgsz 640 \
  --batch 4 \
  --workers 4 \
  --patience 20 \
  --optimizer AdamW \
  --lr0 0.0001 \
  --lrf 0.1 \
  --warmup-epochs 1 \
  --mosaic 0.1 \
  --translate 0.03 \
  --scale 0.15 \
  --close-mosaic 10 \
  --save-period 5 \
  --device 0 \
  --name "$run_name" \
  --evaluate-test

for checkpoint in best last; do
  model="$run_dir/weights/$checkpoint.pt"
  "$python_bin" script_yolo/evaluate_negative_subset.py --model "$model" --split all --source negative_20260821 --device cpu --output "$eval_dir/domain_finetune_${checkpoint}_all_232_new_negatives.json"
  "$python_bin" script_yolo/evaluate_positive_subset.py --model "$model" --split all --source new_positive_20260821 --device cpu --output "$eval_dir/domain_finetune_${checkpoint}_all_115_new_positives.json"
  "$python_bin" script_yolo/evaluate_negative_subset.py --model "$model" --split test --source negative_20260821 --device cpu --output "$eval_dir/domain_finetune_${checkpoint}_test_23_new_negatives.json"
  "$python_bin" script_yolo/evaluate_positive_subset.py --model "$model" --split test --source new_positive_20260821 --device cpu --output "$eval_dir/domain_finetune_${checkpoint}_test_8_new_positives.json"
  "$python_bin" script_yolo/evaluate_positive_subset.py --model "$model" --split test --source supplement --device cpu --output "$eval_dir/domain_finetune_${checkpoint}_test_27_supplement_positives.json"
  "$python_bin" script_yolo/compare_regression_reports.py \
    --old-negative "$eval_dir/baseline_old_model_all_232_new_negatives.json" \
    --new-negative "$eval_dir/domain_finetune_${checkpoint}_all_232_new_negatives.json" \
    --old-positive "$eval_dir/baseline_old_model_all_115_new_positives.json" \
    --new-positive "$eval_dir/domain_finetune_${checkpoint}_all_115_new_positives.json" \
    --threshold 0.35 \
    --output "$eval_dir/old_vs_domain_finetune_${checkpoint}_at_035.json"
done
