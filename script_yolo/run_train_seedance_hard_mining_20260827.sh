#!/usr/bin/env bash
set -euo pipefail

project_dir=/mnt/cn600_1tb/ZY/landslide
python_bin=/mnt/cn600_1tb/miniconda3/envs/yolo-obb/bin/python
data_dir="$project_dir/dataset/landslide_yolo_gpt_clean_2026-08-26"
base_model="$project_dir/script_yolo/runs/train/landslide_yolov8n_gpt_clean_finetune_2026-08-27/weights/selected_candidate.pt"
real_test_yaml="$data_dir/data_real_test.yaml"
round1_name=landslide_yolov8n_seedance_hard_round1_2026-08-27
round2_name=landslide_yolov8n_seedance_hard_round2_2026-08-27

cd "$project_dir"
"$python_bin" script_yolo/integrate_seedance_annotations.py
"$python_bin" script_yolo/build_domain_balanced_train_list.py --data-dir "$data_dir"
"$python_bin" script_yolo/validate_dataset.py --data "$data_dir/data.yaml"

"$python_bin" script_yolo/mine_hard_negatives.py \
  --data-dir "$data_dir" \
  --model "$base_model" \
  --conf 0.20 \
  --hard-repeat 4 \
  --max-hard 200 \
  --device cpu \
  --batch 4 \
  --scan-chunk 16 \
  --output-list "$data_dir/hard_negative_train_round1.txt" \
  --output-yaml "$data_dir/data_hard_negative_round1.yaml" \
  --report-csv "$data_dir/hard_negative_mining_round1.csv" \
  --summary "$data_dir/hard_negative_mining_round1_summary.json"

"$python_bin" script_yolo/train.py \
  --data "$data_dir/data_hard_negative_round1.yaml" \
  --model "$base_model" \
  --epochs 15 \
  --imgsz 640 \
  --batch 2 \
  --workers 4 \
  --patience 8 \
  --optimizer AdamW \
  --lr0 0.00005 \
  --lrf 0.1 \
  --warmup-epochs 0 \
  --mosaic 0.05 \
  --translate 0.02 \
  --scale 0.10 \
  --close-mosaic 5 \
  --save-period 5 \
  --device 0 \
  --name "$round1_name"

round1_model="$project_dir/script_yolo/runs/train/$round1_name/weights/best.pt"
"$python_bin" script_yolo/mine_hard_negatives.py \
  --data-dir "$data_dir" \
  --model "$round1_model" \
  --conf 0.20 \
  --hard-repeat 4 \
  --max-hard 200 \
  --device cpu \
  --batch 4 \
  --scan-chunk 16 \
  --output-list "$data_dir/hard_negative_train_round2.txt" \
  --output-yaml "$data_dir/data_hard_negative_round2.yaml" \
  --report-csv "$data_dir/hard_negative_mining_round2.csv" \
  --summary "$data_dir/hard_negative_mining_round2_summary.json"

"$python_bin" script_yolo/train.py \
  --data "$data_dir/data_hard_negative_round2.yaml" \
  --model "$round1_model" \
  --epochs 20 \
  --imgsz 640 \
  --batch 2 \
  --workers 4 \
  --patience 10 \
  --optimizer AdamW \
  --lr0 0.00003 \
  --lrf 0.1 \
  --warmup-epochs 0 \
  --mosaic 0.05 \
  --translate 0.02 \
  --scale 0.10 \
  --close-mosaic 5 \
  --save-period 5 \
  --device 0 \
  --name "$round2_name" \
  --evaluate-test \
  --evaluate-real-test "$real_test_yaml"
