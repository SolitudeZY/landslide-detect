# 滑坡 YOLO 数据融合、训练与视频推理

本目录默认只使用人工审核后的数据：

- `road_landslide_reviewed`：审核后的公开数据集正样本。
- `road_landslide_补充_positive_only`：自行抽帧并标注的补充正样本。
- `captures_v2_reviewed_negative` 和 `negative_samples`：无框负样本。

审核前的 `raw/` 和 `youluoyuan-dataset/` 不会再次混入。

## 1. 生成融合数据集

### 1.1 正样本质量审计

融合后可先运行只读审计，生成 `positive_quality_audit.csv`，不会删除或修改图片：

```bash
conda run -n landslide python script_yolo/audit_positive_quality.py \
  --data-dir dataset/landslide_yolo_screened_2026-08-21
```

将需要复核的图片按风险类型导出到同一目录的不同子目录：

```bash
conda run -n landslide python script_yolo/export_quality_review.py \
  --audit dataset/landslide_yolo_screened_2026-08-21/positive_quality_audit.csv
```

默认输出到 `dataset/positive_quality_review_2026-08-26/`，包含
`portrait_domain_mismatch/`、`possibly_blurred/`、`extreme_exposure/`、
`tiny_target/`、`box_near_edge/`、`all_flagged/` 等目录，以及
`review_index.csv`。同一图片命中多个条件时会出现在多个分类目录中；原始数据不变。

CSV 中的 `reasons` 用于排序人工复核：`portrait_domain_mismatch`（竖屏域差异）、
`low_resolution`、`possibly_blurred`、`extreme_exposure`、`tiny_target`、
`box_near_edge`。这些是风险提示，不是自动删样规则；确认后应在来源清单或审核表中记录
`keep/review/exclude`，再重建数据集。默认阈值可按实际监控分辨率用命令行参数调整。

当前筛选数据集已经生成，训练时无需重新执行本节。若以后需要重建数据集，先在所用
环境中安装 `requirements.txt` 里的 `ImageHash`、Pillow 和 PyYAML，再执行：

```bash
conda run -n landslide python script_yolo/prepare_dataset.py --dry-run
conda run -n landslide python script_yolo/prepare_dataset.py \
  --calibration-size 500 \
  --calibration-positive-ratio 0.5
conda run -n yolo-obb python script_yolo/validate_dataset.py
```

默认输出到 `dataset/landslide_yolo/`，结构如下：

```text
landslide_yolo/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── annotations/instances_{train,val,test}.json
├── data.yaml
├── manifest.csv
├── duplicates.csv
├── calibration.txt
├── calibration_wsl.txt
└── summary.json
```

融合规则：

- 正样本 COCO 矩形框被裁剪到图像范围后转换成 YOLO 格式。
- 负样本生成同名空 `.txt` 标签，这是 Ultralytics 支持的标准做法。
- 全部来源使用 SHA-256 精确去重。
- 仅对审核公开数据使用 pHash 近重复去重，并检查水平翻转；默认汉明距离阈值为 4。
- 同一连续视频片段、连续采集会话和同一重复簇不会跨 train/val/test。
- 正样本目录中无框的图片会被隔离并记录，不会自动改成负样本。
- `calibration.txt` 使用本机路径，`calibration_wsl.txt` 使用 `/mnt/<drive>/...`
  路径且强制 LF 行尾，可直接用于 WSL 中的 RKNN-Toolkit2。
- 校准集从 train split 分层抽样；参数 `--calibration-size 500` 和
  `--calibration-positive-ratio 0.5` 会固定生成 250 张正样本和 250 张负样本。

需要重建时显式加入 `--overwrite`。如希望节省磁盘，可使用 `--copy-mode hardlink`。

## 2. 训练

RK3568 的算力和内存有限，先用 `yolov8n` 建立基线。服务器使用 RTX 4060 Ti，
现有 `yolo-obb` Conda 环境可识别 GPU；`landslide` 环境仅用于数据整理，未安装
PyTorch 和 Ultralytics，不能用于训练。

```bash
cd /mnt/cn600_1tb/ZY/landslide
conda run -n yolo-obb python script_yolo/train.py \
  --epochs 150 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --export-onnx
```

上述命令默认使用 `dataset/landslide_yolo_screened_2026-08-19/data.yaml` 和
`yolov8/ckpt/yolov8n.pt`。显存不足时把 `--batch` 改为 4；训练结果写入
`script_yolo/runs/train/landslide_yolov8n/`。部署前应以 test 集固定评估一次，并按召回率而不是只按 mAP 选择置信度阈值。
最终超参数确定后，可在训练命令中加一次 `--evaluate-test`；不要在反复调参时持续查看 test 指标。

如训练后单独导出：

```bash
conda run -n yolo-obb python script_yolo/export_onnx.py \
  --model script_yolo/runs/train/landslide_yolov8n/weights/best.pt \
  --imgsz 640 --opset 12
```

导出使用 batch=1、静态输入、无 NMS 的 ONNX，适合作为 RKNN Toolkit2 的输入。

## 3. 视频推理

```bash
conda run -n yolo-obb python script_yolo/infer_video.py \
  --model script_yolo/runs/train/landslide_yolov8n_640/weights/best.pt \
  --source /path/to/video.mp4 \
  --conf 0.35 --device 0
```

脚本输出标注视频和同名 JSONL。主推理链路只使用 YOLO 检测结果，不再用帧差或光流
判定滑坡是否成立。默认在完整 10 帧窗口内至少 6 帧检出时激活报警；报警后窗口命中数
连续20帧未检出时解除，用迟滞避免短暂漏检导致报警抖动：

- `--alert-window 10 --alert-min-hits 6`：报警激活门槛。
- `--alert-clear-misses 20`：报警解除门槛，约0.8秒@25FPS，应按实际推理FPS标定。
- `--conf 0.35`：单帧检测阈值，必须结合独立真实负样本回放标定。

`motion.py` 和 `test_motion.py` 只保留用于复现早期实验，不再被 `infer_video.py` 引用。
RTSP 地址和摄像头编号仍可作为 `--source`。10 帧窗口在 25 FPS 下约为 0.4 秒，在
10 FPS 下约为 1 秒；板端主动跳帧时应按实际推理帧率评估告警延迟。

## 3.1 动态困难负样本训练

`mine_hard_negatives.py` 用当前模型扫描 train 中的空标签负样本，选出高置信度误报，
通过在训练清单中重复路径增加训练曝光，不复制图片、不修改 val/test。推荐使用两轮流程：

```bash
bash script_yolo/run_train_seedance_hard_mining_20260827.sh
```

流程会先用起始模型挖掘一次，微调15轮；再用第一轮最佳权重重新挖掘，继续微调20轮。
默认入选阈值为0.20，每个难负样本在对应训练清单中总计出现4次。每轮都会保留 CSV
明细和 JSON 汇总，最终仍需用固定真实 val/test、完整正样本视频和负样本视频选择权重。

## 4. RK3568 部署边界

`infer_video.py` 用于服务器上的 `.pt`/ONNX 验证，不应原样部署到 RK3568。边缘端需要：

1. 在 x86 主机安装与板端驱动匹配的 RKNN Toolkit2，将静态 ONNX 转成 RKNN。
2. INT8 量化使用 `landslide_yolo/calibration.txt` 中的正负代表图；不要只用正样本。
3. 板端使用 RKNN Lite2，并实现与训练一致的 letterbox、DFL 解码和 NMS。
4. 首次转换后，用同一批图片逐项对比 PyTorch、ONNX、RKNN 的框和置信度，再测速。

RKNN Toolkit2、板端 RKNPU 驱动和 RKNN Lite2 必须使用 Rockchip 兼容版本。模型转换脚本应在确认板卡系统镜像与驱动版本后再固定，避免生成不可加载的 RKNN 文件。
