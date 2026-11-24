# 滑坡数据集发布说明

本包对应 2026-08-27 完成的最新训练数据状态，打包日期为 2026-08-28。

## 数据统计

| 划分 | 图片 | 正样本 | 负样本 | 框 |
|---|---:|---:|---:|---:|
| train | 2620 | 1465 | 1155 | 1488 |
| val | 314 | 185 | 129 | 188 |
| test | 285 | 156 | 129 | 159 |

类别只有一个：`landslide`。`manifest.csv` 记录来源、分组、划分和 SHA-256，
`annotations/` 保留 COCO 标注，`summary.json` 和 `gpt_quality_audit.json` 保留审计结果。

## 解包后训练

`data.yaml` 保留原工作站绝对路径，原机可直接使用。迁移到其他目录后，从数据集目录运行：

```bash
yolo detect train data=data_portable.yaml model=yolov8n.pt imgsz=640
```

`data_portable.yaml` 不写固定根目录，由 Ultralytics 以 YAML 所在目录解析三个划分。

## RKNN INT8 校准

`calibration_relative.txt` 是固定抽取的 500 张 train 图片，正负各 250 张。解包后先在
数据集根目录生成本机绝对路径清单：

```bash
awk -v root="$(pwd)" '{print root "/" $0}' calibration_relative.txt > calibration_local.txt
```

再把 `calibration_local.txt` 传给 RKNN Toolkit2 的 `dataset` 参数。不要用 val/test 做量化校准。

## 使用边界

`hard_negative_mining_round*.csv/json` 和 `hard_negative_train_round*.txt` 是两轮动态难负样本
重采样记录；它们不是新增的独立验证数据。Seedance 视频有 27 帧进入 train，因此同一视频
回放只能证明定向修复生效，不能作为独立泛化证据。
