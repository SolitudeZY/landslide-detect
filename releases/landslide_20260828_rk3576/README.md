# 滑坡数据集与 RK3576 模型发布包

发布日期：2026-08-28

## 直接使用

- 最新数据集压缩包：`landslide_yolo_latest_20260828.tar.zst`
- RK3576 INT8 模型：
  `models/landslide_yolov8n_seedance_hard_r2_20260827_int8_rk3576.rknn`
- 原始选中权重：`models/landslide_yolov8n_seedance_hard_r2_20260827.pt`
- 静态 ONNX：`models/landslide_yolov8n_seedance_hard_r2_20260827.onnx`
- RKNN 转换输入：`models/landslide_yolov8n_seedance_hard_r2_20260827_split.onnx`

校验所有文件：

```bash
sha256sum -c SHA256SUMS
```

解压数据集：

```bash
tar --use-compress-program=zstd -xf landslide_yolo_latest_20260828.tar.zst
```

解压后的 `PACKAGE_README.md` 说明了迁移后训练与生成本机 RKNN 校准清单的方法。

## 数据版本

数据根目录为 `landslide_yolo_gpt_clean_2026-08-26`，最后一次训练数据更新发生在
2026-08-27，已加入 20 张 Seedance 正样本和 7 张正常困难负样本，全部只进入 train。

| 划分 | 图片 | 正样本 | 负样本 | 框 |
|---|---:|---:|---:|---:|
| train | 2620 | 1465 | 1155 | 1488 |
| val | 314 | 185 | 129 | 188 |
| test | 285 | 156 | 129 | 159 |

`validate_dataset.py` 结果为 `errors: 0`。压缩包保留 YOLO 标签、COCO 标注、
`manifest.csv`、质量审计和两轮动态难负样本记录，排除了可再生的 `*.cache`。

## 模型版本

模型来自第二轮动态难负样本训练的 `last.pt`，已按场景评估选中并保持原哈希：

```text
PT SHA-256: 1e145841080fb5e3d89a741101e9cf7b92c6092119b7da6451d40a6a18c7c07b
```

部署起始阈值为 `conf=0.35`。详细指标和选择理由见 `MODEL_SELECTION_REPORT.md`。

## RKNN 转换与验证

- RKNN-Toolkit2：2.3.2
- 目标平台：RK3576
- 输入：RGB、NHWC、uint8、`1x640x640x3`
- 精度：INT8
- 输出：boxes `[1,4,8400]`，scores `[1,1,8400]`
- 校准集：500 张 train 图片，正负各 250 张

本机 simulator 已在 test 正样本上完成真实推理：ONNX 最大分数为 `0.862656`，
RKNN INT8 最大分数为 `0.815538`，没有发生分数塌缩。PT/ONNX 均输出 1 个检测，
最大置信度分别为 `0.8626560569` 和 `0.8626559973`。

当前验证覆盖本机 Toolkit2 simulator；仍需在实际 RK3576 板端用匹配的
RKNNLite2 和 `librknnrt.so` 2.3.2 做加载、图片回归与速度测试。
