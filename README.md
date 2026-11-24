# 滑坡检测（Landslide Detection）

固定摄像头场景下的滑坡检测与端侧部署工程：**YOLOv8n 单类别检测器** + **跨帧时序确认**，配套 RKNN INT8 量化、RK3576 板端推理服务与实时视频实验台；同时包含跨帧追踪 / 多证据事件状态机的完整技术方案。

> 仓库只收录**代码、配置与文档**。数据集、模型权重（`*.pt` / `*.onnx` / `*.rknn`）、视频、压缩包等大文件不入库，详见 [`.gitignore`](.gitignore) 与「数据准备」一节。

---

## 1. 项目简介

- **业务目标**：对固定机位（1080P）的边坡 / 公路场景做滑坡检测，输出目标框与事件级告警（关注 → 注意 → 预警）。
- **检测**：单类别 `landslide`，YOLOv8n @ 640，训练、评估、ONNX 导出全流程在 `script_yolo/`。
- **时序**：当前代码实现的是**检测器 + N 帧投票迟滞**（`infer_video.py`）；更完整的「ByteTrack 短时关联 + 场景目标注册表 + 多证据事件状态机」在方案文档中定义，尚未全部落地。
- **端侧**：Rockchip RK3576（4GB RAM / 32GB eMMC）上以 RKNN INT8 运行，配套板端 RTSP 服务、性能基准与本地实时演示台。

技术方案全文见 [滑坡跨帧检测与目标追踪方案.md](滑坡跨帧检测与目标追踪方案.md)（v1.1，评估稿，含性能/内存预算、分阶段计划与待决策点）。

---

## 2. 核心思路

### 2.1 检测

- 单类别、单尺度：`landslide`；训练集包含「灾害后堆积」正样本与大量**无框负样本**（正常路面、夜间虫影、光照突变等），负样本用空标签文件表达（Ultralytics 标准做法）。
- 数据融合时做 SHA-256 精确去重、审核数据 pHash 近重复去重，并保证**同一视频片段 / 采集会话 / 重复簇不跨 train/val/test**，避免泄漏。
- 选型结论：瓶颈在数据与决策层，不在检测器，因此沿用 YOLOv8n 而不换更大模型。

### 2.2 跨帧时序确认（当前已实现）

`script_yolo/infer_video.py` 把逐帧检测结果交给一个带迟滞的投票器：

- 最近 `--alert-window`（默认 10）帧中至少 `--alert-min-hits`（默认 6）帧检出 → 激活报警；
- 报警后连续 `--alert-clear-misses`（默认 20）帧未检出 → 解除，避免短暂漏检造成抖动；
- 每帧结果（帧号、时间戳、检测框、票数、报警状态与状态迁移）写入同名 `.jsonl`，便于离线复核；
- `--conf` 默认 0.35，需用**独立真实负样本视频**重新标定后再上线。

> 说明：`motion.py`（相机补偿帧差 + 稀疏光流）是早期实验模块，**不再被 `infer_video.py` 引用**，仅用于复现历史实验。方案文档已指出其应改为输出残差**向量**并将硬判定下放到决策层。

### 2.3 跨帧追踪与多证据（方案设计）

方案文档的核心是「观测层与决策层分离」：

| 层 | 职责 | 组成 |
|---|---|---|
| 观测层 | 只产生带时间戳的证据，不下业务结论 | ① 检测 + ByteTrack 短时关联；② 相机补偿帧差 + 光流残差（独立于检测框）；③ 长基线配准 + 累积位移（慢速蠕变） |
| 决策层 | 只消费证据 | 摄像头状态机（WARMUP→HEALTHY⇄DEGRADED→BLIND）+ 事件状态机（IDLE→OBSERVING→WATCH→WARNING→ALARM⇄COOLDOWN） |

- **长期身份**不依赖 `track_id`：由「场景目标注册表」（归一化位置 + 尺寸 + 灰度/纹理描述子）承担跨遮挡、跨昼夜、跨重启的身份延续。
- 三路证据源相互独立，任一路都能推动状态机，解决「无框即无运动」的缺陷。
- 夜间用轻量规则门控（亮度直方图 + 连通域），场景降级时**暂停累计而非清零**。

---

## 3. 目录结构

```text
landslide/
├── 滑坡跨帧检测与目标追踪方案.md   # 技术方案 v1.1（架构、预算、阶段计划）
├── README.md
├── .gitignore
├── script_yolo/                    # 数据处理、训练、评估、视频推理
│   ├── prepare_dataset.py          # 多来源 COCO/负样本 → YOLO 数据集（去重、防泄漏、校准集）
│   ├── audit_positive_quality.py   # 正样本质量只读审计
│   ├── export_quality_review.py    # 按风险类型导出待复核图片
│   ├── build_gpt_augmented_dataset.py / build_domain_balanced_train_list.py
│   ├── integrate_seedance_annotations.py / integrate_seedance_review_frames.py
│   ├── clean_dataset_from_review.py / augment_night_negatives.py
│   ├── validate_dataset.py         # 数据集结构校验
│   ├── train.py                    # Ultralytics 训练入口（可导出 ONNX、固定 test 评估）
│   ├── export_onnx.py              # batch=1、静态输入、无 NMS 的 ONNX
│   ├── mine_hard_negatives.py      # 动态困难负样本挖掘（两轮重采样）
│   ├── evaluate_*.py               # 正/负子集、真实 test、checkpoint 对比
│   ├── compare_regression_reports.py
│   ├── infer_video.py              # 视频/RTSP 推理 + 时序迟滞 + JSONL
│   ├── motion.py                   # 相机补偿帧差 + 光流（早期实验）
│   ├── run_train_*.sh              # 各轮训练复现脚本
│   ├── requirements.txt
│   └── test_*.py                   # 单元测试
├── dataset/                        # 仅保留 data*.yaml 与说明文档（图片/标签/标注不入库）
├── rk3576_deploy/                  # RK3576 端侧
│   ├── infer_rknn.py               # RKNN Lite2 推理（图片/视频/RTSP，分段耗时）
│   ├── runtime_utils.py            # RKNN runtime 创建与核心绑定
│   ├── edge_rtsp_service.py        # 板端 RTSP 推理 worker + 监控/控制 API
│   ├── landslide-edge-stream.service
│   ├── split_yolo_output.py        # 单输出 [1,5,8400] → [1,4,8400] + [1,1,8400]
│   ├── convert_onnx_to_rknn.py     # ONNX → RKNN（INT8/FP16）
│   ├── benchmark_rknn.py           # 纯推理性能基准（多核心掩码）
│   ├── test_rk3576_deploy.py       # 无需板卡的单元测试
│   └── stream_lab/                 # 本机 RTSP 推流 + Web 控制台 + HLS 预览
├── releases/landslide_20260828_rk3576/
│   ├── README.md                   # 发布包说明（数据集版本、模型版本）
│   ├── MODEL_SELECTION_REPORT.md   # 模型选型与场景回放报告
│   ├── DATASET_SUMMARY.json        # 数据来源与划分统计
│   ├── calibration_relative.txt    # 相对路径校准清单
│   └── SHA256SUMS                  # 发布物校验值（二进制本身不入库）
├── test_real/                      # 真实场景测试截图
└── yolov8/                         # 预训练权重目录（*.pt 不入库）
```

被忽略但本地存在的目录：`dataset/`（2.7GB 图片与标签）、`script_yolo/runs/`（训练/推理输出）、`rk3576_deploy/infer_res/`、`rk3576_deploy/stream_lab/uploads/`、`releases/*/models/`、`runs/`、`rknn/`（本地 Windows venv）。

---

## 4. 环境安装

### 4.1 训练 / 推理（Linux + NVIDIA GPU）

```bash
pip install -r script_yolo/requirements.txt
```

主要依赖：`ultralytics`（固定版本）、`opencv-python-headless`、`Pillow`、`ImageHash`、`PyYAML`。

仓库内脚本示例使用 conda 环境（如 `conda run -n yolo-obb python ...`），按本机环境替换即可；数据整理与训练使用不同环境也可以，只要缺 `PyTorch`/`ultralytics` 的环境不要用于训练。

### 4.2 RKNN 转换（x86 主机）

- RKNN-Toolkit2 **2.3.2**，目标平台 `rk3576`
- 转换脚本在 `rk3576_deploy/`，需与板端 `librknnrt.so` / RKNN-Toolkit-Lite2 **版本一致**

### 4.3 板端（RK3576 / AArch64）

```bash
source ~/venvs/landslide-rknn/bin/activate
pip install -r rk3576_deploy/requirements-board.txt   # numpy<2, opencv-python-headless
python -m pip show rknn-toolkit-lite2                  # 期望 2.3.2
sha256sum /usr/lib/librknnrt.so                        # 期望官方 v2.3.2 哈希
```

---

## 5. 数据准备

数据本体不入库，需要按 `releases/.../README.md` 与 `DATASET_SUMMARY.json` 描述的结构自行准备。目录约定（以 `dataset/landslide_yolo_gpt_clean_2026-08-26` 为例）：

```text
dataset/<版本>/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── annotations/instances_{train,val,test}.json
├── data.yaml                  # 基础配置（入库）
├── data_domain_balanced.yaml  # 域平衡训练清单（入库）
├── data_real_test.yaml        # 真实 test 子集（入库）
├── manifest.csv / summary.json
└── calibration.txt            # RKNN INT8 校准清单（本机绝对路径，不入库）
```

重建数据集：

```bash
python script_yolo/prepare_dataset.py --dry-run
python script_yolo/prepare_dataset.py --calibration-size 500 --calibration-positive-ratio 0.5
python script_yolo/validate_dataset.py
```

关键规则：COCO 框裁剪到图像范围后转 YOLO；负样本生成空 `.txt`；SHA-256 精确去重 + 审核数据 pHash 近重复去重；连续片段/会话不跨 split；正样本目录中无框图片被隔离而非自动转为负样本；校准集从 train 分层抽样（默认正负各半）。

> **注意**：仓库里的 `data*.yaml` 保留了原始路径（`path: /mnt/cn600_1tb/ZY/landslide/...`）。迁移到新机器后需要改成自己的数据目录，或使用相对路径的 `data_portable.yaml`。

当前已发布数据版本（引用自 `releases/.../DATASET_SUMMARY.json`）：

| 划分 | 图片 | 正样本 | 负样本 | 框 |
|---|---:|---:|---:|---:|
| train | 2620 | 1465 | 1155 | 1488 |
| val | 314 | 185 | 129 | 188 |
| test | 285 | 156 | 129 | 159 |

---

## 6. 训练

```bash
# 从数据到权重（默认 dataset/landslide_yolo_screened_2026-08-21/data.yaml + yolov8/ckpt/yolov8n.pt）
python script_yolo/train.py --epochs 150 --imgsz 640 --batch 2 --device 0 --export-onnx
```

- 显存不足时把 `--batch` 降到 2/1，或减小 `--imgsz`。
- 结果写入 `script_yolo/runs/train/<name>/`（该目录被忽略）。
- 部署阈值应按**召回率**而不是只看 mAP 选择；最终超参确定后再加一次 `--evaluate-test`，不要反复用 test 调参。
- 困难负样本（动态重采样，非 loss 层 OHEM）：

```bash
bash script_yolo/run_train_seedance_hard_mining_20260827.sh
```

其余各轮实验的完整命令行见 `script_yolo/run_train_*.sh`。

导出 ONNX：

```bash
python script_yolo/export_onnx.py --model script_yolo/runs/train/<name>/weights/best.pt \
  --imgsz 640 --opset 12
```

---

## 7. 推理（服务器侧）

```bash
python script_yolo/infer_video.py \
  --model script_yolo/runs/train/<name>/weights/best.pt \
  --source /path/to/video.mp4 \
  --conf 0.35 --device 0 \
  --alert-window 10 --alert-min-hits 6 --alert-clear-misses 20
```

- 默认输出 `<stem>_annotated.mp4` 与同名 `.jsonl`（`--no-video` 可只出 JSONL）。
- `--source` 支持视频路径、摄像头编号、RTSP URL。
- 窗口按**帧数**计算：10 帧在 25FPS 下约 0.4s，板端跳帧时需按实际推理帧率换算延迟。

RKNN 侧（板端）：见 `rk3576_deploy/README.md`，图片 / 视频 / RTSP 均支持：

```bash
python rk3576_deploy/infer_rknn.py --model model.rknn --source test.mp4 --output result.mp4
```

---

## 8. RK3576 端侧部署

整体链路：`best.pt` → 静态 ONNX（batch=1、无 NMS）→ 拆输出头 → RKNN INT8 → 板端 RKNN Lite2。

```bash
# 1) 拆输出头：单输出 [1,5,8400] 不适合直接 INT8（坐标与分数共用 scale）
python rk3576_deploy/split_yolo_output.py --source best.onnx --output landslide_split.onnx

# 2) 转 RKNN（INT8 为默认部署精度，FP16 仅用于精度对照）
python rk3576_deploy/convert_onnx_to_rknn.py \
  --onnx landslide_split.onnx \
  --output landslide_int8_rk3576.rknn \
  --precision int8 \
  --dataset calibration.txt
```

板端：

- 模型统一放在 `/home/kickpi/landslide/rk3576_deploy/model/`；
- `infer_rknn.py` 支持单输出与 6/9 输出两种 head 布局，打印预处理 / 推理 / 后处理分段耗时；
- 后处理在 IoU 之外增加「小框 85% 被大框包含则抑制」，处理大框套小框的重复而不过度合并；
- `benchmark_rknn.py` 做纯 `runtime.inference()` 基准（多核心掩码对比，不含前后处理）；
- `edge_rtsp_service.py` 是板端常驻服务（`landslide-edge-stream.service`），提供 `/health`、`/api/metrics`、`/api/models` 等接口；
- `stream_lab/` 是 Windows 侧实验台：上传视频循环推 RTSP → 板端检测 → 推回 → 网页 HLS 播放并轮询板端指标。依赖 `stream_lab/bin/mediamtx.exe` 与本机 ffmpeg（**二进制不入库**，需自行放置）。

> 版本必须对齐：RKNN-Toolkit2 / RKNN-Toolkit-Lite2 / `librknnrt.so` 均为 2.3.2，否则生成的模型可能无法加载。

---

## 9. 已知限制

1. **仓库不含权重与数据**：训练好的 `*.pt` / `*.onnx` / `*.rknn`、数据集压缩包、演示视频均被 `.gitignore` 排除，需按第 5、8 节自行准备或在发布包中获取。
2. **时序能力仍偏弱**：当前只实现「N 帧投票 + 迟滞」，方案中的 ByteTrack 关联、场景目标注册表、长基线蠕变、事件状态机尚未全部实现；`motion.py` 为早期实验代码。
3. **静态图指标 ≠ 在线误报率**：选型报告中的指标来自固定 test 集与场景回放，真实连续视频的报警率仍需独立负样本视频验证（详见 `releases/.../MODEL_SELECTION_REPORT.md` 的「边界」一节）。
4. **硬解未验证**：板端用 `cv2.VideoCapture` + FFmpeg 取流，是否走 MPP 硬解取决于板载 FFmpeg 是否编译 rkmpp，需实测；1080P 软解可能占满多核。
5. **4GB 内存约束**：全链路峰值目标控制在 ~3GB 内，禁止大缓存；长时间运行需做 72h 稳定性与内存增长验证。
6. **INT8 精度**：量化后置信度可能整体偏移，需用同一批图片对比 PT / ONNX / RKNN 的框与分数后再确认阈值。
7. **路径硬编码**：`data*.yaml`、训练脚本中的绝对路径（`/mnt/cn600_1tb/ZY/landslide`）需按新环境修改。
8. **网络素材合规**：用于补充训练的公开视频素材仅作内部研究用途，不商用，保留溯源。

---

## 10. 参考文档

- [滑坡跨帧检测与目标追踪方案.md](滑坡跨帧检测与目标追踪方案.md) —— 架构、证据清单、性能与内存预算、分阶段计划
- [releases/landslide_20260828_rk3576/README.md](releases/landslide_20260828_rk3576/README.md) —— 数据集与模型发布说明
- [releases/landslide_20260828_rk3576/MODEL_SELECTION_REPORT.md](releases/landslide_20260828_rk3576/MODEL_SELECTION_REPORT.md) —— 模型选型、指标与场景回放
- [script_yolo/README.md](script_yolo/README.md) —— 数据融合、训练与视频推理细节
- [rk3576_deploy/README.md](rk3576_deploy/README.md) —— RKNN 转换、基准与板端部署
- [rk3576_deploy/stream_lab/README.md](rk3576_deploy/stream_lab/README.md) —— 实时视频实验台
