# 2026-08-21 无框样本待复核

这些图片来自 `正样本-滑坡/landslide-08-21` 的 CVAT COCO 导出。
导出包共有 155 张图片，其中 122 张至少有一个 `landslide` 框，以下 33 张没有框。

COCO 导出无法区分“在 CVAT 中排除”与“保留但未标框”，因此这些图片目前不会进入
训练数据。请在 `review.csv` 的 `decision` 列填写以下值之一：

- `negative`：确认没有滑坡，可作为负样本。
- `exclude`：模糊、重复、无效或不希望参与训练。
- `positive_missing_box`：存在滑坡但漏标，需要回 CVAT 补框后重新导出。

`images/` 中均为原图副本，原始 CVAT 导出目录没有移动或删除。
