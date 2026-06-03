# segmentation_result.py
# 负责存储每个分割结果的元数据和紧凑掩膜

import numpy as np
from datetime import datetime

class SegmentationResult:
    """
    只保存裁剪后的小掩膜 + bbox，在需要时再还原成整幅图掩膜。
    """

    def __init__(self, mask, points, labels,
                 name=None, disease_type=None,
                 pattern_path=None, is_manual=False):
        """
        mask: 原图分辨率的布尔或 0/1 掩膜，会在内部自动裁剪到 bbox。
        """
        # 预览图上的掩膜（预览分辨率，用于显示）
        self.preview_mask = None

        # bbox: (y0, x0, y1, x1) 以原图坐标为基准
        self.bbox = None

        # 裁剪后的小掩膜（只覆盖 bbox 范围）
        self.mask = None

        if mask is not None:
            mask_bool = mask.astype(bool)
            ys, xs = np.where(mask_bool)
            if len(ys) > 0:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                self.bbox = (y0, x0, y1, x1)
                self.mask = mask_bool[y0:y1, x0:x1]
                self.area_pixels = int(np.sum(self.mask))
            else:
                self.bbox = None
                self.mask = None
                self.area_pixels = 0
        else:
            self.bbox = None
            self.mask = None
            self.area_pixels = 0

        self.points = points
        self.labels = labels
        self.name = name or f"分割_{datetime.now().strftime('%H%M%S')}"
        self.disease_type = disease_type
        self.pattern_path = pattern_path

        # pattern_image 在 GUI 里生成后挂上
        self.pattern_image = None

        self.created_time = datetime.now()
        self.is_manual = is_manual

    def to_dict(self):
        return {
            'name': self.name,
            'created_time': self.created_time.isoformat(),
            'points': self.points,
            'labels': self.labels,
            'disease_type': self.disease_type,
            'pattern_path': self.pattern_path,
            'is_manual': self.is_manual,
            'area_pixels': int(self.area_pixels),
            'bbox': self.bbox,
        }
