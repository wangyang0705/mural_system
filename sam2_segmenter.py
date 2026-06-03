# sam2_segmenter.py

import os
import sys
from typing import Callable, Optional

import numpy as np
import cv2
import torch
import torchvision.transforms as T

current_dir = os.path.dirname(os.path.abspath(__file__))

# 保证 sam2 可以被 import
sam2_path = current_dir
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("✓ SAM2 成功导入")
except ImportError as e:
    print(f"✗ 无法导入 SAM2: {e}")
    SAM2_AVAILABLE = False


# ====== 与训练代码对齐的预处理 ======

def build_preprocess_img(size: int = 1024):
    """
    与训练脚本保持一致：
    - HWC uint8 → PIL
    - Resize 到 (size, size)
    - ToTensor
    - Normalize (ImageNet)
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),   # /255, CHW
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


preprocess_image = build_preprocess_img(1024)


def preprocess_image_np_to_tensor(img_np: np.ndarray, device: torch.device):
    """
    和训练一样，只是把 .cuda() 替换成 to(device)，兼容 CPU/GPU。
    """
    img_tensor = preprocess_image(img_np)          # 3x1024x1024
    return img_tensor.unsqueeze(0).to(device)      # 1x3x1024x1024


class SAM2MuralSegmenter:
    """使用 SAM2 进行壁画病害分割 + 滑窗自动分割"""

    def __init__(self, model_type: str = "large",
                 finetuned_weights: Optional[str] = None):
        self.model = None
        self.predictor: Optional[SAM2ImagePredictor] = None
        self.model_type = model_type
        self.is_loaded = False

        self.model_configs = {
            "large": {
                "config_file": "sam2.1_hiera_l.yaml",
                "checkpoint": "sam2.1_hiera_l.pt",
                "model_name": "SAM2.1-Hiera-L"
            }
        }

        # 首先加载训练得到的微调权重，如果没有，再使用sam2预训练权重
        if finetuned_weights is None:
            self.finetuned_weights = os.path.join(
                current_dir, "mural_dataset", "trained_sam2_model_best.pth"
            )
        else:
            self.finetuned_weights = finetuned_weights

    # ---------- 路径查找 ----------

    def find_config_path(self, config_file):
        possible_paths = [
            os.path.join(sam2_path, "sam2", "configs", "sam2.1", config_file),
            os.path.join(sam2_path, "sam2", "configs", "sam2", config_file),
            os.path.join(sam2_path, "configs", "sam2.1", config_file),
            os.path.join(sam2_path, "configs", "sam2", config_file),
            os.path.join(sam2_path, "configs", config_file),
            os.path.join(sam2_path, config_file),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到配置文件: {path}")
                return path
        print(f"未找到配置文件: {config_file}")
        return None

    def find_checkpoint_path(self, checkpoint_file):
        possible_paths = [
            os.path.join(sam2_path, "checkpoints", checkpoint_file),
            os.path.join(sam2_path, "weights", checkpoint_file),
            os.path.join(sam2_path, checkpoint_file),
            os.path.join(current_dir, "checkpoints", checkpoint_file),
        ]
        for checkpoint_path in possible_paths:
            if os.path.exists(checkpoint_path):
                print(f"找到权重文件: {checkpoint_path}")
                return checkpoint_path
        print(f"未找到权重文件: {checkpoint_file}")
        return None

    # ---------- 模型加载 ----------

    def load_model(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> bool:
        """加载 SAM2 模型 + 优先微调权重"""
        if not SAM2_AVAILABLE:
            print("SAM2 不可用")
            return False
        if self.is_loaded:
            print("模型已加载")
            return True

        try:
            if progress_callback:
                progress_callback(10, "正在初始化 SAM2 模型...")

            model_config = self.model_configs.get(self.model_type,
                                                  self.model_configs["large"])
            print(f"正在加载模型: {model_config['model_name']}")

            config_path = self.find_config_path(model_config["config_file"])
            checkpoint_path = self.find_checkpoint_path(model_config["checkpoint"])
            if not config_path or not checkpoint_path:
                return False

            if progress_callback:
                progress_callback(30, "正在加载基础权重...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {device}")

            self.model = build_sam2(config_path, checkpoint_path, device=str(device))

            if progress_callback:
                progress_callback(50, "正在初始化预测器...")

            self.predictor = SAM2ImagePredictor(self.model)

            # 尝试加载你训练得到的微调权重
            if self.finetuned_weights and os.path.exists(self.finetuned_weights):
                print(f"尝试加载微调权重: {self.finetuned_weights}")
                state = torch.load(self.finetuned_weights, map_location=device)
                # 直接覆盖 predictor.model 的权重
                self.predictor.model.load_state_dict(state, strict=False)
                print("✅ 已加载微调权重")
            else:
                print("⚠ 未找到微调权重，将使用原始 SAM2 权重")

            self.is_loaded = True

            if progress_callback:
                progress_callback(100, "模型加载完成!")

            print(f"✅ SAM2 模型加载成功: {model_config['model_name']}")
            return True

        except Exception as e:
            print(f"❌ 加载 SAM2 模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ---------- 点提示分割----------

    def segment_with_points(self, image: np.ndarray,
                            points, labels,
                            progress_callback: Optional[Callable[[float, str], None]] = None):
        """
        使用正/负点提示分割（image 为原图 RGB HxWx3）。
        """
        orig_size = image.shape[:2]

        if not self.is_loaded or self.predictor is None:
            print("模型未加载")
            return None

        try:
            if progress_callback:
                progress_callback(10, "正在设置图像...")

            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            self.predictor.set_image(image)

            if progress_callback:
                progress_callback(40, "正在处理点提示...")

            input_points = np.array([[p[0], p[1]] for p in points])
            input_labels = np.array(labels)

            if progress_callback:
                progress_callback(70, "正在进行分割预测...")

            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
                mask_input=None
            )

            best_mask = masks[0]
            print(f"分割完成 - IoU 评分（模型内部估计）: {scores[0]:.4f}")

            best_mask = cv2.resize(
                best_mask.astype(np.uint8),
                (orig_size[1], orig_size[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # 只保留最大连通区域
            try:
                kernel = np.ones((3, 3), np.uint8)
                best_mask = cv2.morphologyEx(best_mask.astype(np.uint8),
                                             cv2.MORPH_CLOSE, kernel)
                num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
                    best_mask.astype(np.uint8)
                )
                if num_labels > 1:
                    max_area_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                    best_mask = (labels_cc == max_area_index).astype(np.uint8)
                    print(f"选择最大连通区域，面积: {stats[max_area_index, cv2.CC_STAT_AREA]} 像素")
            except Exception as e:
                print(f"后处理失败: {e}")

            return best_mask.astype(bool)

        except Exception as e:
            print(f"SAM2 分割失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---------- 滑窗分割（全图自动扫描，适配训练预处理） ----------

    # def _infer_tile_auto(self,
    #                     tile_rgb: np.ndarray,
    #                     device: torch.device,
    #                     tile_offset: tuple,  # 新增：当前tile在原图中的偏移量 (y0, x0)
    #                     global_points: list,  # 新增：全局点坐标
    #                     global_labels: list,  # 新增：全局点标签
    #                     point_grid: int = 2,
    #                     score_thresh: float = 0.5) -> np.ndarray:
    #     """
    #     对单个 tile 进行自动分割，结合全局点提示
    #     - tile_rgb: tile图像
    #     - tile_offset: (y0, x0) tile在原图中的起始坐标
    #     - global_points: 全局点坐标列表 [[x,y], ...]
    #     - global_labels: 全局点标签列表 [1,0,...]
    #     - point_grid: 均匀采样点的网格密度
    #     - score_thresh: 得分阈值
    #     """
    #     if self.predictor is None:
    #         return np.zeros(tile_rgb.shape[:2], dtype=bool)

    #     h_t, w_t = tile_rgb.shape[:2]
    #     y0_tile, x0_tile = tile_offset
        
    #     # 使用2048作为处理尺寸
    #     target_size = 2048
    #     internal_size = 1024

    #     # 预处理到2048×2048
    #     with torch.no_grad():
    #         if h_t != target_size or w_t != target_size:
    #             tile_resized = cv2.resize(tile_rgb, (target_size, target_size), 
    #                                     interpolation=cv2.INTER_LINEAR)
    #         else:
    #             tile_resized = tile_rgb
                
    #         img_tensor = preprocess_image_np_to_tensor(tile_resized, device)
    #         img_2048 = (img_tensor.squeeze(0)
    #                     .permute(1, 2, 0)
    #                     .detach().cpu().numpy() * 255).astype(np.uint8)

    #     # SAM2 以2048×2048作为"原始分辨率"
    #     self.predictor.set_image(img_2048)
    #     mask_acc_2048 = np.zeros((target_size, target_size), dtype=bool)

    #     # 收集当前tile内的点
    #     tile_points = []
    #     tile_labels = []
        
    #     # 将全局点转换到tile坐标系
    #     for point, label in zip(global_points, global_labels):
    #         x, y = point
    #         # 检查点是否在当前tile内
    #         if x0_tile <= x < x0_tile + w_t and y0_tile <= y < y0_tile + h_t:
    #             # 转换到tile坐标系
    #             tile_x = x - x0_tile
    #             tile_y = y - y0_tile
                
    #             # 缩放到2048坐标系
    #             scale_x = target_size / w_t
    #             scale_y = target_size / h_t
    #             scaled_x = tile_x * scale_x
    #             scaled_y = tile_y * scale_y
                
    #             tile_points.append([scaled_x, scaled_y])
    #             tile_labels.append(label)
        
    #     print(f"Tile [{y0_tile}:{y0_tile+h_t}, {x0_tile}:{x0_tile+w_t}] 包含 {len(tile_points)} 个标记点")

    #     with torch.no_grad():
    #         # 如果有用户标记点，优先使用这些点
    #         if len(tile_points) >= 1:
    #             # 使用用户标记的点
    #             input_points = np.array(tile_points, dtype=np.float32).reshape(-1, 2)
    #             input_labels = np.array(tile_labels, dtype=np.int64)
                
    #             # 转换为SAM2需要的格式 [1, N, 2]
    #             input_points = input_points[np.newaxis, :, :]
    #             input_labels = input_labels[np.newaxis, :]
                
    #             masks, scores, _ = self.predictor.predict(
    #                 point_coords=input_points,
    #                 point_labels=input_labels,
    #                 multimask_output=False
    #             )
                
    #             best_mask = masks[0]
    #             score = float(scores[0])
    #             print(f"使用用户标记点推理，得分: {score:.4f}")
                
    #             # 将结果上采样到2048
    #             bin_mask_2048 = cv2.resize(
    #                 best_mask.astype(np.uint8),
    #                 (target_size, target_size),
    #                 interpolation=cv2.INTER_NEAREST
    #             ).astype(bool)
                
    #             mask_acc_2048 |= bin_mask_2048
            
    #         # 如果用户标记点不够（少于3个），补充均匀采样点
    #         remaining_points = max(0, point_grid * point_grid - len(tile_points))
    #         if remaining_points > 0:
    #             print(f"补充 {remaining_points} 个均匀采样点")
                
    #             # 生成均匀采样点
    #             for gy in range(point_grid):
    #                 for gx in range(point_grid):
    #                     # 跳过已经有点的位置
    #                     # 简单的策略：如果tile_points少于网格点数量，我们补充到网格密度
    #                     if len(tile_points) < point_grid * point_grid:
    #                         y = (gy + 0.5) * target_size / point_grid
    #                         x = (gx + 0.5) * target_size / point_grid
                            
    #                         # 检查这个网格点是否已经被用户点覆盖
    #                         # 简单的距离检查
    #                         is_covered = False
    #                         for tp in tile_points:
    #                             dist = np.sqrt((tp[0] - x)**2 + (tp[1] - y)**2)
    #                             if dist < target_size / point_grid / 2:  # 如果距离小于网格间距的一半
    #                                 is_covered = True
    #                                 break
                            
    #                         if not is_covered:
    #                             input_points = np.array([[[x, y]]], dtype=np.float32)
    #                             input_labels = np.array([[1]], dtype=np.int64)

    #                             mask_input, unnorm_coords, labels, _ = self.predictor._prep_prompts(
    #                                 input_points, input_labels,
    #                                 box=None, mask_logits=None,
    #                                 normalize_coords=True
    #                             )

    #                             sparse_emb, dense_emb = self.predictor.model.sam_prompt_encoder(
    #                                 points=(unnorm_coords, labels), boxes=None, masks=None
    #                             )

    #                             high_res_feats = [
    #                                 f[-1].unsqueeze(0)
    #                                 for f in self.predictor._features["high_res_feats"]
    #                             ]

    #                             low_res_masks, prd_scores, _, _ = \
    #                                 self.predictor.model.sam_mask_decoder(
    #                                     image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
    #                                     image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
    #                                     sparse_prompt_embeddings=sparse_emb,
    #                                     dense_prompt_embeddings=dense_emb,
    #                                     multimask_output=True,
    #                                     repeat_image=False,
    #                                     high_res_features=high_res_feats,
    #                                 )

    #                             prd_masks = self.predictor._transforms.postprocess_masks(
    #                                 low_res_masks, (internal_size, internal_size)
    #                             )

    #                             logits = prd_masks[:, 0]
    #                             scores = prd_scores[:, 0]
    #                             score = float(scores.mean().item())

    #                             if score >= score_thresh:
    #                                 prob = torch.sigmoid(logits)
    #                                 bin_mask_1024 = (prob > 0.5)[0].detach().cpu().numpy().astype(bool)
                                    
    #                                 bin_mask_2048 = cv2.resize(
    #                                     bin_mask_1024.astype(np.uint8),
    #                                     (target_size, target_size),
    #                                     interpolation=cv2.INTER_NEAREST
    #                                 ).astype(bool)
                                    
    #                                 mask_acc_2048 |= bin_mask_2048

    #     # 将2048掩膜缩放回 tile 原始大小
    #     if h_t == target_size and w_t == target_size:
    #         mask_tile = mask_acc_2048
    #     else:
    #         mask_tile = cv2.resize(
    #             mask_acc_2048.astype(np.uint8),
    #             (w_t, h_t),
    #             interpolation=cv2.INTER_NEAREST
    #         ).astype(bool)

    #     return mask_tile


    def sliding_window_inference(self,
                                image: np.ndarray,
                                global_points: list = None,
                                global_labels: list = None,
                                window_size: int = 2048,
                                stride: int = 1024,
                                point_grid: int = 2,  # 这个参数在点提示模式下不再使用，保留是为了兼容
                                score_thresh: float = 0.5,  # 这个参数在点提示模式下不再使用
                                progress_callback: Optional[Callable[[float, str], None]] = None,
                                tile_callback: Optional[Callable[[np.ndarray], None]] = None
                                ) -> np.ndarray:
        """
        对整幅大图进行滑窗分割，在每个窗口中调用点提示分割
        - image: 原图 RGB HxWx3
        - global_points: 全局点坐标列表 [[x,y], ...]
        - global_labels: 全局点标签列表 [1,0,...]
        - window_size: 滑窗大小，默认2048
        - stride: 步长，默认1024（50%重叠）
        - 返回 full_mask: bool [H,W]
        """
        if not self.is_loaded or self.predictor is None:
            raise RuntimeError("SAM2 模型尚未加载")

        # 初始化点列表
        if global_points is None:
            global_points = []
        if global_labels is None:
            global_labels = []

        try:
            H, W = image.shape[:2]
            print(f"滑窗点提示分割开始: 图像尺寸 {H}x{W}, 窗口 {window_size}, 步长 {stride}")
            print(f"全局标记点数量: {len(global_points)}")
            
            device = next(self.predictor.model.parameters()).device
            full_mask = np.zeros((H, W), dtype=bool)

            # 所有 tile 坐标
            tiles = []
            
            # 计算x和y方向的步数
            y_steps = max(1, ((H - window_size) // stride) + 1) if H > window_size else 1
            x_steps = max(1, ((W - window_size) // stride) + 1) if W > window_size else 1
            
            if H <= window_size and W <= window_size:
                tiles.append((0, H, 0, W))
            else:
                for y_step in range(y_steps):
                    y0 = y_step * stride
                    y1 = min(y0 + window_size, H)
                    
                    if y_step == y_steps - 1 and y1 < H:
                        y0 = H - window_size
                        y1 = H
                    
                    for x_step in range(x_steps):
                        x0 = x_step * stride
                        x1 = min(x0 + window_size, W)
                        
                        if x_step == x_steps - 1 and x1 < W:
                            x0 = W - window_size
                            x1 = W
                        
                        tiles.append((y0, y1, x0, x1))

            total_tiles = len(tiles)
            print(f"共 {total_tiles} 个tiles")

            if total_tiles == 0:
                return full_mask

            for idx, (y0, y1, x0, x1) in enumerate(tiles):
                try:
                    print(f"\n处理 tile {idx+1}/{total_tiles}: y[{y0}:{y1}], x[{x0}:{x1}]")
                    
                    # 提取当前tile的图像
                    tile_rgb = image[y0:y1, x0:x1, :]
                    
                    # 收集当前tile内的点
                    tile_points = []
                    tile_labels = []
                    
                    # 将全局点转换到tile坐标系
                    for point, label in zip(global_points, global_labels):
                        x, y = point
                        # 检查点是否在当前tile内
                        if x0 <= x < x1 and y0 <= y < y1:
                            # 转换到tile坐标系
                            tile_x = x - x0
                            tile_y = y - y0
                            tile_points.append([tile_x, tile_y])
                            tile_labels.append(label)
                    
                    print(f"当前tile包含 {len(tile_points)} 个标记点")
                    
                    # 如果当前tile内没有标记点，跳过此tile
                    if len(tile_points) == 0:
                        print(f"Tile {idx+1} 内没有标记点，跳过")
                        
                        # 更新进度
                        if progress_callback is not None:
                            p = 100.0 * (idx + 1) / total_tiles
                            msg = f"滑窗分割 {idx + 1}/{total_tiles} (跳过，无标记点)"
                            progress_callback(p, msg)
                        
                        # 仍然调用回调以更新UI
                        if tile_callback is not None:
                            tile_callback(full_mask.copy())
                        
                        continue
                    
                    # 调用点提示分割方法
                    tile_mask = self.segment_with_points(
                        tile_rgb,
                        tile_points,
                        tile_labels,
                        None  # 不传递progress_callback，避免干扰
                    )
                    
                    if tile_mask is not None:
                        # 将tile掩膜放到原图对应位置
                        h_sub = y1 - y0
                        w_sub = x1 - x0
                        # 确保掩膜尺寸匹配
                        if tile_mask.shape[0] != h_sub or tile_mask.shape[1] != w_sub:
                            tile_mask = cv2.resize(
                                tile_mask.astype(np.uint8),
                                (w_sub, h_sub),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        
                        full_mask[y0:y1, x0:x1] |= tile_mask
                        
                        print(f"Tile {idx+1} 分割完成，掩膜非零值: {np.sum(tile_mask)}")
                    else:
                        print(f"Tile {idx+1} 分割失败")

                    # 进度更新
                    if progress_callback is not None:
                        p = 100.0 * (idx + 1) / total_tiles
                        msg = f"滑窗分割 {idx + 1}/{total_tiles}"
                        progress_callback(p, msg)

                    # 热更新 GUI overlay
                    if tile_callback is not None:
                        tile_callback(full_mask.copy())
                        
                except Exception as e:
                    print(f"处理 tile {idx+1} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"\n滑窗点提示分割完成，最终掩膜非零值: {np.sum(full_mask)}")
            return full_mask
            
        except Exception as e:
            print(f"滑窗分割整体失败: {e}")
            import traceback
            traceback.print_exc()
            raise
