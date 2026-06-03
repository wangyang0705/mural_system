# mural_app.py

# -*- coding: utf-8 -*-
import os
import json
import threading
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

import torch
from torchvision import models, transforms

from segmentation_result import SegmentationResult
from sam2_segmenter import SAM2MuralSegmenter
from drawing_utils import generate_pattern_image, apply_pattern_fill

current_dir = os.path.dirname(os.path.abspath(__file__))


class SAM2MuralApp:
    """集成 SAM2 的壁画病害分割"""

    def __init__(self, root):
        self.root = root
        self.root.title("智能分割系统")
        self.root.geometry("1400x900")

        print("初始化 SAM2 壁画病害分割系统...")

        # 图像相关
        self.original_image = None
        self.preview_image = None
        self.current_image = None
        self.display_image_pil = None
        self.base_display_image = None
        self.display_scale = 1.0
        self.max_preview_side = 4096

        # 视图控制
        self.scale_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False

        # 分割相关（点提示）
        self.current_points = []
        self.current_labels = []
        self.saved_results = []
        self.active_result_index = -1
        self.is_segmenting = False

        self.current_mask = None        # 预览掩膜
        self.current_mask_full = None   # 原图掩膜

        # 滑窗自动分割的全局掩膜
        self.global_mask_full = None        # 原图分辨率
        self.global_mask_preview = None     # 预览分辨率

        #分割
        self.is_manual_mode = False
        self.manual_mode_type = None  # "polygon" / "polyline"
        self.manual_points = []
        self.manual_polygon_items = []
        self.manual_line_items = []

        # 点可视化
        self.point_items = []

        # SAM2 分割器
        self.sam2_segmenter = SAM2MuralSegmenter("large")

        # 病害分类模型
        self.cls_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cls_model = None
        self.cls_class_to_idx = None
        self.idx_to_class = None
        self.cls_transform = None
        self._load_disease_classifier()

        # 显示选项
        self.show_boundary = tk.BooleanVar(value=True)
        self.show_fill = tk.BooleanVar(value=True)
        self.show_saved_results = tk.BooleanVar(value=True)
        self.show_current_result = tk.BooleanVar(value=True)
        self.show_global_result = tk.BooleanVar(value=True)

        # 添加自动预测开关
        self.auto_use_prediction = tk.BooleanVar(value=True)  # 默认开启

        self.fill_color = (255, 0, 0)
        self.boundary_color = (0, 255, 0)
        self.global_fill_color = (0, 0, 255)   # 全图自动分割的覆盖颜色
        self.fill_alpha = 0.3

        # 病害颜色表（传给 drawing_utils）
        self.disease_colors = {
            "颜料层脱落": (0, 0, 255),
            "地仗层脱落": (255, 255, 255),
            "水渍": (128, 0, 128),
            "裂隙": (0, 255, 0),
            "泥渍": (139, 69, 19),
        }

        # 统计缓存
        self.stats_cache = None
        self.stats_dirty = True

        self.create_widgets()
        self.load_sam2_model()
        self.bind_keyboard_events()

        print("系统初始化完成")

    # ====================== UI ======================

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧画布框架 - 设置 expand=True 让其占据所有剩余空间
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg="darkgray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_left_click)
        self.canvas.bind("<Control-Button-1>", self.on_negative_click)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self.on_pan_end)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)

        # 右侧可滚动面板
        right_container = ttk.Frame(main_frame, width=280)
        right_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_container.pack_propagate(False)

        # 创建水平布局框架来容纳画布和滚动条
        right_content = ttk.Frame(right_container)
        right_content.pack(fill=tk.BOTH, expand=True)

        # 创建画布
        canvas = tk.Canvas(right_content, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_content, orient="vertical", command=canvas.yview)

        # 可滚动的内部框架
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 打包画布和滚动条
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 以下所有右侧控件都放在 scrollable_frame 中
        right_frame = scrollable_frame

        title_label = ttk.Label(
            right_frame, text="图像智能分割",
            font=("Arial", 14, "bold"), foreground="darkblue"
        )
        title_label.pack(pady=10)

        # 模型状态
        status_frame = ttk.LabelFrame(right_frame, text="模型状态", padding=5)
        status_frame.pack(fill=tk.X, pady=5)

        self.model_status_var = tk.StringVar(value="正在加载 SAM2 模型...")
        ttk.Label(status_frame, textvariable=self.model_status_var).pack(anchor=tk.W)

        # 文件操作
        file_frame = ttk.LabelFrame(right_frame, text="文件操作", padding=5)
        file_frame.pack(fill=tk.X, pady=5)

        ttk.Button(file_frame, text="打开图像",
                command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="保存所有结果",
                command=self.save_all_results).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="导出统计报告",
                command=self.export_statistics_report).pack(fill=tk.X, pady=2)

        # 分割控制
        seg_frame = ttk.LabelFrame(right_frame, text="分割控制", padding=5)
        seg_frame.pack(fill=tk.X, pady=5)

        ttk.Button(seg_frame, text="智能分割",
                command=self.sam2_segment).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="全图滑窗自动分割",
                command=self.sam2_sliding_window_segment).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="多边形分割",
                command=self.toggle_manual_polygon_mode).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="裂隙线段",
                command=self.toggle_manual_polyline_mode).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="清除当前分割结果",
                command=self.clear_current_segmentation).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="保存当前分割",
                command=self.save_current_segmentation).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="清除当前标记",
                command=self.clear_current_marks).pack(fill=tk.X, pady=2)

        # 已保存结果
        results_frame = ttk.LabelFrame(right_frame, text="已保存结果", padding=5)
        results_frame.pack(fill=tk.X, pady=5)

        self.results_listbox = tk.Listbox(results_frame, height=4)
        self.results_listbox.pack(fill=tk.BOTH, expand=True, pady=2)
        self.results_listbox.bind('<<ListboxSelect>>', self.on_result_selected)

        results_btn_frame = ttk.Frame(results_frame)
        results_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(results_btn_frame, text="删除选中",
                command=self.delete_selected_result).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        ttk.Button(results_btn_frame, text="清除所有",
                command=self.clear_all_results).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # 显示控制
        display_frame = ttk.LabelFrame(right_frame, text="显示控制", padding=5)
        display_frame.pack(fill=tk.X, pady=5)

        ttk.Checkbutton(display_frame, text="显示病害边界",
                        variable=self.show_boundary,
                        command=self.display_image).pack(anchor=tk.W, pady=1)

        ttk.Checkbutton(display_frame, text="显示病害填充",
                        variable=self.show_fill,
                        command=self.display_image).pack(anchor=tk.W, pady=1)

        ttk.Checkbutton(display_frame, text="显示已保存结果",
                        variable=self.show_saved_results,
                        command=self.display_image).pack(anchor=tk.W, pady=1)

        ttk.Checkbutton(display_frame, text="显示当前分割结果",
                        variable=self.show_current_result,
                        command=self.display_image).pack(anchor=tk.W, pady=1)

        ttk.Checkbutton(display_frame, text="显示全图自动结果",
                        variable=self.show_global_result,
                        command=self.display_image).pack(anchor=tk.W, pady=1)

        # 添加自动预测开关复选框
        auto_frame = ttk.Frame(display_frame)
        auto_frame.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(auto_frame, text="自动使用病害类型预测",
                        variable=self.auto_use_prediction).pack(anchor=tk.W)

        alpha_frame = ttk.Frame(display_frame)
        alpha_frame.pack(fill=tk.X, pady=2)
        ttk.Label(alpha_frame, text="透明度:").pack(side=tk.LEFT)
        self.alpha_scale = ttk.Scale(
            alpha_frame, from_=0.1, to=0.8,
            orient=tk.HORIZONTAL, value=self.fill_alpha,
            command=self.on_alpha_change
        )
        self.alpha_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 进度
        progress_frame = ttk.LabelFrame(right_frame, text="进度", padding=5)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X)

        self.progress_label = ttk.Label(progress_frame, text="就绪")
        self.progress_label.pack(anchor=tk.W)

        # 统计信息
        info_frame = ttk.LabelFrame(right_frame, text="统计信息", padding=5)
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = tk.Text(info_frame, height=6, width=30, font=("Arial", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # ===== 滚轮事件绑定 =====
        def on_canvas_mousewheel(event):
            """画布滚轮事件 - 滚动整个右侧面板"""
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"  # 阻止事件继续传播

        def on_listbox_mousewheel(event):
            """Listbox滚轮事件 - 只滚动Listbox"""
            self.results_listbox.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"  # 阻止事件继续传播

        def on_text_mousewheel(event):
            """Text滚轮事件 - 只滚动Text"""
            self.info_text.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"  # 阻止事件继续传播

        # 绑定画布的滚轮事件
        canvas.bind("<MouseWheel>", on_canvas_mousewheel)
        
        # 当鼠标进入画布时，绑定全局滚轮事件（用于滚动整个面板）
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_canvas_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # 为Listbox绑定独立的滚轮事件
        self.results_listbox.bind("<MouseWheel>", on_listbox_mousewheel)
        self.results_listbox.bind("<Enter>", lambda e: self.results_listbox.focus_set())

        # 为Text组件绑定独立的滚轮事件
        self.info_text.bind("<MouseWheel>", on_text_mousewheel)
        self.info_text.bind("<Enter>", lambda e: self.info_text.focus_set())

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请打开图像")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                            relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ====================== 模型 & 进度 ======================

    def bind_keyboard_events(self):
        self.root.bind('<Escape>', lambda e: self.cancel_manual_mode())
        self.root.bind('<Return>', lambda e: self.finish_manual_shape())
        self.root.bind('<Delete>', lambda e: self.cancel_manual_drawing())

    def load_sam2_model(self):
        def load_thread():
            success = self.sam2_segmenter.load_model(self.update_progress)
            if success:
                self.root.after(0, lambda: self.model_status_var.set("模型加载成功!"))
            else:
                self.root.after(0, lambda: self.model_status_var.set("模型加载失败"))

        threading.Thread(target=load_thread, daemon=True).start()

    def _load_disease_classifier(self):
        """加载病害分类模型（如果存在的话）"""
        try:
            ckpt_path = os.path.join(current_dir, "checkpoints", "disease_classifier.pth")
            print(f"尝试加载病害分类模型: {ckpt_path}")
            
            if not os.path.exists(ckpt_path):
                print(f"未找到病害分类模型权重: {ckpt_path}")
                print("请将训练好的模型文件复制到 checkpoints/disease_classifier.pth")
                return

            state = torch.load(ckpt_path, map_location=self.cls_device)
            
            # 打印模型文件中的键用于调试
            print(f"模型文件中的键: {list(state.keys())}")
            
            # 检查模型格式 - 匹配训练脚本保存的格式
            if 'class_to_idx' not in state:
                print("模型文件中缺少 class_to_idx")
                return
            
            if 'model_state_dict' not in state:
                print("模型文件中缺少 model_state_dict")
                return
            
            class_to_idx = state['class_to_idx']
            state_dict = state['model_state_dict']
            
            print(f"类别映射: {class_to_idx}")
            
            num_classes = len(class_to_idx)
            print(f"类别数量: {num_classes}")

            # 创建模型
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except:
                model = models.resnet18(pretrained=True)
            
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            
            # 加载权重
            model.load_state_dict(state_dict, strict=False)
            print("权重加载成功")
            
            model.to(self.cls_device)
            model.eval()

            self.cls_model = model
            self.cls_class_to_idx = class_to_idx
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}

            self.cls_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            print("✅ 病害分类模型加载完成，类别映射：", self.idx_to_class)

        except Exception as e:
            print(f"❌ 加载病害分类模型失败: {e}")
            import traceback
            traceback.print_exc()
            self.cls_model = None

    def auto_predict_disease_type(self):
        """根据当前掩膜和原图，自动预测病害类型"""
        print("=== 开始自动预测 ===")  # 添加调试信息
        
        if self.cls_model is None:
            print("cls_model is None")
            return None
        if self.current_mask_full is None:
            print("current_mask_full is None")
            return None
        if self.original_image is None:
            print("original_image is None")
            return None
        if self.cls_transform is None:
            print("cls_transform is None")
            return None
        if self.idx_to_class is None:
            print("idx_to_class is None")
            return None

        try:
            mask = self.current_mask_full.astype(np.uint8)
            img = self.original_image

            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                print("掩膜中没有正像素")
                return None

            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            print(f"病害区域边界: y[{y0}:{y1}], x[{x0}:{x1}]")

            h, w = mask.shape[:2]
            margin = 10
            y0 = max(int(y0) - margin, 0)
            y1 = min(int(y1) + margin, h - 1)
            x0 = max(int(x0) - margin, 0)
            x1 = min(int(x1) + margin, w - 1)
            print(f"扩展后区域: y[{y0}:{y1}], x[{x0}:{x1}]")

            patch = img[y0:y1 + 1, x0:x1 + 1, :]
            print(f"裁剪图像尺寸: {patch.shape}")

            x = self.cls_transform(patch)
            x = x.unsqueeze(0).to(self.cls_device)
            print(f"转换后张量形状: {x.shape}")

            with torch.no_grad():
                logits = self.cls_model(x)
                probs = torch.softmax(logits, dim=1)[0]
                idx = int(torch.argmax(probs).item())
                prob = float(probs[idx].item())

            cls_name = self.idx_to_class.get(idx, None)
            if cls_name is None:
                print(f"找不到类别索引 {idx}")
                return None

            print(f"自动预测病害类型: {cls_name} (置信度 {prob:.3f})")
            return cls_name, prob

        except Exception as e:
            print(f"自动预测病害类型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_progress(self, progress, message):
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.status_var.set(f"{message} ({progress:.1f}%)")

    # ====================== 图像载入 & 预览 ======================

    def open_image(self):
        Image.MAX_IMAGE_PIXELS = None

        file_path = filedialog.askopenfilename(
            title="选择壁画图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if not file_path:
            return

        try:
            pil_image = Image.open(file_path)
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert('RGB')

            self.original_image = np.array(pil_image)

            h_full, w_full = self.original_image.shape[:2]
            max_side_full = max(h_full, w_full)
            if max_side_full > self.max_preview_side:
                self.display_scale = self.max_preview_side / float(max_side_full)
            else:
                self.display_scale = 1.0

            if self.display_scale < 1.0:
                w_prev = int(w_full * self.display_scale)
                h_prev = int(h_full * self.display_scale)
                self.preview_image = cv2.resize(
                    self.original_image,
                    (w_prev, h_prev),
                    interpolation=cv2.INTER_AREA
                )
                print(f"生成预览图: {w_prev}x{h_prev} (原图: {w_full}x{h_full})")
            else:
                self.preview_image = self.original_image.copy()

            self.current_image = self.preview_image.copy()
            self.base_display_image = self.preview_image.copy()

            # 重置状态
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.scale_factor = 1.0
            self.current_points = []
            self.current_labels = []
            self.saved_results = []
            self.current_mask = None
            self.current_mask_full = None
            self.active_result_index = -1
            self.point_items = []
            self.stats_cache = None
            self.stats_dirty = True

            self.is_manual_mode = False
            self.manual_mode_type = None
            self.manual_points = []
            self.clear_manual_drawing()

            # 清空全图滑窗结果
            self.global_mask_full = None
            self.global_mask_preview = None
            self.show_global_result.set(False)

            self.update_results_list()
            self.calculate_auto_scale()
            self.display_image()

            file_name = os.path.basename(file_path)
            self.status_var.set(
                f"已加载: {file_name} (原图: {w_full}x{h_full}，预览: {self.preview_image.shape[1]}x{self.preview_image.shape[0]})"
            )
            self.update_info()

        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {e}")

    def calculate_auto_scale(self):
        if self.preview_image is None:
            return

        self.root.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600

        h, w = self.preview_image.shape[:2]
        scale_x = canvas_width / w
        scale_y = canvas_height / h
        self.scale_factor = min(scale_x, scale_y, 1.0)

    def calculate_display_position(self):
        if self.preview_image is None:
            return 0, 0

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        h, w = self.preview_image.shape[:2]

        scaled_w = int(w * self.scale_factor)
        scaled_h = int(h * self.scale_factor)

        x = (canvas_width - scaled_w) // 2 + self.pan_offset_x
        y = (canvas_height - scaled_h) // 2 + self.pan_offset_y
        return x, y

    # ====================== 模式 ======================
    
    def toggle_manual_polygon_mode(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        if self.is_manual_mode and self.manual_mode_type == "polygon":
            self.cancel_manual_mode()
            return

        self.is_manual_mode = True
        self.manual_mode_type = "polygon"
        self.canvas.config(cursor="plus")
        self.manual_points = []
        self.clear_manual_drawing()
        self.clear_current_marks()
        self.status_var.set("多边形模式 - 左键添加顶点，右键闭合多边形")

    def toggle_manual_polyline_mode(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        if self.is_manual_mode and self.manual_mode_type == "polyline":
            self.cancel_manual_mode()
            return

        self.is_manual_mode = True
        self.manual_mode_type = "polyline"
        self.canvas.config(cursor="plus")
        self.manual_points = []
        self.clear_manual_drawing()
        self.clear_current_marks()
        self.status_var.set("裂隙线段模式 - 左键添加折线点，右键结束绘制")

    def cancel_manual_mode(self):
        if self.is_manual_mode:
            self.is_manual_mode = False
            self.manual_mode_type = None
            self.canvas.config(cursor="crosshair")
            self.manual_points = []
            self.clear_manual_drawing()
            self.status_var.set("已退出分割模式")

    def clear_manual_drawing(self):
        for item in self.manual_line_items + self.manual_polygon_items:
            self.canvas.delete(item)
        self.manual_line_items = []
        self.manual_polygon_items = []

    def on_left_click(self, event):
        if self.preview_image is None:
            return

        if self.is_manual_mode:
            self.add_manual_point(event)
            return

        # 采集点提示
        x0, y0 = self.calculate_display_position()
        disp_x = (event.x - x0) / self.scale_factor
        disp_y = (event.y - y0) / self.scale_factor

        h_prev, w_prev = self.preview_image.shape[:2]
        if 0 <= disp_x < w_prev and 0 <= disp_y < h_prev:
            full_x = int(disp_x / self.display_scale)
            full_y = int(disp_y / self.display_scale)
            h_full, w_full = self.original_image.shape[:2]
            if 0 <= full_x < w_full and 0 <= full_y < h_full:
                self.current_points.append([full_x, full_y])
                self.current_labels.append(1)
                self.current_mask = None
                self.current_mask_full = None
                self.draw_point_immediately(event.x, event.y, "green")
                self.status_var.set(f"添加病害标记点 ({full_x}, {full_y})")

    def on_negative_click(self, event):
        if self.preview_image is None or self.is_manual_mode:
            return

        x0, y0 = self.calculate_display_position()
        disp_x = (event.x - x0) / self.scale_factor
        disp_y = (event.y - y0) / self.scale_factor

        h_prev, w_prev = self.preview_image.shape[:2]
        if 0 <= disp_x < w_prev and 0 <= disp_y < h_prev:
            full_x = int(disp_x / self.display_scale)
            full_y = int(disp_y / self.display_scale)
            h_full, w_full = self.original_image.shape[:2]
            if 0 <= full_x < w_full and 0 <= full_y < h_full:
                self.current_points.append([full_x, full_y])
                self.current_labels.append(0)
                self.current_mask = None
                self.current_mask_full = None
                self.draw_point_immediately(event.x, event.y, "red")
                self.status_var.set(f"添加非病害标记点 ({full_x}, {full_y})")

    def add_manual_point(self, event):
        if not self.is_manual_mode:
            return

        x0, y0 = self.calculate_display_position()
        disp_x = (event.x - x0) / self.scale_factor
        disp_y = (event.y - y0) / self.scale_factor

        h_prev, w_prev = self.preview_image.shape[:2]
        if 0 <= disp_x < w_prev and 0 <= disp_y < h_prev:
            full_x = int(disp_x / self.display_scale)
            full_y = int(disp_y / self.display_scale)
            h_full, w_full = self.original_image.shape[:2]
            if 0 <= full_x < w_full and 0 <= full_y < h_full:
                self.manual_points.append((full_x, full_y))

                radius = 4
                point_item = self.canvas.create_oval(
                    event.x - radius, event.y - radius,
                    event.x + radius, event.y + radius,
                    fill="blue", outline="white", width=1,
                    tags="manual_points"
                )
                self.manual_line_items.append(point_item)

                if len(self.manual_points) > 1:
                    prev_x_full, prev_y_full = self.manual_points[-2]
                    prev_x_disp = prev_x_full * self.display_scale
                    prev_y_disp = prev_y_full * self.display_scale
                    canvas_prev_x = int(prev_x_disp * self.scale_factor) + x0
                    canvas_prev_y = int(prev_y_disp * self.scale_factor) + y0
                    line_item = self.canvas.create_line(
                        canvas_prev_x, canvas_prev_y,
                        event.x, event.y,
                        fill="cyan", width=2, tags="manual_lines"
                    )
                    self.manual_line_items.append(line_item)

                mode_text = "多边形" if self.manual_mode_type == "polygon" else "线段"
                self.status_var.set(
                    f"{mode_text} - 已添加顶点 {len(self.manual_points)}: ({full_x}, {full_y})"
                )

    def finish_manual_shape(self):
        if not self.is_manual_mode:
            return
        if self.manual_mode_type == "polygon":
            self.complete_manual_polygon()
        elif self.manual_mode_type == "polyline":
            self.complete_manual_polyline()

    def complete_manual_polygon(self):
        if len(self.manual_points) < 3:
            messagebox.showwarning("警告", "多边形至少需要 3 个点")
            return False

        mask_full = self.create_polygon_mask(self.manual_points)
        if mask_full is None:
            messagebox.showerror("错误", "创建多边形掩膜失败")
            return False

        if self.display_scale != 1.0:
            h_prev, w_prev = self.preview_image.shape[:2]
            mask_preview = cv2.resize(mask_full.astype(np.uint8),
                                      (w_prev, h_prev),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_preview = mask_full

        self.current_mask_full = mask_full
        self.current_mask = mask_preview
        self.show_current_result.set(True)

        self.current_points = [list(p) for p in self.manual_points]
        self.current_labels = [1] * len(self.manual_points)

        self.display_image()

        x0, y0 = self.calculate_display_position()
        polygon_points = []
        for px_full, py_full in self.manual_points:
            px_disp = px_full * self.display_scale
            py_disp = py_full * self.display_scale
            canvas_x = int(px_disp * self.scale_factor) + x0
            canvas_y = int(py_disp * self.scale_factor) + y0
            polygon_points.extend([canvas_x, canvas_y])

        polygon_item = self.canvas.create_polygon(
            polygon_points,
            outline="yellow", fill="", width=3,
            tags="manual_polygon"
        )
        self.manual_polygon_items.append(polygon_item)

        self.status_var.set(f"多边形完成 - 多边形面积: {np.sum(mask_full)} 像素")
        return True

    def complete_manual_polyline(self):
        if len(self.manual_points) < 2:
            messagebox.showwarning("警告", "线段至少需要 2 个点")
            return False

        mask_full = self.create_polyline_mask(self.manual_points, thickness=3)
        if mask_full is None:
            messagebox.showerror("错误", "创建线段掩膜失败")
            return False

        if self.display_scale != 1.0:
            h_prev, w_prev = self.preview_image.shape[:2]
            mask_preview = cv2.resize(mask_full.astype(np.uint8),
                                      (w_prev, h_prev),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_preview = mask_full

        self.current_mask_full = mask_full
        self.current_mask = mask_preview
        self.show_current_result.set(True)

        self.current_points = [list(p) for p in self.manual_points]
        self.current_labels = [1] * len(self.manual_points)

        self.display_image()
        self.status_var.set(f"裂隙线段完成 - 线段覆盖像素: {np.sum(mask_full)}")
        return True

    def create_polygon_mask(self, points):
        h, w = self.original_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask.astype(bool)

    def create_polyline_mask(self, points, thickness=3):
        h, w = self.original_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 255, thickness)
        return mask.astype(bool)

    def cancel_manual_drawing(self, *_):
        self.manual_points = []
        self.clear_manual_drawing()
        self.status_var.set("已取消当前绘制")

    # ====================== 平移 & 缩放 ======================

    def on_pan_start(self, event):
        if self.is_manual_mode and len(self.manual_points) >= 2:
            self.finish_manual_shape()
        else:
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y

    def on_pan_motion(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.display_image()

    def on_pan_end(self, event):
        self.is_panning = False

    def on_mousewheel(self, event):
        if self.preview_image is None:
            return

        h_prev, w_prev = self.preview_image.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        old_scale = self.scale_factor
        zoom = 1.1

        if event.delta > 0:
            new_scale = old_scale * zoom
            max_scale = 200.0
            new_scale = min(new_scale, max_scale)
        else:
            new_scale = old_scale / zoom
            min_scale = max(0.01, 100 / max(w_prev, h_prev))
            new_scale = max(new_scale, min_scale)

        if abs(new_scale - old_scale) < 1e-6:
            return

        scaled_w_old = w_prev * old_scale
        scaled_h_old = h_prev * old_scale
        x0_old = (canvas_width - scaled_w_old) / 2 + self.pan_offset_x
        y0_old = (canvas_height - scaled_h_old) / 2 + self.pan_offset_y

        ix = (event.x - x0_old) / old_scale
        iy = (event.y - y0_old) / old_scale

        scaled_w_new = w_prev * new_scale
        scaled_h_new = h_prev * new_scale
        pan_x_new = event.x - ix * new_scale - (canvas_width - scaled_w_new) / 2
        pan_y_new = event.y - iy * new_scale - (canvas_height - scaled_h_new) / 2

        self.scale_factor = new_scale
        self.pan_offset_x = int(pan_x_new)
        self.pan_offset_y = int(pan_y_new)

        self.display_image()
        self.status_var.set(f"缩放: {self.scale_factor:.2f}x")

    # ====================== 显示 ======================

    def rebuild_base_display_image(self):
        if self.preview_image is None:
            self.base_display_image = None
            return

        base = self.preview_image.copy()
        if self.saved_results:
            for result in self.saved_results:
                if result.preview_mask is not None and result.disease_type:
                    base = apply_pattern_fill(
                        base,
                        result.preview_mask,
                        result.disease_type,
                        result.pattern_image,
                        self.disease_colors
                    )
        self.base_display_image = base

    def create_display_image(self):
        if self.preview_image is None:
            return None

        if self.show_saved_results.get() and self.base_display_image is not None:
            display_image = self.base_display_image.copy()
        else:
            display_image = self.preview_image.copy()

        # 全图滑窗自动结果
        if self.show_global_result.get() and self.global_mask_preview is not None:
            mask_area = self.global_mask_preview.astype(bool)
            if np.any(mask_area):
                if self.show_fill.get():
                    over = np.zeros_like(display_image)
                    over[mask_area] = [*self.global_fill_color]
                    mask_3 = np.stack([mask_area] * 3, axis=2)
                    display_image = np.where(
                        mask_3,
                        display_image * (1 - self.fill_alpha) + over * self.fill_alpha,
                        display_image
                    ).astype(np.uint8)

        # 当前分割结果
        if self.show_current_result.get() and self.current_mask is not None:
            mask_area = self.current_mask.astype(bool)
            if self.show_fill.get():
                red_fill = np.zeros_like(display_image)
                red_fill[mask_area] = [*self.fill_color]
                mask_3 = np.stack([mask_area] * 3, axis=2)
                display_image = np.where(
                    mask_3,
                    display_image * (1 - self.fill_alpha) + red_fill * self.fill_alpha,
                    display_image
                ).astype(np.uint8)

            if self.show_boundary.get():
                try:
                    contours, _ = cv2.findContours(
                        self.current_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(display_image, contours, -1, self.boundary_color, 2)
                except Exception as e:
                    print(f"当前边界绘制失败: {e}")

        return display_image

    def display_image(self):
        if self.preview_image is None:
            return

        self.canvas.delete("image")
        x0, y0 = self.calculate_display_position()

        h_prev, w_prev = self.preview_image.shape[:2]
        scaled_w = int(w_prev * self.scale_factor)
        scaled_h = int(h_prev * self.scale_factor)

        display_array = self.create_display_image()
        if display_array is None:
            return

        pil_image = Image.fromarray(display_array.astype(np.uint8))
        if self.scale_factor != 1.0:
            pil_image = pil_image.resize(
                (scaled_w, scaled_h),
                Image.Resampling.LANCZOS
            )

        new_display_image = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(
            x0, y0, anchor=tk.NW,
            image=new_display_image, tags="image"
        )
        self.display_image_pil = new_display_image

        self.update_points_display()
        self.update_info()
        self.canvas.update_idletasks()

    def update_points_display(self):
        for item in self.point_items:
            self.canvas.delete(item)
        self.point_items = []

        if not self.current_points or self.preview_image is None:
            return

        x0, y0 = self.calculate_display_position()

        for point, label in zip(self.current_points, self.current_labels):
            full_x, full_y = point
            disp_x = full_x * self.display_scale
            disp_y = full_y * self.display_scale
            canvas_x = int(disp_x * self.scale_factor) + x0
            canvas_y = int(disp_y * self.scale_factor) + y0

            color = "green" if label == 1 else "red"
            radius = 4
            item = self.canvas.create_oval(
                canvas_x - radius, canvas_y - radius,
                canvas_x + radius, canvas_y + radius,
                fill=color, outline="white", width=1,
                tags="points"
            )
            self.point_items.append(item)

    def draw_point_immediately(self, canvas_x, canvas_y, color):
        radius = 4
        item = self.canvas.create_oval(
            canvas_x - radius, canvas_y - radius,
            canvas_x + radius, canvas_y + radius,
            fill=color, outline="white", width=1,
            tags="points"
        )
        self.point_items.append(item)

    def on_alpha_change(self, value):
        self.fill_alpha = float(value)
        self.display_image()

    # ====================== SAM2 分割（局部点提示） ======================

    def _compute_visible_original_region(self):
        if self.preview_image is None or self.original_image is None:
            return None

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return None

        h_prev, w_prev = self.preview_image.shape[:2]
        x0, y0 = self.calculate_display_position()

        vx0 = max(0.0, (0 - x0) / self.scale_factor)
        vy0 = max(0.0, (0 - y0) / self.scale_factor)
        vx1 = min(float(w_prev), (canvas_width - x0) / self.scale_factor)
        vy1 = min(float(h_prev), (canvas_height - y0) / self.scale_factor)

        if vx1 <= vx0 or vy1 <= vy0:
            return None

        h_full, w_full = self.original_image.shape[:2]

        ox0 = int(max(0, np.floor(vx0 / self.display_scale)))
        oy0 = int(max(0, np.floor(vy0 / self.display_scale)))
        ox1 = int(min(w_full, np.ceil(vx1 / self.display_scale)))
        oy1 = int(min(h_full, np.ceil(vy1 / self.display_scale)))

        if ox1 <= ox0 or oy1 <= oy0:
            return None

        return ox0, oy0, ox1, oy1

    def sam2_segment(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return
        if not self.sam2_segmenter.is_loaded:
            messagebox.showwarning("警告", "SAM2 模型尚未加载完成")
            return
        if not self.current_points:
            messagebox.showwarning("警告", "请先标记病害区域")
            return
        if sum(1 for l in self.current_labels if l == 1) == 0:
            messagebox.showwarning("警告", "请至少标记一个病害区域点")
            return

        region = self._compute_visible_original_region()
        if region is None:
            messagebox.showwarning("警告", "无法确定当前视图对应的原图区域，请稍微缩放或移动后重试")
            return
        ox0, oy0, ox1, oy1 = region
        crop_w = ox1 - ox0
        crop_h = oy1 - oy0

        crop_points = []
        crop_labels = []
        for (x, y), l in zip(self.current_points, self.current_labels):
            if ox0 <= x < ox1 and oy0 <= y < oy1:
                crop_points.append([x - ox0, y - oy0])
                crop_labels.append(l)

        if not crop_points:
            messagebox.showwarning(
                "警告",
                "当前标记点不在视图对应的裁剪区域内，请在当前视图中添加病害点后再分割"
            )
            return

        self.current_mask = None
        self.current_mask_full = None
        self.display_image()
        self.show_current_result.set(True)

        self.is_segmenting = True
        self.status_var.set(f"正在进行 SAM2 分割（局部 {crop_w}x{crop_h}）...")

        def segment_thread():
            try:
                crop_img = self.original_image[oy0:oy1, ox0:ox1]

                mask_local = self.sam2_segmenter.segment_with_points(
                    crop_img,
                    crop_points,
                    crop_labels,
                    self.update_progress
                )
                if mask_local is not None:
                    h_full, w_full = self.original_image.shape[:2]
                    mask_full = np.zeros((h_full, w_full), dtype=bool)
                    mask_full[oy0:oy1, ox0:ox1] = mask_local

                    self.root.after(0, lambda: self.on_segmentation_complete(mask_full))
                else:
                    self.root.after(0, lambda: self.on_segmentation_failed("分割失败"))
            except Exception as e:
                self.root.after(0, lambda: self.on_segmentation_failed(f"分割错误: {e}"))
            finally:
                self.root.after(0, lambda: setattr(self, "is_segmenting", False))

        threading.Thread(target=segment_thread, daemon=True).start()

    def on_segmentation_complete(self, mask_full):
        self.current_mask_full = mask_full.copy()
        if self.display_scale != 1.0:
            h_prev, w_prev = self.preview_image.shape[:2]
            mask_preview = cv2.resize(
                mask_full.astype(np.uint8),
                (w_prev, h_prev),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_preview = mask_full

        self.current_mask = mask_preview
        self.show_current_result.set(True)

        self.display_image()

        mask_pixels = np.sum(mask_full)
        total_pixels = mask_full.shape[0] * mask_full.shape[1]
        mask_ratio = mask_pixels / total_pixels
        self.status_var.set(f"分割完成! 病害区域占比: {mask_ratio:.2%}")
        self.update_info()

    def on_segmentation_failed(self, message):
        self.status_var.set(message)
        messagebox.showerror("错误", message)

    # ====================== 全图滑窗自动分割 ======================

    def sam2_sliding_window_segment(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return
        if not self.sam2_segmenter.is_loaded:
            messagebox.showwarning("警告", "SAM2 模型尚未加载完成")
            return
        if self.is_segmenting:
            messagebox.showinfo("提示", "正在进行其它分割，请稍候完成后再试。")
            return
        
        # 检查是否有标记点
        if not self.current_points:
            result = messagebox.askyesno(
                "提示", 
                "当前没有标记点，是否继续使用均匀采样点进行分割？\n\n选择'是'：使用均匀采样点\n选择'否'：返回添加标记点"
            )
            if not result:
                return
            # 使用空点列表
            points = []
            labels = []
        else:
            points = self.current_points
            labels = self.current_labels
            print(f"使用 {len(points)} 个标记点进行滑窗分割")

        H, W = self.original_image.shape[:2]
        self.global_mask_full = np.zeros((H, W), dtype=bool)
        self.global_mask_preview = None
        self.show_global_result.set(True)
        self.show_fill.set(True)
        self.is_segmenting = True
        self.status_var.set("开始全图滑窗自动分割...")

        def progress_cb(p, msg):
            self.root.after(0, lambda: self.update_progress(p, msg))

        def tile_cb(mask_so_far: np.ndarray):
            def ui_update():
                self.global_mask_full = mask_so_far
                if self.preview_image is not None:
                    h_prev, w_prev = self.preview_image.shape[:2]
                    if (h_prev, w_prev) != mask_so_far.shape:
                        preview_mask = cv2.resize(
                            mask_so_far.astype(np.uint8),
                            (w_prev, h_prev),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        preview_mask = mask_so_far
                    self.global_mask_preview = preview_mask
                self.show_global_result.set(True)
                self.display_image()
                self.root.update_idletasks()
            self.root.after(0, ui_update)

        def worker():
            try:
                full_mask = self.sam2_segmenter.sliding_window_inference(
                    self.original_image,
                    global_points=points,  # 传递点坐标
                    global_labels=labels,  # 传递点标签
                    window_size=2048,
                    stride=1024,
                    point_grid=2,
                    score_thresh=0.5,
                    progress_callback=progress_cb,
                    tile_callback=tile_cb
                )

                def finish():
                    self.global_mask_full = full_mask
                    if self.preview_image is not None:
                        h_prev, w_prev = self.preview_image.shape[:2]
                        if (h_prev, w_prev) != full_mask.shape:
                            preview_mask = cv2.resize(
                                full_mask.astype(np.uint8),
                                (w_prev, h_prev),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        else:
                            preview_mask = full_mask
                        self.global_mask_preview = preview_mask

                    self.show_global_result.set(True)
                    self.is_segmenting = False
                    self.status_var.set("全图滑窗自动分割完成")
                    self.display_image()
                    self.update_info()

                self.root.after(0, finish)
            except Exception as e:
                def fail():
                    self.is_segmenting = False
                    self.status_var.set(f"全图滑窗分割失败: {e}")
                    messagebox.showerror("错误", f"全图滑窗分割失败: {e}")
                self.root.after(0, fail)

        threading.Thread(target=worker, daemon=True).start()

    # ====================== 保存/管理分割结果 ======================

    def ask_disease_type(self, default_type=None, auto_confirm=False):
        """
        选择病害类型
        - default_type: 默认选中的类型
        - auto_confirm: 如果为True且default_type不为None，直接返回default_type
        """
        disease_types = ["颜料层脱落", "地仗层脱落", "水渍", "裂隙"]
        
        # 如果开启自动确认且有默认类型，直接返回
        if auto_confirm and default_type is not None and default_type in disease_types:
            return default_type

        dialog = tk.Toplevel(self.root)
        dialog.title("选择病害类型")
        dialog.geometry("300x220")
        dialog.transient(self.root)
        dialog.grab_set()

        if default_type in disease_types:
            initial = default_type
        else:
            initial = disease_types[0]

        selected_type = tk.StringVar(value=initial)

        ttk.Label(dialog, text="请选择病害类型:", font=("Arial", 12)).pack(pady=10)
        for disease in disease_types:
            ttk.Radiobutton(
                dialog, text=disease,
                variable=selected_type, value=disease
            ).pack(anchor=tk.W, padx=20, pady=2)

        if default_type is not None:
            ttk.Label(
                dialog,
                text=f"自动预测: {default_type}",
                foreground="blue"
            ).pack(pady=5)

        def on_confirm():
            dialog.result = selected_type.get()
            dialog.destroy()

        def on_cancel():
            dialog.result = None
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="确认", command=on_confirm).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=on_cancel).pack(
            side=tk.LEFT, padx=5)

        self.root.wait_window(dialog)
        return getattr(dialog, "result", None)

    def save_current_segmentation(self):
        # 检查是否有可保存的结果（当前分割结果或全局滑窗结果）
        has_current = self.current_mask_full is not None
        has_global = self.global_mask_full is not None and np.any(self.global_mask_full)
        
        if not has_current and not has_global:
            messagebox.showwarning("警告", "没有可保存的分割结果")
            return

        # 确定要保存的是哪个结果
        save_global = False
        if has_current and has_global:
            # 两者都有，让用户选择
            result = messagebox.askyesnocancel(
                "选择保存结果",
                "检测到当前有多个分割结果，请选择：\n\n"
                "是 - 保存当前分割结果\n"
                "否 - 保存全图滑窗结果\n"
                "取消 - 不保存"
            )
            if result is None:  # 取消
                return
            elif result is False:  # 保存全局
                save_global = True
            # result为True时保存当前，保持save_global=False
        elif has_global and not has_current:
            save_global = True
        # 如果只有has_current，save_global保持False

        # 获取要保存的掩膜
        if save_global:
            mask_to_save = self.global_mask_full
            points_to_save = []  # 全局滑窗没有对应的点
            labels_to_save = []
            is_manual = False
            source_name = "全图滑窗"
        else:
            mask_to_save = self.current_mask_full
            points_to_save = self.current_points
            labels_to_save = self.current_labels
            is_manual = self.is_manual_mode or len(self.manual_points) > 0
            source_name = "当前分割"

        if mask_to_save is None or not np.any(mask_to_save):
            messagebox.showwarning("警告", f"{source_name}掩膜为空，无法保存")
            return

        # 自动预测病害类型
        auto_type = None
        auto_prob = None
        confidence_msg = ""
        auto_info = self.auto_predict_disease_type()
        
        if auto_info is not None:
            auto_type, auto_prob = auto_info
            print(f"{source_name}自动预测类型: {auto_type} (置信度 {auto_prob:.3f})")
            
            if self.auto_use_prediction.get():
                # 如果开启自动使用，直接使用预测结果
                disease_type = auto_type
                confidence_msg = f" (置信度: {auto_prob:.1%})"
            else:
                # 如果关闭自动使用，显示预测结果供参考，但仍让用户选择
                # 注意：这里传入 auto_confirm=False（默认值），所以会弹出窗口
                disease_type = self.ask_disease_type(default_type=auto_type, auto_confirm=False)
                if not disease_type:
                    return
                confidence_msg = f" (预测建议: {auto_type})"
        else:
            # 如果自动预测失败，才让用户选择
            disease_type = self.ask_disease_type(default_type=None, auto_confirm=False)
            if not disease_type:
                return
            confidence_msg = ""

        # 创建分割结果对象
        result = SegmentationResult(
            mask=mask_to_save.copy(),
            points=[p.copy() if isinstance(p, list) else list(p) for p in points_to_save],
            labels=labels_to_save.copy(),
            disease_type=disease_type,
            pattern_path=None,
            is_manual=is_manual
        )

        # 设置预览掩膜
        if self.preview_image is not None:
            h_prev, w_prev = self.preview_image.shape[:2]
            if mask_to_save.shape[:2] != (h_prev, w_prev):
                preview_mask = cv2.resize(
                    mask_to_save.astype(np.uint8),
                    (w_prev, h_prev),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                preview_mask = mask_to_save
            result.preview_mask = preview_mask

        # 生成图例图像
        result.pattern_image = generate_pattern_image(disease_type, self.disease_colors)
        
        # 添加到保存结果列表
        self.saved_results.append(result)

        self.stats_dirty = True
        self.update_results_list()
        self.rebuild_base_display_image()

        # 清除所有正/负样本点
        self.current_points = []
        self.current_labels = []
        for item in self.point_items:
            self.canvas.delete(item)
        self.point_items = []
        
        # 清除当前分割掩膜
        self.current_mask = None
        self.current_mask_full = None
        self.show_current_result.set(False)
        
        # 如果保存的是全局滑窗结果，也清除全局结果
        if save_global:
            self.global_mask_full = None
            self.global_mask_preview = None
            self.show_global_result.set(False)
            self.status_var.set("已保存全图滑窗结果并清除所有标记点")
        else:
            self.status_var.set("已保存当前分割结果并清除所有标记点")
        
        # 清除手动模式相关
        self.manual_points = []
        self.clear_manual_drawing()
        self.is_manual_mode = False
        self.manual_mode_type = None
        self.canvas.config(cursor="crosshair")

        self.show_saved_results.set(True)
        self.display_image()

        result_type = "手动" if is_manual else "SAM2"
        self.status_var.set(f"已保存{source_name}{result_type}分割结果: {result.name} ({disease_type})")
        messagebox.showinfo(
            "成功",
            f"分割结果已保存为: {result.name}\n"
            f"来源: {source_name}\n"
            f"类型: {result_type}分割\n"
            f"病害类型: {disease_type}{confidence_msg}"
        )

    def update_results_list(self):
        self.results_listbox.delete(0, tk.END)
        for i, r in enumerate(self.saved_results):
            t = "手动" if r.is_manual else "SAM2"
            self.results_listbox.insert(
                tk.END, f"{i + 1}. {r.name} [{t}] - {r.area_pixels} 像素"
            )

    def on_result_selected(self, event):
        sel = self.results_listbox.curselection()
        if sel:
            self.active_result_index = sel[0]
            self.display_image()

    def delete_selected_result(self):
        sel = self.results_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        name = self.saved_results[idx].name
        if messagebox.askyesno("确认", f"确定要删除结果 '{name}' 吗？"):
            del self.saved_results[idx]
            self.active_result_index = -1
            self.stats_dirty = True
            self.update_results_list()
            self.rebuild_base_display_image()
            self.display_image()

    def clear_all_results(self):
        if not self.saved_results:
            return
        if messagebox.askyesno("确认", "确定要清除所有保存的结果吗？"):
            self.saved_results = []
            self.active_result_index = -1
            self.stats_cache = None
            self.stats_dirty = False
            self.update_results_list()
            self.rebuild_base_display_image()
            self.display_image()

    def clear_current_marks(self):
        if not self.current_points and not self.manual_points:
            return
        self.current_points = []
        self.current_labels = []
        self.manual_points = []
        self.clear_manual_drawing()
        for item in self.point_items:
            self.canvas.delete(item)
        self.point_items = []
        self.display_image()
        self.status_var.set("已清除当前标记")

    def clear_current_segmentation(self):
        has_current = self.current_mask_full is not None or self.current_mask is not None
        has_global = self.global_mask_full is not None and np.any(self.global_mask_full)
        
        if not has_current and not has_global:
            messagebox.showinfo("提示", "没有当前分割结果可清除")
            return

        # 询问要清除哪个
        if has_current and has_global:
            result = messagebox.askyesnocancel(
                "选择清除内容",
                "检测到多个分割结果，请选择：\n\n"
                "是 - 清除当前分割结果\n"
                "否 - 清除全图滑窗结果\n"
                "取消 - 返回"
            )
            if result is None:  # 取消
                return
            elif result is False:  # 清除全局
                self.global_mask_full = None
                self.global_mask_preview = None
                self.show_global_result.set(False)
                self.status_var.set("已清除全图滑窗结果")
            else:  # 清除当前
                self.current_mask = None
                self.current_mask_full = None
                self.show_current_result.set(False)
                self.manual_points = []
                self.clear_manual_drawing()
                self.is_manual_mode = False
                self.manual_mode_type = None
                self.canvas.config(cursor="crosshair")
                self.status_var.set("已清除当前分割结果")
        elif has_global and not has_current:
            # 只有全局结果
            if messagebox.askyesno("确认", "确定要清除全图滑窗结果吗？"):
                self.global_mask_full = None
                self.global_mask_preview = None
                self.show_global_result.set(False)
                self.status_var.set("已清除全图滑窗结果")
        else:
            # 只有当前结果
            if messagebox.askyesno("确认", "确定要清除当前分割结果吗？"):
                self.current_mask = None
                self.current_mask_full = None
                self.show_current_result.set(False)
                self.manual_points = []
                self.clear_manual_drawing()
                self.is_manual_mode = False
                self.manual_mode_type = None
                self.canvas.config(cursor="crosshair")
                self.status_var.set("已清除当前分割结果")

        self.display_image()
        self.update_info()

    # ====================== 统计 & 导出 ======================

    def _reconstruct_full_mask(self, result: SegmentationResult):
        if self.original_image is None:
            return None
        h, w = self.original_image.shape[:2]
        full = np.zeros((h, w), dtype=bool)
        if result.mask is None or result.bbox is None:
            return full
        y0, x0, y1, x1 = result.bbox
        mh, mw = result.mask.shape
        y1 = min(y1, h)
        x1 = min(x1, w)
        h_sub = y1 - y0
        w_sub = x1 - x0
        full[y0:y1, x0:x1] = result.mask[:h_sub, :w_sub]
        return full

    def calculate_statistics(self, force=False):
        if self.original_image is None or not self.saved_results:
            self.stats_cache = None
            self.stats_dirty = False
            return None

        if (not force) and (not self.stats_dirty) and (self.stats_cache is not None):
            return self.stats_cache

        h, w = self.original_image.shape[:2]
        total_pixels = h * w
        stats = {
            "total_pixels": total_pixels,
            "disease_types": {},
            "total_disease_pixels": 0,
            "overall_ratio": 0.0
        }
        for r in self.saved_results:
            t = r.disease_type
            area = r.area_pixels
            if t not in stats["disease_types"]:
                stats["disease_types"][t] = {
                    "total_pixels": 0,
                    "ratio": 0.0,
                    "count": 0
                }
            stats["disease_types"][t]["total_pixels"] += area
            stats["disease_types"][t]["count"] += 1
        for t, d in stats["disease_types"].items():
            d["ratio"] = d["total_pixels"] / total_pixels
            stats["total_disease_pixels"] += d["total_pixels"]
        stats["overall_ratio"] = stats["total_disease_pixels"] / total_pixels

        self.stats_cache = stats
        self.stats_dirty = False
        return stats

    def export_statistics_report(self):
        if not self.saved_results:
            messagebox.showwarning("警告", "没有可导出的分割结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存统计报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("CSV 文件", "*.csv"), ("所有文件", "*.*")]
        )
        if not file_path:
            return

        try:
            stats = self.calculate_statistics(force=True)
            if not stats:
                messagebox.showerror("错误", "无法计算统计信息")
                return

            if file_path.lower().endswith(".csv"):
                self.export_statistics_csv(file_path, stats)
            else:
                self.export_statistics_txt(file_path, stats)

            messagebox.showinfo("成功", f"统计报告已导出到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出统计报告失败: {e}")

    def export_statistics_txt(self, file_path, stats):
        with open(file_path, "w", encoding="utf-8") as f:
            h, w = self.original_image.shape[:2]
            f.write("壁画病害分割统计报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"图像尺寸: {w} x {h} 像素\n")
            f.write(f"总像素数: {stats['total_pixels']:,}\n")
            f.write(f"分割结果总数: {len(self.saved_results)}\n\n")

            f.write("病害类型统计:\n")
            f.write("-" * 30 + "\n")
            for t, d in sorted(
                    stats["disease_types"].items(),
                    key=lambda x: x[1]["ratio"],
                    reverse=True):
                f.write(f"{t}:\n")
                f.write(f"  区域数量: {d['count']}\n")
                f.write(f"  总面积: {d['total_pixels']:,} 像素\n")
                f.write(f"  占比: {d['ratio']:.4%}\n\n")

            f.write("总体统计:\n")
            f.write("-" * 20 + "\n")
            f.write(f"病害总像素: {stats['total_disease_pixels']:,}\n")
            f.write(f"病害总面积占比: {stats['overall_ratio']:.4%}\n")
            manual = sum(1 for r in self.saved_results if r.is_manual)
            auto = len(self.saved_results) - manual
            f.write(f"分割结果: {manual}\n")
            f.write(f"自动分割结果: {auto}\n")

    def export_statistics_csv(self, file_path, stats):
        import csv
        h, w = self.original_image.shape[:2]
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["壁画病害分割统计报告"])
            writer.writerow(
                [f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([f"图像尺寸: {w} x {h} 像素"])
            writer.writerow([f"总像素数: {stats['total_pixels']:,}"])
            writer.writerow([f"分割结果总数: {len(self.saved_results)}"])
            writer.writerow([])
            writer.writerow(["病害类型统计"])
            writer.writerow(["病害类型", "区域数量", "总面积(像素)", "占比"])
            for t, d in sorted(
                    stats["disease_types"].items(),
                    key=lambda x: x[1]["ratio"],
                    reverse=True):
                writer.writerow([
                    t, d["count"], f"{d['total_pixels']:,}",
                    f"{d['ratio']:.4%}"
                ])
            writer.writerow([])
            writer.writerow(["总体统计"])
            writer.writerow(
                ["病害总像素", f"{stats['total_disease_pixels']:,}"])
            writer.writerow(
                ["病害总面积占比", f"{stats['overall_ratio']:.4%}"])
            manual = sum(1 for r in self.saved_results if r.is_manual)
            auto = len(self.saved_results) - manual
            writer.writerow(["分割结果", manual])
            writer.writerow(["自动分割结果", auto])

    def save_all_results(self):
        if not self.saved_results:
            # 检查是否有未保存的全局滑窗结果
            if self.global_mask_full is not None and np.any(self.global_mask_full):
                result = messagebox.askyesno(
                    "提示",
                    "当前有全图滑窗结果未保存，是否先保存该结果？\n\n"
                    "选择'是'：先保存全图滑窗结果\n"
                    "选择'否'：直接保存已保存的结果列表"
                )
                if result:  # 用户选择先保存全局结果
                    # 自动保存全局结果
                    mask_to_save = self.global_mask_full
                    disease_type = self.ask_disease_type(default_type="未分类")
                    if disease_type:
                        result_obj = SegmentationResult(
                            mask=mask_to_save.copy(),
                            points=[],
                            labels=[],
                            disease_type=disease_type,
                            pattern_path=None,
                            is_manual=False
                        )
                        result_obj.pattern_image = generate_pattern_image(disease_type, self.disease_colors)
                        self.saved_results.append(result_obj)
                        self.update_results_list()
                        self.stats_dirty = True
                        messagebox.showinfo("成功", "全图滑窗结果已添加到保存列表")
                    else:
                        return
                else:
                    messagebox.showwarning("警告", "没有可保存的结果")
                    return
            else:
                messagebox.showwarning("警告", "没有可保存的结果")
                return

        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return

        def save_thread():
            try:
                self.root.after(
                    0, lambda: self.status_var.set("正在保存结果..."))

                mask_path = os.path.join(save_dir, "所有病害掩膜_combined.png")
                self.save_combined_mask(mask_path)

                result_path = os.path.join(save_dir, "所有分割结果_combined.png")
                self.save_combined_result(result_path)

                info_path = os.path.join(save_dir, "分割信息.json")
                info_data = {
                    "image_size": self.original_image.shape[:2],
                    "total_results": len(self.saved_results),
                    "manual_results": sum(1 for r in self.saved_results if r.is_manual),
                    "sam2_results": sum(1 for r in self.saved_results if not r.is_manual),
                    "results": [r.to_dict() for r in self.saved_results],
                }
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info_data, f, ensure_ascii=False, indent=2)

                stats_path = os.path.join(save_dir, "病害统计报告.txt")
                stats = self.calculate_statistics(force=True)
                if stats:
                    self.export_statistics_txt(stats_path, stats)

                self.root.after(
                    0,
                    lambda: self.status_var.set(f"所有结果已保存到: {save_dir}")
                )
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "成功",
                        "所有分割结果已保存!\n"
                        f"目录: {save_dir}\n"
                        f"共 {len(self.saved_results)} 个结果\n\n"
                        "包含:\n"
                        "- 所有病害掩膜_combined.png\n"
                        "- 所有分割结果_combined.png\n"
                        "- 分割信息.json\n"
                        "- 病害统计报告.txt"
                    )
                )
            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror("错误", f"保存失败: {e}")
                )

        threading.Thread(target=save_thread, daemon=True).start()

    def save_combined_mask(self, file_path):
        if not self.saved_results:
            return
        h, w = self.original_image.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        for r in self.saved_results:
            full_mask = self._reconstruct_full_mask(r)
            combined[full_mask] = 255
        Image.fromarray(combined).save(file_path)
        print(f"所有病害掩膜已保存: {file_path}")

    def save_combined_result(self, file_path):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        for r in self.saved_results:
            full_mask = self._reconstruct_full_mask(r)
            img = apply_pattern_fill(
                img,
                full_mask,
                r.disease_type,
                r.pattern_image,
                self.disease_colors
            )
        Image.fromarray(img).save(file_path, optimize=True, quality=95)

    # ====================== 信息 ======================

    def update_info(self):
        if self.original_image is None:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "请打开图像")
            return

        h, w = self.original_image.shape[:2]
        total_pixels = h * w

        txt = f"原图尺寸: {w} x {h}\n"
        if self.preview_image is not None:
            ph, pw = self.preview_image.shape[:2]
            txt += f"预览尺寸: {pw} x {ph}\n"
            txt += f"预览缩放: {self.display_scale:.4f}\n"
        txt += f"视图缩放: {self.scale_factor:.2f}x\n\n"

        txt += f"当前标记: {len(self.current_points)} 个\n"
        txt += f"病害点: {sum(1 for l in self.current_labels if l == 1)}\n"
        txt += f"非病害点: {sum(1 for l in self.current_labels if l == 0)}\n\n"

        txt += f"已保存结果: {len(self.saved_results)} 个\n"
        manual = sum(1 for r in self.saved_results if r.is_manual)
        auto = len(self.saved_results) - manual
        txt += f"手动分割: {manual} 个\n"
        txt += f"SAM2 分割: {auto} 个\n"

        if self.saved_results:
            stats = self.calculate_statistics()
            if stats:
                txt += "\n病害面积统计:\n"
                txt += f"总病害面积: {stats['total_disease_pixels']:,} 像素\n"
                txt += f"总面积占比: {stats['overall_ratio']:.4%}\n"
                for t, d in stats["disease_types"].items():
                    txt += f"{t}: {d['ratio']:.4%} ({d['total_pixels']:,} 像素)\n"

        if self.current_mask_full is not None:
            mask_pixels = int(np.sum(self.current_mask_full))
            mask_ratio = mask_pixels / total_pixels
            txt += "\n当前分割:\n"
            txt += f"病害像素: {mask_pixels:,}\n"
            txt += f"占比: {mask_ratio:.4%}\n"

        if self.global_mask_full is not None:
            gm_pixels = int(np.sum(self.global_mask_full))
            gm_ratio = gm_pixels / total_pixels
            txt += "\n全图自动分割:\n"
            txt += f"病害像素: {gm_pixels:,}\n"
            txt += f"占比: {gm_ratio:.4%}\n"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, txt)
