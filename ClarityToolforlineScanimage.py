# -*- coding: utf-8 -*-
"""
@File    : Tools for Template Matching and Clarity Analysis
@Author  : Cichun Li
@Date    : 2025-11-06
@Copyright: Copyright (c) 2025 Cichun Li. All rights reserved.
"""




import cv2
from scipy.signal import detrend
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class TemplateMatcher:
    def __init__(self, threshold=0.9, iou_threshold=0.3):
        """
        初始化模板匹配器
        :param threshold: 匹配阈值
        :param iou_threshold: NMS的IOU阈值
        """
        self.threshold = threshold
        self.iou_threshold = iou_threshold
    
    def subpixel_peak(self, response_map, x, y):
        """
        亚像素级峰值定位（使用二次曲面拟合）
        :param response_map: 匹配结果矩阵
        :param x: 整数X坐标（列坐标）
        :param y: 整数Y坐标（行坐标）
        :return: (x_sub, y_sub) 亚像素级坐标
        """
        # 检查边界条件
        if x < 1 or x >= response_map.shape[1]-1 or y < 1 or y >= response_map.shape[0]-1:
            return (float(x), float(y))

        # 提取3x3邻域
        neighborhood = response_map[y-1:y+2, x-1:x+2]
        
        # X方向二次拟合
        dx_num = neighborhood[1, 0] - neighborhood[1, 2]
        dx_den = 2 * (neighborhood[1, 0] + neighborhood[1, 2] - 2*neighborhood[1, 1])
        dx = dx_num / dx_den if dx_den != 0 else 0.0

        # Y方向二次拟合
        dy_num = neighborhood[0, 1] - neighborhood[2, 1]
        dy_den = 2 * (neighborhood[0, 1] + neighborhood[2, 1] - 2*neighborhood[1, 1])
        dy = dy_num / dy_den if dy_den != 0 else 0.0

        return (x + dx, y + dy)
    
    def nms(self, rects, scores):
        """
        非极大值抑制实现
        :param rects: 包含矩形框的numpy数组，每个元素为[x, y, w, h]
        :param scores: 每个矩形框的匹配得分
        :return: 保留的矩形框索引列表
        """
        if len(rects) == 0:
            return []
            
        x1 = rects[:, 0]
        y1 = rects[:, 1]
        x2 = rects[:, 0] + rects[:, 2]
        y2 = rects[:, 1] + rects[:, 3]

        areas = rects[:, 2] * rects[:, 3]
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # 计算IOU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-8)
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return keep
    
    def match(self, img, template):
        """
        执行模板匹配并返回匹配结果
        :param img: 输入图像
        :param template: 模板图像
        :return: 包含匹配矩形框的numpy数组
        """
        if img is None or template is None:
            raise ValueError("无法读取图像或模板")

        self.t_h, self.t_w = template.shape[:2]
        
        # 执行模板匹配
        self.result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        print(f"匹配结果矩阵形状: {self.result.shape}")

        # 根据分数阈值收集候选点
        loc = np.where(self.result >= self.threshold)
        if len(loc[0]) == 0:
            print("未找到匹配目标")
            # return empty rectangles and an empty DataFrame with expected columns
            empty_rects = np.empty((0, 4), dtype=np.float32)
            df = pd.DataFrame(columns=['center_x', 'center_y', 'x', 'y', 'w', 'h']).astype(np.float32)
            return empty_rects, df

        rects = []
        scores = []
        for pt in zip(*loc[::-1]):  # pt = (x, y)
            x, y = pt
            # 亚像素修正
            x_sub, y_sub = self.subpixel_peak(self.result, x, y)
            
            # 收集数据
            rects.append([x_sub, y_sub, float(self.t_w), float(self.t_h)])
            scores.append(self.result[y, x])  # 使用原始得分

        # 转换为numpy数组
        rects = np.array(rects)
        scores = np.array(scores)

        # 执行NMS
        keep = self.nms(rects, scores)
        final_rects = rects[keep]
        print(final_rects)

        # 构建DataFrame
        data = []
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for rect in final_rects:
                x, y, w, h = rect
                center_x = x + w/2.0
                center_y = y + h/2.0
                
                # 记录数据
                data.append({
                    'center_x': round(center_x, 2),
                    'center_y': round(center_y, 2),
                    'x': round(x, 2),
                    'y': round(y, 2),
                    'w': round(w, 2),
                    'h': round(h, 2)
                })
            
        # 创建DataFrame
        df = pd.DataFrame(data)
        df = df.astype(np.float32)  # 确保所有列为浮点型
        df = df.sort_values(by='center_x', ascending=True).reset_index(drop=True) # 按 center_x 升序排序，并重置索引

        return final_rects, df
    
    def draw_rectangles(self, img, rects, output_path='detection_result.jpg', color=(0, 0, 255)):
        """
        在图像上绘制矩形框并保存结果
        :param img: 输入图像
        :param rects: 包含矩形框的数组list, 每个元素为[x, y, w, h]
        :param output_path: 结果图像保存路径
        :param color: 矩形框颜色
        :return: 包含检测结果的DataFrame
        """
        data = []
        # 如果是灰度图则转换为彩色
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()

        for rect in rects:
            x, y, w, h = rect
            center_x = x + w/2.0
            center_y = y + h/2.0
            
            # 记录数据
            data.append({
                'center_x': round(center_x, 2),
                'center_y': round(center_y, 2),
                'x': round(x, 2),
                'y': round(y, 2),
                'width': round(w, 2),
                'height': round(h, 2)
            })
            
            # 绘制图形
            cv2.rectangle(img_color, 
                          (int(x), int(y)), 
                          (int(x + w), int(y + h)), 
                          color, 2)
            cv2.circle(img_color, 
                      (int(round(center_x)), int(round(center_y))), 
                      3, (0, 255, 0), -1)

        # 保存结果图
        cv2.imwrite(output_path, img_color)
        print(f"结果图已保存为 {output_path}")
        
        return


class ClarityAnalyzer:
    """图像清晰度分析器，支持多种清晰度计算方法"""
    def __init__(self):
        self.methods = {
            "Tenengrad": self.calculate_tenengrad,
            "Brenner": self.calculate_brenner,
            "Entropy": self.calculate_entropy
        }

    def calculate_tenengrad(self, image):
        """计算Tenengrad清晰度值"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用Sobel算子计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度平方和
        grad_squared = grad_x**2 + grad_y**2
        tenengrad_value = np.sum(grad_squared)
        
        return tenengrad_value
    
    def calculate_brenner(self, image):
        """计算Brenner清晰度值"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 为避免 uint8 溢出，将灰度图转换为有符号或更大的整数类型
        # 使用向量化计算水平和垂直方向的二阶差分，速度更快且不会溢出
        gray = gray.astype(np.int64)

        # 水平方向差分（j 与 j+2）
        if gray.shape[1] >= 3:
            horiz = gray[:, 2:] - gray[:, :-2]
            horiz_sq_sum = np.sum(horiz * horiz, dtype=np.int64)
        else:
            horiz_sq_sum = np.int64(0)

        # 垂直方向差分（i 与 i+2）
        if gray.shape[0] >= 3:
            vert = gray[2:, :] - gray[:-2, :]
            vert_sq_sum = np.sum(vert * vert, dtype=np.int64)
        else:
            vert_sq_sum = np.int64(0)

        brenner_value = horiz_sq_sum + vert_sq_sum

        # 返回 Python 原生 int（或需要时可返回 float）
        return int(brenner_value)
    
    def calculate_entropy(self, image):
        """计算信息熵清晰度值"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算灰度直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # 计算概率分布
        total_pixels = gray.shape[0] * gray.shape[1]
        prob = hist / total_pixels
        
        # 计算信息熵
        entropy_value = 0
        for p in prob:
            if p > 0:
                entropy_value -= p * np.log2(p)
        
        return entropy_value
    
    def calculate_clarity(self, image, method="Tenengrad"):
        """根据选择的方法计算清晰度值"""
        if method == "Tenengrad":
            return self.calculate_tenengrad(image)
        elif method == "Brenner":
            return self.calculate_brenner(image)
        elif method == "Entropy":
            return self.calculate_entropy(image)
        else:
            return self.calculate_tenengrad(image)  # 默认使用Tenengrad    



def mian_arrays(img, template, match_threshold=0.86):
    """Run matching and analysis on numpy arrays (grayscale).
    Returns (rects, df, out_path).
    """
    if img is None or template is None:
        raise ValueError('img and template must be provided')

    matcher = TemplateMatcher(threshold=match_threshold, iou_threshold=0.2)
    rects, df = matcher.match(img, template)
    out_path = 'detection_result_ui.jpg'
    matcher.draw_rectangles(img, rects, output_path=out_path)

    Clarity_values = []
    matched_region_gray_values = []
    for _, row in df.iterrows():
        # ensure coordinates are within image bounds and integers
        x = int(max(0, np.floor(row['x'])))
        y = int(max(0, np.floor(row['y'])))
        w = int(max(0, np.ceil(row['w'])))
        h = int(max(0, np.ceil(row['h'])))

        # clamp to image
        x1 = min(x + w, img.shape[1])
        y1 = min(y + h, img.shape[0])
        x0 = max(0, x)
        y0 = max(0, y)

        if x0 >= x1 or y0 >= y1:
            # empty region, append NaN and skip clarity calc
            Clarity_values.append(np.nan)
            continue

        matched_region = img[y0:y1, x0:x1]
        matched_region_gray_value = cv2.mean(matched_region)[0]
        if matched_region.size == 0:
            Clarity_values.append(np.nan)
            continue

        analyzer = ClarityAnalyzer()
        tenengrad_value = analyzer.calculate_clarity(matched_region, method="Tenengrad")
        Clarity_values.append(tenengrad_value)
        matched_region_gray_values.append(matched_region_gray_value)

    # If df is empty this will add no column; otherwise attach clarity values
    if not df.empty:
        df['Tenengrad_Clarity'] = Clarity_values
        df['Matched_Region_Gray_Value'] = matched_region_gray_values

    # plotting (same as original mian)
    if not df.empty:
        # 1. 提取数据
        x = df['center_x'].values
        y_distortion = df['center_y'].values
        y_Clarity = df['Tenengrad_Clarity'].values
        y_detrended = detrend(y_distortion, type='linear')
        
        # 2. 检查关键变量是否存在（避免运行错误）
        if 'matched_region_gray_values' not in locals():
            raise ValueError("变量'matched_region_gray_values'未定义，请先提取或定义该数据")
        
        # 3. 创建2行1列的子图网格（统一管理图形和轴）
        fig, (ax_top, ax1) = plt.subplots(2, 1, figsize=(10, 6))  # 2行1列，共享一个figure
        
        # 4. 绘制上方子图（去趋势畸变数据）
        ax_top.plot(x, y_detrended, 'ro')
        ax_top.set_ylabel('Distortion Detrended (pixs)')
        ax_top.grid(True)
        
        # 5. 绘制下方子图（双Y轴：清晰度 + 灰度值）
        # 左Y轴：Tenengrad Clarity（蓝色）
        color_blue = 'tab:blue'
        ax1.set_xlabel('Camera pixel')
        ax1.set_ylabel('Tenengrad Clarity', color=color_blue)
        ax1.plot(x, y_Clarity, 'bo')
        ax1.tick_params(axis='y', labelcolor=color_blue)
        ax1.grid(True)
        
        # 右Y轴：Matched Region Gray Value（绿色）
        color_green = 'tab:green'
        ax2 = ax1.twinx()  # 共享X轴，创建第二个Y轴
        ax2.set_ylabel('Gray Value', color=color_green)
        ax2.plot(x, matched_region_gray_values, 'go')
        ax2.tick_params(axis='y', labelcolor=color_green)
        
        # 6. 调整布局，避免标签重叠
        fig.tight_layout()
        plot_path = 'clarity_analysis_ui.jpg'
        plt.savefig(plot_path)

        # 用win打开图片查看
        import os
        os.startfile(plot_path)


    return rects, df, out_path


class ImageApp:
    """Tkinter UI: open image, smooth pan/zoom, draw ROI (template), set threshold and run matching."""
    def __init__(self, root):
        self.root = root
        self.root.title("Template Matcher UI")

        # Top controls
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, side=tk.TOP)

        open_btn = tk.Button(ctrl, text="Open Image", command=self.open_image)
        open_btn.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(ctrl, text="Match Threshold:").pack(side=tk.LEFT, padx=(8,2))
        self.thresh_var = tk.DoubleVar(value=0.86)
        self.thresh_entry = tk.Entry(ctrl, textvariable=self.thresh_var, width=6)
        self.thresh_entry.pack(side=tk.LEFT)

        run_btn = tk.Button(ctrl, text="Run Match", command=self.run_match)
        run_btn.pack(side=tk.RIGHT, padx=4)

        confirm_btn = tk.Button(ctrl, text="Confirm ROI", command=self.confirm_roi)
        confirm_btn.pack(side=tk.RIGHT, padx=4)

        clear_btn = tk.Button(ctrl, text="Clear ROI", command=self.clear_roi)
        clear_btn.pack(side=tk.RIGHT, padx=4)

        # Canvas
        self.canvas = tk.Canvas(root, bg='#333333', width=900, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings: left button = ROI draw, right button = pan, mouse wheel = zoom (center)
        self.canvas.bind('<ButtonPress-1>', self.on_left_press)
        self.canvas.bind('<B1-Motion>', self.on_left_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_left_release)

        self.canvas.bind('<ButtonPress-3>', self.on_right_press)
        self.canvas.bind('<B3-Motion>', self.on_right_move)

        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)  # Windows

        # Image state
        self.image = None              # original as RGB numpy (H,W,3)
        self.image_pil = None          # PIL.Image of original for resizing
        self.photo = None              # ImageTk.PhotoImage currently displayed
        self.image_id = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # ROI drawing
        self.drawing = False
        self.roi_start = None  # canvas coords
        self.roi_rect_id = None
        self.confirmed_roi = None  # (x0,y0,x1,y1) in image pixels (original image coords)
        self._overlay_ids = []

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image files','*.png;*.jpg;*.jpeg;*.bmp;*.tif')])
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror('Error', 'Failed to open image')
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.image = rgb
        self.image_pil = Image.fromarray(rgb)
        # reset view
        self.scale = min(self.canvas.winfo_width() / max(1, self.image.shape[1]),
                         self.canvas.winfo_height() / max(1, self.image.shape[0]))
        if self.scale <= 0:
            self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.confirmed_roi = None
        self.last_drawn_roi = None
        self.roi_rect_id = None
        self._overlay_ids = []
        self.display_image()

    def display_image(self):
        if self.image_pil is None:
            return
        # resize for current scale
        w = max(1, int(self.image_pil.width * self.scale))
        h = max(1, int(self.image_pil.height * self.scale))
        resized = self.image_pil.resize((w, h), resample=Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        # redraw everything (image + overlays). We recreate overlays after deleting.
        # Keep last_drawn_roi / confirmed_roi to re-create rectangles so ROI persists across redraws.
        last_temp = getattr(self, 'last_drawn_roi', None)
        confirmed = getattr(self, 'confirmed_roi', None)

        self.canvas.delete('all')
        # create image
        self.image_id = self.canvas.create_image(int(self.offset_x), int(self.offset_y), anchor=tk.NW, image=self.photo)

        # recreate temporary last-drawn ROI (red)
        self.roi_rect_id = None
        if last_temp is not None and getattr(self, 'drawing', False) is False:
            try:
                tx0, ty0, tx1, ty1 = last_temp
                cx0, cy0 = self.image_to_canvas(tx0, ty0)
                cx1, cy1 = self.image_to_canvas(tx1, ty1)
                self.roi_rect_id = self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline='red', width=2)
            except Exception:
                self.roi_rect_id = None

        # recreate confirmed ROI (lime) on top
        if confirmed is not None:
            try:
                x0, y0, x1, y1 = confirmed
                cx0, cy0 = self.image_to_canvas(x0, y0)
                cx1, cy1 = self.image_to_canvas(x1, y1)
                # if there is an existing temp rect, draw confirmed on top
                self.roi_rect_id = self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline='lime', width=2)
            except Exception:
                pass

        # keep a list of overlay IDs for easy moving during pan
        self._overlay_ids = []
        if self.roi_rect_id:
            self._overlay_ids.append(self.roi_rect_id)

    def image_to_canvas(self, ix, iy):
        """Convert image (original) pixel coords to canvas coords"""
        cx = int(ix * self.scale + self.offset_x)
        cy = int(iy * self.scale + self.offset_y)
        return cx, cy

    def canvas_to_image(self, cx, cy):
        """Convert canvas coords to original image pixel coords (float)"""
        ix = (cx - self.offset_x) / self.scale
        iy = (cy - self.offset_y) / self.scale
        return ix, iy

    def image_to_canvas_box(self, box):
        x0, y0, x1, y1 = box
        cx0, cy0 = self.image_to_canvas(x0, y0)
        cx1, cy1 = self.image_to_canvas(x1, y1)
        return cx0, cy0, cx1, cy1

    # --- Left button: ROI drawing ---
    def on_left_press(self, event):
        self.drawing = True
        self.roi_start = (event.x, event.y)
        if self.roi_rect_id:
            try:
                self.canvas.delete(self.roi_rect_id)
            except Exception:
                pass
        self.roi_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
        # ensure overlay list updated
        self._overlay_ids = [self.roi_rect_id]

    def on_left_move(self, event):
        if not self.drawing:
            return
        x0, y0 = self.roi_start
        self.canvas.coords(self.roi_rect_id, x0, y0, event.x, event.y)

    def on_left_release(self, event):
        if not self.drawing:
            return
        self.drawing = False
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        cx0, cy0 = min(x0, x1), min(y0, y1)
        cx1, cy1 = max(x0, x1), max(y0, y1)
        ix0, iy0 = self.canvas_to_image(cx0, cy0)
        ix1, iy1 = self.canvas_to_image(cx1, cy1)
        ix0 = max(0, min(self.image_pil.width, ix0))
        ix1 = max(0, min(self.image_pil.width, ix1))
        iy0 = max(0, min(self.image_pil.height, iy0))
        iy1 = max(0, min(self.image_pil.height, iy1))
        if abs(ix1-ix0) < 1 or abs(iy1-iy0) < 1:
            if self.roi_rect_id:
                try:
                    self.canvas.delete(self.roi_rect_id)
                except Exception:
                    pass
                self.roi_rect_id = None
            self.roi_start = None
            return
        self.roi_start = (cx0, cy0)
        self.last_drawn_roi = (ix0, iy0, ix1, iy1)
        # ensure overlay list updated
        self._overlay_ids = [self.roi_rect_id]

    # --- Right button: panning ---
    def on_right_press(self, event):
        # start pan
        self.pan_start = (event.x, event.y)

    def on_right_move(self, event):
        # compute delta and update offset, then redraw
        if not hasattr(self, 'pan_start') or self.pan_start is None:
            self.pan_start = (event.x, event.y)
            return
        sx, sy = self.pan_start
        dx = event.x - sx
        dy = event.y - sy
        self.offset_x += dx
        self.offset_y += dy
        self.pan_start = (event.x, event.y)
        # move image and overlays for a smooth pan without re-rendering
        try:
            if self.image_id:
                self.canvas.move(self.image_id, dx, dy)
            # move overlays
            for oid in getattr(self, '_overlay_ids', []) or []:
                try:
                    self.canvas.move(oid, dx, dy)
                except Exception:
                    pass
        except Exception:
            # fallback to full redraw
            self.display_image()

    def on_mouse_wheel(self, event):
        # zoom centered at canvas center
        if self.image_pil is None:
            return
        # Use center of canvas as zoom focus: keep image point at canvas center fixed.
        factor = 1.0 + (0.001 * event.delta)
        new_scale = max(0.05, min(10.0, self.scale * factor))
        if new_scale == self.scale:
            return
        Cx = self.canvas.winfo_width() / 2.0
        Cy = self.canvas.winfo_height() / 2.0
        # image pixel at canvas center before zoom
        ix_center = (Cx - self.offset_x) / self.scale
        iy_center = (Cy - self.offset_y) / self.scale
        # update scale
        self.scale = new_scale
        # compute new offsets so that image pixel (ix_center,iy_center) maps to canvas center again
        self.offset_x = Cx - ix_center * self.scale
        self.offset_y = Cy - iy_center * self.scale
        # clamp offsets to avoid image completely leaving canvas (keep at least one corner visible)
        img_w = self.image_pil.width * self.scale
        img_h = self.image_pil.height * self.scale
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # ensure image covers at least partially the canvas by clamping offsets
        if img_w < cw:
            # center horizontally
            self.offset_x = (cw - img_w) / 2.0
        else:
            self.offset_x = min(self.offset_x, 0)
            self.offset_x = max(self.offset_x, cw - img_w)
        if img_h < ch:
            self.offset_y = (ch - img_h) / 2.0
        else:
            self.offset_y = min(self.offset_y, 0)
            self.offset_y = max(self.offset_y, ch - img_h)

        self.display_image()

    def confirm_roi(self):
        # confirm last drawn ROI as template
        if not hasattr(self, 'last_drawn_roi') or self.last_drawn_roi is None:
            messagebox.showinfo('Info', 'No ROI drawn to confirm')
            return
        ix0, iy0, ix1, iy1 = self.last_drawn_roi
        # convert to integer pixels on original
        x0, y0, x1, y1 = int(round(ix0)), int(round(iy0)), int(round(ix1)), int(round(iy1))
        if x1 <= x0 or y1 <= y0:
            messagebox.showerror('Error', 'Invalid ROI')
            return
        self.confirmed_roi = (x0, y0, x1, y1)
        # draw confirmed ROI in lime
        if self.roi_rect_id:
            try:
                self.canvas.delete(self.roi_rect_id)
            except Exception:
                pass
        cx0, cy0, cx1, cy1 = self.image_to_canvas_box(self.confirmed_roi)
        self.roi_rect_id = self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline='lime', width=2)
        # update overlay ids
        self._overlay_ids = [self.roi_rect_id]
        messagebox.showinfo('ROI Confirmed', f'ROI confirmed: ({x0},{y0})-({x1},{y1})')

    def clear_roi(self):
        """Clear temporary and confirmed ROI selections."""
        self.last_drawn_roi = None
        self.confirmed_roi = None
        if getattr(self, 'roi_rect_id', None):
            try:
                self.canvas.delete(self.roi_rect_id)
            except Exception:
                pass
        self.roi_rect_id = None
        self._overlay_ids = []

    def run_match(self):
        if self.image is None:
            messagebox.showerror('Error', 'No image loaded')
            return
        if self.confirmed_roi is None:
            messagebox.showerror('Error', 'No confirmed ROI (template)')
            return
        try:
            thresh = float(self.thresh_var.get())
        except Exception:
            messagebox.showerror('Error', 'Invalid threshold')
            return
        # extract template from original image (RGB -> GRAY)
        x0, y0, x1, y1 = self.confirmed_roi
        template_rgb = self.image[y0:y1, x0:x1]
        if template_rgb.size == 0:
            messagebox.showerror('Error', 'Empty template')
            return
        template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        try:
            rects, df, out_path = mian_arrays(img_gray, template_gray, match_threshold=thresh)
        except Exception as e:
            messagebox.showerror('Error', f'Matching failed: {e}')
            return
        # show saved detection result image in a preview window
        try:
            preview = tk.Toplevel(self.root)
            preview.title('Matching Result')
            pilr = Image.open(out_path)
            # resize preview to reasonable size if large
            maxw, maxh = 900, 700
            rw, rh = pilr.size
            scale = min(1.0, maxw / rw, maxh / rh)
            if scale < 1.0:
                pilr = pilr.resize((int(rw*scale), int(rh*scale)), Image.LANCZOS)
            photo_r = ImageTk.PhotoImage(pilr)
            lbl = tk.Label(preview, image=photo_r)
            lbl.image = photo_r
            lbl.pack()
        except Exception:
            pass
        messagebox.showinfo('Done', f'Matching done. Result saved to {out_path}. Rows: {len(df)}')


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

