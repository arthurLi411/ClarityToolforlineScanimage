# -*- coding: utf-8 -*-
"""
@File    : Tools for Template Matching and Clarity Analysis GUI
@Author  : Cichun Li
@Date    : 2025-11-06
@Copyright (c) 2025 Cichun Li, Alphabetter Co., Ltd. All rights reserved.
"""


import cv2
from scipy.signal import detrend
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

from resolutionRatioFunc import AcutanceAnalyzer  
from MTFcalc import MTFCalculator
from TempMatcher import TemplateMatcher
from ClarityAnalyzer import ClarityAnalyzer



class ImageApp:
    """Tkinter UI: open image, smooth pan/zoom, draw ROI (template), set threshold and run matching."""
    def __init__(self, root):
        self.root = root
        self.root.title("Template Matcher UI")
        # default watermark text (module docstring). Can be overridden externally.
        self.watermark_text = (__doc__ or '').strip()

        # Top controls
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, side=tk.TOP)

        open_btn = tk.Button(ctrl, text="Open Image", command=self.open_image)
        open_btn.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(ctrl, text="Match Threshold:").pack(side=tk.LEFT, padx=(8,2))
        self.thresh_var = tk.DoubleVar(value=0.93)
        self.thresh_entry = tk.Entry(ctrl, textvariable=self.thresh_var, width=6)
        self.thresh_entry.pack(side=tk.LEFT)

        tk.Label(ctrl, text="Pixel Size (um):").pack(side=tk.LEFT, padx=(8,2))
        self.pixel_size_var = tk.DoubleVar(value=0.4994)
        self.pixel_size_entry = tk.Entry(ctrl, textvariable=self.pixel_size_var, width=8)
        self.pixel_size_entry.pack(side=tk.LEFT)

        tk.Label(ctrl, text="Reference Line (lp/mm):").pack(side=tk.LEFT, padx=(8,2))
        self.standard_line_var = tk.DoubleVar(value=500)
        self.standard_line_entry = tk.Entry(ctrl, textvariable=self.standard_line_var, width=6)
        self.standard_line_entry.pack(side=tk.LEFT)

        tk.Label(ctrl, text="Attenuation Coef:").pack(side=tk.LEFT, padx=(8,2))
        self.attenuation_var = tk.DoubleVar(value=1.0)
        self.attenuation_entry = tk.Entry(ctrl, textvariable=self.attenuation_var, width=6)
        self.attenuation_entry.pack(side=tk.LEFT)

        # Global fit options
        self.calculate_global_var = tk.BooleanVar(value=True)
        self.calculate_global_chk = tk.Checkbutton(ctrl, text='Calculate Global', variable=self.calculate_global_var)
        self.calculate_global_chk.pack(side=tk.LEFT, padx=(8,2))

        tk.Label(ctrl, text="Fit Threshold (row,col):").pack(side=tk.LEFT, padx=(8,2))
        self.fit_row_var = tk.IntVar(value=2)
        self.fit_col_var = tk.IntVar(value=2)
        self.fit_row_entry = tk.Entry(ctrl, textvariable=self.fit_row_var, width=3)
        self.fit_row_entry.pack(side=tk.LEFT)
        tk.Label(ctrl, text=",", pady=0).pack(side=tk.LEFT)
        self.fit_col_entry = tk.Entry(ctrl, textvariable=self.fit_col_var, width=3)
        self.fit_col_entry.pack(side=tk.LEFT)

        run_btn = tk.Button(ctrl, text="Run Match", command=self.run_match)
        run_btn.pack(side=tk.RIGHT, padx=4)

        confirm_btn = tk.Button(ctrl, text="Confirm ROI", command=self.confirm_roi)
        confirm_btn.pack(side=tk.RIGHT, padx=4)

        clear_btn = tk.Button(ctrl, text="Clear ROI", command=self.clear_roi)
        clear_btn.pack(side=tk.RIGHT, padx=4)

        # Canvas
        self.canvas = tk.Canvas(root, bg='#333333', width=900, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Recreate watermark when canvas size changes
        self.canvas.bind('<Configure>', self._on_canvas_configure)

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
        # Ensure background watermark canvas items exist (drawn behind image)
        try:
            self._ensure_watermark()
        except Exception:
            pass

        # resize for current scale using the original PIL image (no compositing)
        w = max(1, int(self.image_pil.width * self.scale))
        h = max(1, int(self.image_pil.height * self.scale))
        resized = self.image_pil.resize((w, h), resample=Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        # redraw everything (image + overlays). We recreate overlays after deleting.
        # Keep last_drawn_roi / confirmed_roi to re-create rectangles so ROI persists across redraws.
        last_temp = getattr(self, 'last_drawn_roi', None)
        confirmed = getattr(self, 'confirmed_roi', None)

        # delete only image and overlays (keep watermark background items)
        try:
            if getattr(self, 'image_id', None):
                self.canvas.delete(self.image_id)
        except Exception:
            pass
        for oid in list(getattr(self, '_overlay_ids', []) or []):
            try:
                self.canvas.delete(oid)
            except Exception:
                pass
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

    def _ensure_watermark(self):
        # draw watermark items once; if already present, do nothing
        if self.canvas.find_withtag('wm_bg'):
            return
        self._draw_watermark()

    def _draw_watermark(self):
        try:
            text = (self.watermark_text or '').strip()
            if not text:
                return

            # compress to a short single-line if multiline docstring
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            wm_text = ' | '.join(lines[:3]) if lines else ''

            cw = max(1, self.canvas.winfo_width())
            ch = max(1, self.canvas.winfo_height())

            font_size = max(10, int(min(cw, ch) / 40))

            # create several text items on canvas (they will sit behind the image)
            positions = [
                (cw // 2, ch // 2),
                (cw // 6, ch // 6),
                (5 * cw // 6, 5 * ch // 6),
                (cw // 6, 5 * ch // 6),
                (5 * cw // 6, ch // 6),
            ]

            # choose a subtle color (no alpha in Tk canvas), let image cover it
            fill = '#dddddd'

            for (x, y) in positions:
                tid = self.canvas.create_text(x, y, text=wm_text, fill=fill, font=(None, font_size), tags='wm_bg')
                # send watermark to the bottom
                try:
                    self.canvas.tag_lower(tid)
                except Exception:
                    pass
        except Exception:
            return

    def _on_canvas_configure(self, event):
        # Recreate watermark when canvas size changes
        try:
            # remove existing watermark items
            for tid in list(self.canvas.find_withtag('wm_bg')):
                try:
                    self.canvas.delete(tid)
                except Exception:
                    pass
            self._draw_watermark()
        except Exception:
            pass

    def _apply_watermark(self, pil_image):
        """在 PIL Image 上添加半透明水印文本并返回新的 PIL Image。
        水印文本从 self.watermark_text 获取；如果为空则直接返回原图。
        水印位置为中心和四个角的附近，透明度低，不会遮挡主体内容。
        """
        try:
            text = (self.watermark_text or "").strip()
            if not text:
                return pil_image

            # 确保是 RGBA
            base = pil_image.convert("RGBA")
            w, h = base.size

            overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)

            # 字体大小随图像尺寸缩放
            fontsize = max(12, int(min(w, h) / 18))
            try:
                font = ImageFont.truetype("arial.ttf", fontsize)
            except Exception:
                font = ImageFont.load_default()

            # 文本样式
            fill = (255, 255, 255, 70)  # 白色半透明
            shadow = (0, 0, 0, 40)

            # 在图像上绘制多处水印（中心和四角近旁），使用锚点居中绘制
            positions = [
                (w // 2, h // 2),
                (w // 6, h // 6),
                (5 * w // 6, 5 * h // 6),
                (w // 6, 5 * h // 6),
                (5 * w // 6, h // 6),
            ]

            for (x, y) in positions:
                # shadow
                try:
                    draw.text((x+1, y+1), text, font=font, fill=shadow, anchor="mm")
                    draw.text((x, y), text, font=font, fill=fill, anchor="mm")
                except TypeError:
                    # older PIL may not support anchor; fallback to simple offset
                    draw.text((x+1, y+1), text, font=font, fill=shadow)
                    draw.text((x, y), text, font=font, fill=fill)

            # 叠加并返回 RGB 图像
            combined = Image.alpha_composite(base, overlay).convert("RGB")
            return combined
        except Exception:
            return pil_image

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

        # Defensive: image_pil may be None if image hasn't finished loading or was cleared.
        # Fall back to using numpy image (`self.image`) shape when available; if neither is
        # available, cancel the ROI creation gracefully.
        if getattr(self, 'image_pil', None) is not None:
            img_w, img_h = self.image_pil.width, self.image_pil.height
        elif getattr(self, 'image', None) is not None:
            # self.image is an RGB numpy array with shape (H, W, ...)
            img_h, img_w = self.image.shape[0], self.image.shape[1]
        else:
            # No image available; abort ROI
            if self.roi_rect_id:
                try:
                    self.canvas.delete(self.roi_rect_id)
                except Exception:
                    pass
            self.roi_rect_id = None
            self.roi_start = None
            self.last_drawn_roi = None
            return

        ix0 = max(0, min(img_w, ix0))
        ix1 = max(0, min(img_w, ix1))
        iy0 = max(0, min(img_h, iy0))
        iy1 = max(0, min(img_h, iy1))
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
        # read pixel size and reference line from UI
        try:
            pixel_size = float(self.pixel_size_var.get())
            if pixel_size <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror('Error', 'Invalid pixel size (must be > 0)')
            return

        try:
            standard_line = float(self.standard_line_var.get())
        except Exception:
            messagebox.showerror('Error', 'Invalid reference line value')
            return
        # read attenuation coefficient from UI
        try:
            attenuation = float(self.attenuation_var.get())
            if attenuation <= 0:
                raise ValueError()
        except Exception:
            messagebox.showerror('Error', 'Invalid attenuation coefficient (must be > 0)')
            return
        # read global calculation flag and fit thresholds
        try:
            calculate_global = bool(self.calculate_global_var.get())
            fit_row = int(self.fit_row_var.get())
            fit_col = int(self.fit_col_var.get())
            if fit_row < 0 or fit_col < 0:
                raise ValueError()
            fit_threshold = (fit_row, fit_col)
        except Exception:
            messagebox.showerror('Error', 'Invalid fit threshold values (must be integers >= 0)')
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
            rects, df, out_path = mian_arrays(
                img_gray,
                template_gray,
                match_threshold=thresh,
                pixel_size=pixel_size,
                standard_line=standard_line,
                attenuation_coefficient=attenuation,
                calculate_global=calculate_global,
                fit_threshold=fit_threshold,
            )
        except Exception as e:
            messagebox.showerror('Error', f'Matching failed: {e}')
            return
        # Note: preview window removed. The detection result image is still saved to disk
        # at `out_path` and can be opened manually by the user if desired.
        messagebox.showinfo('Done', f'Matching done. Result saved to {out_path}. Rows: {len(df)}')


def mian_arrays(img, template, match_threshold=0.93, pixel_size=0.4994, standard_line=500, attenuation_coefficient=1.0, calculate_global=True, fit_threshold=(2, 2)):
    # new params calcualte_global and fit_threshold are provided via function args in UI caller
    """
    主函数：执行模板匹配和清晰度分析，并绘制结果图表
    :param img: 输入图像（numpy数组）
    :param template: 模板图像（numpy数组）
    :param match_threshold: 模板匹配阈值
    :param pixel_size: 像素大小（微米）
    :param standard_line: 参考线对数量（lp/mm）
    :param attenuation_coefficient: 衰减系数
    :return: 匹配矩形框数组，包含分析结果的DataFrame，结果图像保存路径
    """
    if img is None or template is None:
        raise ValueError('img and template must be provided')
    ## TODO 1：模板匹配
    matcher = TemplateMatcher(threshold=match_threshold, iou_threshold=0.2)
    rects, df = matcher.match(img, template)
    out_path = 'detection_result_ui.jpg'
    matcher.draw_rectangles(img, rects, output_path=out_path)

    Clarity_values = []
    matched_region_gray_values = []

    Col_line_pairs = []
    Row_line_pairs = []
    

    for _, row in df.iterrows(): # 遍历DataFrame每一行，取出匹配区域进行清晰度分析
        # 提取匹配区域坐标和尺寸
        x = int(max(0, np.floor(row['x'])))
        y = int(max(0, np.floor(row['y'])))
        w = int(max(0, np.ceil(row['w'])))
        h = int(max(0, np.ceil(row['h'])))

        # 确保区域在图像范围内
        x1 = min(x + w, img.shape[1])
        y1 = min(y + h, img.shape[0])
        x0 = max(0, x)
        y0 = max(0, y)

        if x0 >= x1 or y0 >= y1:
            # 无效区域，跳过
            Clarity_values.append(np.nan)
            continue

        # 提取匹配区域
        matched_region = img[y0:y1, x0:x1]
        matched_region_gray_value = cv2.mean(matched_region)[0]

        if matched_region.size == 0:
            # 空区域，跳过
            Clarity_values.append(np.nan)
            continue

        ## TODO 2：清晰度分析
        analyzer = ClarityAnalyzer()
        tenengrad_value = analyzer.calculate_clarity(matched_region, method="Tenengrad")
        Clarity_values.append(tenengrad_value)
        matched_region_gray_values.append(matched_region_gray_value)

        ## TODO 3：全局锐度分析
        analyzer = AcutanceAnalyzer(img_array=matched_region)
        print("\n=== 全局锐度分析 ===")
        # respect caller-provided calculate_global and fit_threshold (defaults handled by caller)
        global_results = analyzer.run_analysis(
            calculate_global=calculate_global,
            fit_threshold=fit_threshold
        )
        col_mean_stddev, row_mean_stddev = global_results

        if col_mean_stddev is None or row_mean_stddev is None:
            # 无效数据，跳过
            Col_line_pairs.append(np.nan)
            Row_line_pairs.append(np.nan)
            continue
        
        ## TODO 4：MTF计算
        # 使用从调用者传入的衰减系数（默认1.0）
        # 理论计算值比实际测量偏大15%，用户可通过UI调整该系数以匹配实际测量
        mtf_calculator = MTFCalculator(
            pixel_size=pixel_size,    # 像素大小（微米）
            linepair=standard_line     # 线对数量
        )
        mtf_value_Col = mtf_calculator.calculate_mtf(sigma=pixel_size * attenuation_coefficient * col_mean_stddev)
        mtf_value_Row = mtf_calculator.calculate_mtf(sigma=pixel_size * attenuation_coefficient * row_mean_stddev)

        # 记录MTF值
        Col_line_pairs.append(mtf_value_Col)
        Row_line_pairs.append(mtf_value_Row)
    
    # 绘制MTF分析图表
    x = df['center_x'].values    
    plt.figure()
    plt.plot(x, Col_line_pairs, marker='.', label='Col line pairs')
    plt.plot(x, Row_line_pairs, marker='.', label='Row line pairs')

    # 画参考线（值由调用者/用户指定）
    plt.axhline(y=0.15, color='r', linestyle='--', label=f'Reference Line ({standard_line} lp/mm)')

    plt.legend(['Col line pairs MTF', 'Row line pairs MTF'])
    plt.title('Resolution Analysis')
    plt.xlabel('Camera pixel')
    plt.ylabel('MTF')
    plt.grid(True)
    plt.show()

    # plot_path = 'resolution_analysis_ui.jpg'
    # plt.savefig(plot_path)

    # # 用win打开图片查看
    # import os
    # os.startfile(plot_path)

    # If df is empty this will add no column; otherwise attach clarity values
    if not df.empty:
        df['Tenengrad_Clarity'] = Clarity_values
        df['Matched_Region_Gray_Value'] = matched_region_gray_values

    # 绘制清晰度分析图表
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
        plt.show()

        # plot_path = 'clarity_analysis_ui.jpg'
        # plt.savefig(plot_path)

        # # 用win打开图片查看
        # import os
        # os.startfile(plot_path)


    return rects, df, out_path


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

