# -*- coding: utf-8 -*-
"""
@File    : Tools for Template Matching and Clarity Analysis
@Author  : Cichun Li
@Date    : 2025-12-04
@Copyright: Copyright (c) 2025 Cichun Li Alphabetter inc. All rights reserved.
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
import numpy as np
from MTFcalc import MTFCalculator


class AcutanceAnalyzer:
    """锐度分析器类，基于双高斯拟合实现图像锐度计算"""
    
    def __init__(self, img_path=None, img_array=None, ifdraw=False, interpnum=3):
        """
        初始化锐度分析器
        
        参数:
            img_path (str, optional): 图像文件路径. 与img_array二选一
            img_array (np.ndarray, optional): 图像numpy数组（灰度图）. 与img_path二选一
            ifdraw (bool): 是否绘制拟合图，默认False
            interpnum (int, optional): 像素插值倍数，默认None不插值
        """
        # 加载图像
        if img_array is not None:
            self.img_array = img_array
            if len(self.img_array.shape) == 3:  # 如果是彩色图，转换为灰度图
                self.img_array = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2GRAY)
        elif img_path is not None:
            self.img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("必须提供img_path或img_array参数")
        
        # 预处理：对img_array进行像素位置线性插值
        self.interpnum = interpnum
        self.ifdraw = ifdraw
        self.rows, self.cols = self.img_array.shape
        self.mid_col = self.rows // 2  # 中间行
        self.mid_row = self.cols // 2  # 中间列
    
    @staticmethod
    def gaussian(x, amp0, mean0, stddev0, amp1, mean1, stddev1):
        """双高斯函数（两个独立振幅）：amp0*高斯1 + amp1*高斯2"""
        g1 = np.exp(-((x - mean0)**2) / (2 * stddev0**2)) / (stddev0 * np.sqrt(2 * np.pi))
        g2 = np.exp(-((x - mean1)**2) / (2 * stddev1**2)) / (stddev1 * np.sqrt(2 * np.pi))

        return amp0 * g1 + amp1 * g2  # 独立振幅叠加（若amp1为负则实现"反向"）
    
    
    def _interp(self, line_array):
        """
        对单行/单列的一维数组执行线性插值（核心插值方法）
        :param line_array: 输入的一维数组（行/列像素值，如 shape=(W,) 或 (H,)）
        :return: 插值后的一维数组，长度 = 原长度 × self.interpnum
        """
        # ------------- 前置检查（避免报错）-------------
        # 检查interpnum属性是否存在
        if not hasattr(self, 'interpnum') or self.interpnum is None:
            raise AttributeError("AcutanceAnalyzer对象未定义interpnum属性！请先初始化该属性")
        # 检查插值倍数有效性（避免无效插值）
        if self.interpnum <= 1:
            return line_array.copy()  # 无需插值，直接返回原数组
        # 检查输入是否为一维数组（行/列必须是一维）
        line_array = np.asarray(line_array).flatten()  # 展平为一维
        if len(line_array) < 2:
            return line_array.copy()  # 少于2个点无法插值，返回原数组

        # ------------- 线性插值核心逻辑 -------------
        # 1. 原始x坐标：0 ~ 原长度-1 的等间距点
        x_original = np.linspace(0, len(line_array)-1, len(line_array))
        # 2. 插值后新x坐标：等间距生成 原长度×interpnum 个点（覆盖原始范围）
        x_new = np.linspace(0, len(line_array)-1, len(line_array)*self.interpnum)
        # 3. 执行线性插值（np.interp核心用法）
        line_interp = np.interp(x_new, x_original, line_array)

        # ------------- 返回插值结果 -------------
        return line_interp

    def _double_gaussian_fit(self, pixel_gradients):
        """
        内部方法：进行反向的双高斯拟合
        
        参数:
            pixel_gradients (np.ndarray): 像素梯度数组（一维）
        
        返回:
            np.ndarray or None: 拟合参数数组 [amp0, mean0, std0, amp1, mean1, std1]，拟合失败返回None
        """
        # 确保输入是一维数组
        pixel_gradients = np.asarray(pixel_gradients).flatten()
        x_data = np.arange(len(pixel_gradients))
        n = len(pixel_gradients)

        # 基础校验：数据量不足直接返回
        if n < 6:  # 双高斯需要至少6个点（6个参数），原3个点拟合易失败
            print(f"拟合失败：数据长度{n} < 6，无法支撑双高斯拟合")
            return None
        
        # 防呆：梯度无差异（所有值相同），无需拟合
        if np.allclose(pixel_gradients, pixel_gradients[0]):
            print("拟合失败：梯度数组无差异，所有值相同")
            return None

        # 1. 提取峰值和对应索引（核心修复：用argmax/argmin直接取单个索引，避免数组）
        peak0 = float(np.max(pixel_gradients))  # 梯度最大值
        peak1 = float(np.min(pixel_gradients))  # 梯度最小值
        peak0_idx = int(np.argmax(pixel_gradients))  # 第一个最大值的索引（单个数值）
        peak1_idx = int(np.argmin(pixel_gradients))  # 第一个最小值的索引（单个数值）

        # 2. 确定梯度方向，构造初始值（修复数组比较问题）
        initial_std =2  
        if peak0_idx < peak1_idx:  # 单个数值比较，无歧义
            amp0, mean0 = peak0, peak0_idx
            amp1, mean1 = peak1, peak1_idx
        else:
            amp1, mean1 = peak0, peak0_idx
            amp0, mean0 = peak1, peak1_idx

        # 构造初始猜测值（确保类型为float，避免优化警告）
        initial_guess = [
            float(amp0), float(mean0), float(initial_std),
            float(amp1), float(mean1), float(initial_std)
        ]

        # 3. 合理的参数边界（避免拟合出无意义值）
        bounds = (
            [-np.inf, 0.0, 1e-3, -np.inf, 0.0, 1e-3],  # 下界：均值≥0，标准差≥1e-3
            [np.inf, float(n-1), float(n), np.inf, float(n-1), float(n)]  # 上界：均值≤n-1
        )

        # 4. 双高斯拟合（带双层异常捕获+稳健策略）
        try:
            # 首次拟合：默认Levenberg-Marquardt方法
            popt, pcov = curve_fit(
                self.gaussian,  # 需确保self.gaussian是双高斯函数（6个参数）
                x_data,
                pixel_gradients,
                p0=initial_guess,
                maxfev=2000
            )
            return popt
        except Exception as e1:
            print(f"首次拟合失败（LM方法）：{str(e1)[:100]}")  # 截断长错误信息
            try:
                # 回退策略：dogbox方法（适合边界约束）+ 更多迭代
                popt, pcov = curve_fit(
                    self.gaussian,
                    x_data,
                    pixel_gradients,
                    p0=initial_guess,
                    method='dogbox',
                    bounds=bounds,
                    maxfev=5000
                )
                return popt
            except Exception as e2:
                print(f"回退拟合失败（dogbox方法）：{str(e2)[:100]}")
                return None
            
    def _draw_line_fit(self, line_pixels, pixel_gradients, fitted_curve, title_suffix=""):
        """
        内部方法：绘制像素值、梯度和拟合曲线，用于测试
        
        参数:
            line_pixels (np.ndarray): 原始像素值
            pixel_gradients (np.ndarray): 像素梯度
            fitted_curve (np.ndarray): 拟合曲线
            title_suffix (str): 标题后缀，用于区分行/列
        """
        # 创建画布和第一个Y轴（ax1）
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 第一个Y轴：绘制 line_pixels
        color = 'tab:blue'
        ax1.set_xlabel('Pixel Position')  # 共享X轴标签
        ax1.set_ylabel('Line Pixels Gray', color=color)  # 第一个Y轴标签
        ax1.plot(line_pixels, color=color, label='Line Pixels')  # 绘制line_pixels
        ax1.tick_params(axis='y', labelcolor=color)  # Y轴刻度颜色

        # 第二个Y轴：绘制梯度和拟合曲线（共享X轴）
        ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
        color = 'tab:green'
        ax2.set_ylabel('Pixel Gradients', color=color)  # 第二个Y轴标签
        # 绘制pixel_gradients（带marker）
        ax2.plot(pixel_gradients, marker='.', color=color, alpha=0.7, label='Pixel Gradients')
        # 绘制拟合曲线（红色虚线）
        ax2.plot(fitted_curve, color='red', linestyle='--', label='Gaussian Fit')
        ax2.tick_params(axis='y', labelcolor=color)  # 第二个Y轴刻度颜色

        # 统一设置标题、网格和图例
        plt.title(f'Pixel Gradients Profile {title_suffix}')
        ax1.grid(alpha=0.3)  # 网格使用第一个轴的设置

        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')  # 合并图例

        plt.tight_layout()  # 自动调整布局，避免标签重叠
        plt.show()
    
    def calculate_single_line_acutance(self, col_number=None, row_number=None):
        """
        计算指定行和列的锐度（灰度梯度高斯拟合的标准差sigma作为锐度值）
        只计算一行和一列，默认使用中间行和中间列，用于快速评估，测试。
        
        参数:
            col_number (int, optional): 指定行号，默认使用中间行
            row_number (int, optional): 指定列号，默认使用中间列
        
        返回:
            tuple: (列标准差均值, 行标准差均值)
        """
        # 使用默认的中间行/列（如果未指定）
        col_number = col_number if col_number is not None else self.mid_col
        row_number = row_number if row_number is not None else self.mid_row
        
        # ---------------------- 分析指定行 ----------------------
        line_pixels = self.img_array[col_number, :]
        line_pixels = self._interp(line_array=line_pixels)
        pixel_gradients = np.gradient(line_pixels)
        popt = self._double_gaussian_fit(pixel_gradients)
        
        # 生成拟合曲线和计算均值
        if popt is not None:
            x_data = np.arange(len(pixel_gradients))
            fitted_curve = self.gaussian(x_data, *popt)
            col_stddev_mean = (popt[2] + popt[5]) / 2
        else:
            fitted_curve = np.zeros_like(pixel_gradients)
            col_stddev_mean = 0
            print(f"行 {col_number} 拟合失败，标准差均值设为0")
        
        # 绘图（如果启用）
        if self.ifdraw:
            self._draw_line_fit(line_pixels, pixel_gradients, fitted_curve, 
                               title_suffix=f'(Row: {col_number})')
        
        # ---------------------- 分析指定列 ----------------------
        line_pixels = self.img_array[:, row_number]
        line_pixels = self._interp(line_array=line_pixels)
        pixel_gradients = np.gradient(line_pixels)
        popt = self._double_gaussian_fit(pixel_gradients)
        
        # 生成拟合曲线和计算均值
        if popt is not None:
            x_data = np.arange(len(pixel_gradients))
            fitted_curve = self.gaussian(x_data, *popt)
            row_stddev_mean = (popt[2] + popt[5]) / 2
        else:
            fitted_curve = np.zeros_like(pixel_gradients)
            row_stddev_mean = 0
            print(f"列 {row_number} 拟合失败，均值设为0")
        
        # 绘图（如果启用）
        if self.ifdraw:
            self._draw_line_fit(line_pixels, pixel_gradients, fitted_curve,
                               title_suffix=f'(Column: {row_number})')
        
        return col_stddev_mean/self.interpnum, row_stddev_mean/self.interpnum  # 返回原始像素尺度的锐度值
    
    def calculate_global_acutance(self, fit_threshold=(10, 10)):
        """
        计算全局锐度（所有行和列的平均锐度，灰度梯度高斯拟合的标准差sigma作为锐度值）
        
        参数:
            fit_threshold (tuple): 拟合阈值，(行阈值, 列阈值)，梯度绝对值小于阈值则跳过拟合，
            去除噪声的影响。
        
        返回:
            tuple: (行平均标准差, 列平均标准差)
        """
        # ---------------------- 分析所有行 ----------------------
        col_fits = []
        print(f"总行数:{self.rows},开始数{int(0.3*self.rows)}, 结束数{int(0.7*self.rows)}")
        for col_number in range(int(0.3*self.rows),int(0.7*self.rows)):
            line_pixels = self.img_array[col_number, :]
            line_pixels = self._interp(line_pixels)
            pixel_gradients = np.gradient(line_pixels)
            
            # 如果梯度全部在阈值范围内，跳过拟合
            if np.all(np.abs(pixel_gradients) <= fit_threshold[0]):
                col_fits.append((col_number, None))
                continue
            
            # 双高斯拟合
            popt = self._double_gaussian_fit(pixel_gradients)
            col_fits.append((col_number, popt))
        
        
        # 计算行方向锐度统计
        stddev0_values = [fit[1][2] for fit in col_fits if fit[1] is not None]
        stddev1_values = [fit[1][5] for fit in col_fits if fit[1] is not None]
        mean_stddev0 = np.mean(stddev0_values)/self.interpnum if stddev0_values else None
        mean_stddev1 = np.mean(stddev1_values)/self.interpnum if stddev1_values else None
        valid_values = [v for v in [mean_stddev0, mean_stddev1] if v is not None]

        print('行列锐度: ',[f"{num:.4g}" for num in stddev0_values],[f"{num:.4g}" for num in stddev1_values])
        col_mean_stddev = np.mean(valid_values) if valid_values else None
        print(f"行平均锐度值 : {col_mean_stddev}")
        
        # ---------------------- 分析所有列 ----------------------
        row_fits = []
        print(f"总列数:{self.cols},开始数{int(0.3*self.cols)}, 结束数{int(0.7*self.cols)}")
        for row_number in range(int(0.3*self.cols),int(0.7*self.cols)):
            line_pixels = self.img_array[:, row_number]
            line_pixels = self._interp(line_pixels)
            pixel_gradients = np.gradient(line_pixels)
            
            # 如果梯度全部在阈值范围内，跳过拟合
            if np.all(np.abs(pixel_gradients) <= fit_threshold[1]):
                row_fits.append((row_number, None))
                continue
            
            # 双高斯拟合
            popt = self._double_gaussian_fit(pixel_gradients)
            row_fits.append((row_number, popt))
        

        # 计算列方向锐度统计
        stddev0_values = [fit[1][2] for fit in row_fits if fit[1] is not None]
        stddev1_values = [fit[1][5] for fit in row_fits if fit[1] is not None]
        mean_stddev0 = np.mean(stddev0_values)/self.interpnum if stddev0_values else None
        mean_stddev1 = np.mean(stddev1_values)/self.interpnum if stddev1_values else None

        valid_values = [v for v in [mean_stddev0, mean_stddev1] if v is not None]
        print('列列锐度: ',[f"{num:.4g}" for num in stddev0_values],[f"{num:.4g}" for num in stddev1_values])
        row_mean_stddev = np.mean(valid_values) if valid_values else None
        print(f"列平均锐度值 : {row_mean_stddev}")


        
        return col_mean_stddev, row_mean_stddev  # 返回原始像素尺度的锐度值
    
    def run_analysis(self, calculate_global=False, fit_threshold=(10, 10)):
        """
        运行完整分析流程
        
        参数:
            calculate_global (bool): 是否计算全局锐度（所有行和列），默认False
            fit_threshold (tuple): 全局拟合阈值，仅当calculate_global=True时有效
        
        返回:
            tuple: 若calculate_global=False，返回(指定行, 指定列)
                   若calculate_global=True，返回(行平均, 列平均)
        """
        # 计算指定行/列的锐度（默认中间行/列）
        single_col_stddev, single_row_stddev = self.calculate_single_line_acutance()
        
        if calculate_global:
            # 计算全局锐度
            global_results = self.calculate_global_acutance(fit_threshold)
            return global_results
        else:
            return single_col_stddev, single_row_stddev


# ---------------------- 示例使用 ----------------------
if __name__ == "__main__":
    # 1. 方式一：通过图像路径创建分析器
    analyzer = AcutanceAnalyzer(
        img_path='m1psta.png',
        ifdraw=True,  # 启用绘图
        interpnum=3
    )
    
    # 2. 方式二：通过图像数组创建分析器（如果已有图像数组）
    # img = cv2.imread('AOI-3_MTG_squre1.png', cv2.IMREAD_GRAYSCALE)
    # analyzer = AcutanceAnalyzer(img_array=img, ifdraw=True)
    
    # # 仅计算中间行/列的锐度
    # print("=== 单行列锐度分析 ===")
    # col_stddev, row_stddev = analyzer.run_analysis(calculate_global=False)
    # print(f"指定行均值: {col_stddev}")
    # print(f"指定列均值: {row_stddev}")
    
    # 计算全局锐度（所有行和列）+ 单行列锐度
    print("\n=== 全局锐度分析 ===")
    global_results = analyzer.run_analysis(
        calculate_global=True,
        fit_threshold=(2, 2)  # (行阈值, 列阈值)
    )
    col_mean_stddev, row_mean_stddev = global_results
    print(f"行平均均值: {col_mean_stddev}")
    print(f"列平均均值: {row_mean_stddev}")

    # 使用从调用者传入的衰减系数（默认1.0）
    # 理论计算值比实际测量偏大15%，用户可通过UI调整该系数以匹配实际测量
    pixel_size = 0.559  # 像素大小（微米）
    attenuation_coefficient = 1  # 衰减系数（用户可调节）
    standard_line = 300  # 标准线对数量（用户可设定）
    mtf_calculator = MTFCalculator(
        pixel_size=pixel_size,    # 像素大小（微米）
        linepair=standard_line,      # 线对数量
    )
    mtf_value_Col = mtf_calculator.calculate_mtf(sigma=pixel_size * attenuation_coefficient * col_mean_stddev)
    mtf_value_Row = mtf_calculator.calculate_mtf(sigma=pixel_size * attenuation_coefficient * row_mean_stddev)

    # 记录MTF值
    print(f"\n计算得到的MTF值 (行): {mtf_value_Col}")
    print(f"计算得到的MTF值 (列): {mtf_value_Row}")