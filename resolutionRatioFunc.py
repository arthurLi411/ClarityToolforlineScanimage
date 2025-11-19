import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
import numpy as np


class AcutanceAnalyzer:
    """锐度分析器类，基于双高斯拟合实现图像锐度计算"""
    
    def __init__(self, img_path=None, img_array=None, ifdraw=False):
        """
        初始化锐度分析器
        
        参数:
            img_path (str, optional): 图像文件路径. 与img_array二选一
            img_array (np.ndarray, optional): 图像numpy数组（灰度图）. 与img_path二选一
            ifdraw (bool): 是否绘制拟合图，默认False
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
        
        self.ifdraw = ifdraw
        self.rows, self.cols = self.img_array.shape
        self.mid_col = self.rows // 2  # 中间行
        self.mid_row = self.cols // 2  # 中间列
    
    @staticmethod
    def gaussian(x, amp0, mean0, stddev0, amp1, mean1, stddev1):
        """双高斯函数（两个独立振幅）：amp0*高斯1 + amp1*高斯2"""
        g1 = np.exp(-((x - mean0) ** 2) / (2 * stddev0 ** 2))  # 第一个高斯
        g2 = np.exp(-((x - mean1) ** 2) / (2 * stddev1 ** 2))  # 第二个高斯
        return amp0 * g1 + amp1 * g2  # 独立振幅叠加（若amp1为负则实现"反向"）
    
    def _double_gaussian_fit(self, pixel_gradients):
        """
        内部方法：进行反向的双高斯拟合
        
        参数:
            pixel_gradients (np.ndarray): 像素梯度数组
        
        返回:
            np.ndarray or None: 拟合参数数组，拟合失败返回None
        """
        x_data = np.arange(len(pixel_gradients))

        # 初始猜测值
        initial_guess = [
            np.max(pixel_gradients),  # amp0：参考正向峰值
            np.argmax(pixel_gradients),  # mean0：正向峰位置
            1,  # stddev0：初始标准差
            np.min(pixel_gradients),  # amp1：参考反向峰值（可能为负）
            np.argmin(pixel_gradients),  # mean1：反向峰位置
            1   # stddev1：初始标准差
        ]

        # 拟合边界：标准差必须大于0
        bounds = (
            [-np.inf, -np.inf, 1e-3, -np.inf, -np.inf, 1e-3],  # 下限
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]     # 上限
        )

        try:
            popt, pcov = curve_fit(
                self.gaussian, 
                x_data, 
                pixel_gradients, 
                p0=initial_guess, 
                bounds=bounds,
                maxfev=10000  # 增加迭代次数
            )
            return popt
        except RuntimeError as e:
            print(f"拟合失败：{e}")
            return None
    
    def _draw_line_fit(self, line_pixels, pixel_gradients, fitted_curve, title_suffix=""):
        """
        内部方法：绘制像素值、梯度和拟合曲线
        
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
        ax1.set_ylabel('Line Pixels', color=color)  # 第一个Y轴标签
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
        计算指定行和列的锐度（半高宽均值）
        
        参数:
            col_number (int, optional): 指定行号，默认使用中间行
            row_number (int, optional): 指定列号，默认使用中间列
        
        返回:
            tuple: (列半高宽均值, 行半高宽均值)
        """
        # 使用默认的中间行/列（如果未指定）
        col_number = col_number if col_number is not None else self.mid_col
        row_number = row_number if row_number is not None else self.mid_row
        
        # ---------------------- 分析指定行 ----------------------
        line_pixels = self.img_array[col_number, :]
        pixel_gradients = np.gradient(line_pixels)
        popt = self._double_gaussian_fit(pixel_gradients)
        
        # 生成拟合曲线和计算半高宽均值
        if popt is not None:
            x_data = np.arange(len(pixel_gradients))
            fitted_curve = self.gaussian(x_data, *popt)
            col_stddev_mean = (popt[2] + popt[5]) / 2
        else:
            fitted_curve = np.zeros_like(pixel_gradients)
            col_stddev_mean = 0
            print(f"行 {col_number} 拟合失败，半高宽均值设为0")
        
        # 绘图（如果启用）
        if self.ifdraw:
            self._draw_line_fit(line_pixels, pixel_gradients, fitted_curve, 
                               title_suffix=f'(Row: {col_number})')
        
        # ---------------------- 分析指定列 ----------------------
        line_pixels = self.img_array[:, row_number]
        pixel_gradients = np.gradient(line_pixels)
        popt = self._double_gaussian_fit(pixel_gradients)
        
        # 生成拟合曲线和计算半高宽均值
        if popt is not None:
            x_data = np.arange(len(pixel_gradients))
            fitted_curve = self.gaussian(x_data, *popt)
            row_stddev_mean = (popt[2] + popt[5]) / 2
        else:
            fitted_curve = np.zeros_like(pixel_gradients)
            row_stddev_mean = 0
            print(f"列 {row_number} 拟合失败，半高宽均值设为0")
        
        # 绘图（如果启用）
        if self.ifdraw:
            self._draw_line_fit(line_pixels, pixel_gradients, fitted_curve,
                               title_suffix=f'(Column: {row_number})')
        
        return col_stddev_mean, row_stddev_mean
    
    def calculate_global_acutance(self, fit_threshold=(10, 10)):
        """
        计算全局锐度（所有行和列的平均锐度）
        
        参数:
            fit_threshold (tuple): 拟合阈值，(行阈值, 列阈值)，梯度绝对值小于阈值则跳过拟合
        
        返回:
            tuple: (行平均幅值, 行平均半高宽, 列平均幅值, 列平均半高宽)
        """
        # ---------------------- 分析所有行 ----------------------
        col_fits = []
        for col_number in range(self.rows):
            line_pixels = self.img_array[col_number, :]
            pixel_gradients = np.gradient(line_pixels)
            
            # 如果梯度全部在阈值范围内，跳过拟合
            if np.all(np.abs(pixel_gradients) <= fit_threshold[0]):
                col_fits.append((col_number, None))
                continue
            
            # 双高斯拟合
            popt = self._double_gaussian_fit(pixel_gradients)
            col_fits.append((col_number, popt))
        
        # # 计算行方向锐度统计（幅值）
        # amp0_values = [fit[1][0] for fit in col_fits if fit[1] is not None]
        # amp1_values = [fit[1][3] for fit in col_fits if fit[1] is not None]
        # mean_amp0 = np.mean(amp0_values) if amp0_values else None
        # mean_amp1 = np.mean(amp1_values) if amp1_values else None
        # col_mean_amp = 0.5 * (abs(mean_amp0) + abs(mean_amp1)) if (mean_amp0 is not None and mean_amp1 is not None) else None
        # print(f"行平均锐度值 幅值: {col_mean_amp}")
        
        # 计算行方向锐度统计（半高宽）
        stddev0_values = [fit[1][2] for fit in col_fits if fit[1] is not None]
        stddev1_values = [fit[1][5] for fit in col_fits if fit[1] is not None]
        mean_stddev0 = np.mean(stddev0_values) if stddev0_values else None
        mean_stddev1 = np.mean(stddev1_values) if stddev1_values else None
        col_mean_stddev = 0.5 * (mean_stddev0 + mean_stddev1) if (mean_stddev0 is not None and mean_stddev1 is not None) else None
        print(f"行平均锐度值 半高宽: {col_mean_stddev}")
        
        # ---------------------- 分析所有列 ----------------------
        row_fits = []
        for row_number in range(self.cols):
            line_pixels = self.img_array[:, row_number]
            pixel_gradients = np.gradient(line_pixels)
            
            # 如果梯度全部在阈值范围内，跳过拟合
            if np.all(np.abs(pixel_gradients) <= fit_threshold[1]):
                row_fits.append((row_number, None))
                continue
            
            # 双高斯拟合
            popt = self._double_gaussian_fit(pixel_gradients)
            row_fits.append((row_number, popt))
        
        # # 计算列方向锐度统计（幅值）
        # amp0_values = [fit[1][0] for fit in row_fits if fit[1] is not None]
        # amp1_values = [fit[1][3] for fit in row_fits if fit[1] is not None]
        # mean_amp0 = np.mean(amp0_values) if amp0_values else None
        # mean_amp1 = np.mean(amp1_values) if amp1_values else None
        # row_mean_amp = 0.5 * (abs(mean_amp0) + abs(mean_amp1)) if (mean_amp0 is not None and mean_amp1 is not None) else None
        # print(f"列平均锐度值 幅值: {row_mean_amp}")
        
        # 计算列方向锐度统计（半高宽）
        stddev0_values = [fit[1][2] for fit in row_fits if fit[1] is not None]
        stddev1_values = [fit[1][5] for fit in row_fits if fit[1] is not None]
        mean_stddev0 = np.mean(stddev0_values) if stddev0_values else None
        mean_stddev1 = np.mean(stddev1_values) if stddev1_values else None
        row_mean_stddev = 0.5 * (mean_stddev0 + mean_stddev1) if (mean_stddev0 is not None and mean_stddev1 is not None) else None
        print(f"列平均锐度值 半高宽: {row_mean_stddev}")
        
        return col_mean_stddev, row_mean_stddev
    
    def run_analysis(self, calculate_global=False, fit_threshold=(10, 10)):
        """
        运行完整分析流程
        
        参数:
            calculate_global (bool): 是否计算全局锐度（所有行和列），默认False
            fit_threshold (tuple): 全局拟合阈值，仅当calculate_global=True时有效
        
        返回:
            tuple: 若calculate_global=False，返回(指定行半高宽, 指定列半高宽)
                  若calculate_global=True，返回(行平均幅值, 行平均半高宽, 列平均幅值, 列平均半高宽, 指定行半高宽, 指定列半高宽)
        """
        # 计算指定行/列的锐度（默认中间行/列）
        single_col_stddev, single_row_stddev = self.calculate_single_line_acutance()
        
        if calculate_global:
            # 计算全局锐度
            global_results = self.calculate_global_acutance(fit_threshold)
            return (*global_results, single_col_stddev, single_row_stddev)
        else:
            return single_col_stddev, single_row_stddev


# ---------------------- 示例使用 ----------------------
if __name__ == "__main__":
    # 1. 方式一：通过图像路径创建分析器
    analyzer = AcutanceAnalyzer(
        img_path='AOI-3_MTG_squre2.png',
        ifdraw=True  # 启用绘图
    )
    
    # 2. 方式二：通过图像数组创建分析器（如果已有图像数组）
    # img = cv2.imread('AOI-3_MTG_squre1.png', cv2.IMREAD_GRAYSCALE)
    # analyzer = AcutanceAnalyzer(img_array=img, ifdraw=True)
    
    # # 仅计算中间行/列的锐度
    # print("=== 单行列锐度分析 ===")
    # col_stddev, row_stddev = analyzer.run_analysis(calculate_global=False)
    # print(f"指定行半高宽均值: {col_stddev}")
    # print(f"指定列半高宽均值: {row_stddev}")
    
    # 计算全局锐度（所有行和列）+ 单行列锐度
    print("\n=== 全局锐度分析 ===")
    global_results = analyzer.run_analysis(
        calculate_global=True,
        fit_threshold=(10, 10)  # (行阈值, 列阈值)
    )
    col_mean_stddev, row_mean_stddev, single_col_stddev, single_row_stddev = global_results
    print(f"指定行半高宽均值: {single_col_stddev}")
    print(f"指定列半高宽均值: {single_row_stddev}")
    print(f"行平均半高宽均值: {col_mean_stddev}")
    print(f"列平均半高宽均值: {row_mean_stddev}")
