# -*- coding: utf-8 -*-
"""
@File    : Tools for Template Matching and MTF Analysis
@Author  : Cichun Li
@Date    : 2025-12-04
@Copyright: Copyright (c) 2025 Cichun Li Alphabetter inc. All rights reserved.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


class MTFCalculator:
    """
    MTF(调制传递函数)计算器类
    基于多个交替振幅的高斯分布叠加计算MTF值
    """

    def __init__(self, pixel_size=0.5, linepair=500, num_points=7000):
        """
        初始化MTF计算器
        
        参数:
            pixel_size: 像素大小，单位微米 (默认0.5)
            linepair: 线对数量 (默认500)
            sigma: 高斯分布标准差，若为None则自动计算 (默认None)
            num_points: 采样点数 (默认7000)
        """
        self.pixel_size = pixel_size
        self.linepair = linepair
        self.num_points = num_points
        
        # 计算分辨率
        self.res = 1000 / (2 * linepair)
        
        # 初始化计算结果属性
        self.mtf_value = 0
        self.IpatternMax = 0
        self.IpatternMin = 0
        self.x = None
        self.gauss_sum = None
        self.antideriv_sum = None
        self.numerical_deriv = None

    @staticmethod
    def gaussian_distribution(x, mu, sigma, amplitude=1.0):
        """
        高斯分布函数
        
        参数:
            x: 自变量数组
            mu: 均值（中心位置）
            sigma: 标准差
            amplitude: 振幅 (默认1.0)
        
        返回:
            高斯分布值数组
        """
        return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def gaussian_antiderivative(x, mu, sigma, amplitude=1.0):
        """
        单个高斯分布的不定积分原函数（基于误差函数erf）
        
        参数:
            x: 自变量数组
            mu: 均值（中心位置）
            sigma: 标准差
            amplitude: 振幅 (默认1.0)
        
        返回:
            原函数值数组
        """
        arg = (x - mu) / (sigma * np.sqrt(2))
        return (amplitude / 2) * erf(arg)

    def calculate_mtf(self, sigma):
        """
        核心计算方法：计算MTF值
        执行高斯分布叠加、积分、极值计算等逻辑
        参数:
            sigma: 高斯分布标准差，单位为um，注意单位转换(必传参数)
        """
        try:
            # 5个高斯分布的参数设置（位置、标准差、振幅）
            mu1, sigma1, amp1 = 5, sigma, 100
            mu2, sigma2, amp2 = mu1 + self.res, sigma1, -amp1
            mu3, sigma3, amp3 = mu2 + self.res, sigma1, amp1
            mu4, sigma4, amp4 = mu3 + self.res, sigma1, -amp1
            mu5, sigma5, amp5 = mu4 + self.res, sigma1, amp1

            # 生成自变量数组
            self.x = np.linspace(mu1, mu5, self.num_points)

            # 计算单个高斯分布
            gauss1 = self.gaussian_distribution(self.x, mu1, sigma1, amp1)
            gauss2 = self.gaussian_distribution(self.x, mu2, sigma2, amp2)
            gauss3 = self.gaussian_distribution(self.x, mu3, sigma3, amp3)
            gauss4 = self.gaussian_distribution(self.x, mu4, sigma4, amp4)
            gauss5 = self.gaussian_distribution(self.x, mu5, sigma5, amp5)

            # 高斯分布叠加
            self.gauss_sum = np.round(gauss1 + gauss2 + gauss3 + gauss4 + gauss5).astype(int)

            # 计算单个高斯的不定积分原函数
            antideriv1 = self.gaussian_antiderivative(self.x, mu1, sigma1, amp1)
            antideriv2 = self.gaussian_antiderivative(self.x, mu2, sigma2, amp2)
            antideriv3 = self.gaussian_antiderivative(self.x, mu3, sigma3, amp3)
            antideriv4 = self.gaussian_antiderivative(self.x, mu4, sigma4, amp4)
            antideriv5 = self.gaussian_antiderivative(self.x, mu5, sigma5, amp5)

            # 叠加后的原函数
            self.antideriv_sum = 0.5 * amp1 + antideriv1 + antideriv2 + antideriv3 + antideriv4 + antideriv5

            # 数值微分验证
            self.numerical_deriv = np.gradient(self.antideriv_sum, self.x)

            # 找到极值点并计算MTF
            extremum_indices = np.where(self.gauss_sum == 0)
            extremum_antideriv_sum = self.antideriv_sum[extremum_indices]

            if extremum_antideriv_sum.size == 0:
                print("No extremum points found in the antiderivative sum.")
                self.mtf_value = 0
                self.IpatternMax = 0
                self.IpatternMin = 0
            else:
                self.IpatternMax = np.max(extremum_antideriv_sum)
                self.IpatternMin = np.min(extremum_antideriv_sum)
                self.mtf_value = (self.IpatternMax - self.IpatternMin) / 100

            print(f"Calculated MTF Value: {self.mtf_value:.6f}")

        except Exception as e:
            print(f"An error occurred during MTF calculation: {str(e)}")
            self.mtf_value = 0
            self.IpatternMax = 0
            self.IpatternMin = 0

        return self.mtf_value

    def plot_results(self, figsize=(12, 10), dpi=100):
        """
        绘制计算结果图表
        
        参数:
            figsize: 图表尺寸 (默认(12,10))
            dpi: 分辨率 (默认100)
        """
        # 检查是否已完成计算
        if self.x is None:
            print("Please run calculate_mtf() first before plotting!")
            return

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

        # 子图1：高斯叠加分布 + 数值导数
        ax1.plot(self.x, self.gauss_sum, color='#F18F01', linewidth=3, linestyle='--', 
                 label='Sum of Gaussians', zorder=5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_ylabel('Gray Gradients', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        
        # 自动调整y轴范围
        y_min = min(min(self.gauss_sum), min(self.numerical_deriv)) * 1.2
        y_max = max(max(self.gauss_sum), max(self.numerical_deriv)) * 1.2
        ax1.set_ylim(y_min, y_max)

        # 子图2：叠加原函数 + 极值参考线
        ax2.axhline(y=self.IpatternMax, color='r', linestyle='--', 
                    label=f'Max ({self.IpatternMax:.4f})')
        ax2.axhline(y=self.IpatternMin, color='g', linestyle='--', 
                    label=f'Min ({self.IpatternMin:.4f})')
        ax2.plot(self.x, self.antideriv_sum, color='#F18F01', linewidth=3, 
                 label='Sum of Antiderivatives', zorder=5)
        ax2.set_xlabel('Pixel', fontsize=12)
        ax2.set_ylabel('Gray', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_ylim(0, 160)

        plt.tight_layout()
        plt.show()

    def get_results(self):
        """
        获取计算结果汇总
        
        返回:
            dict: 包含MTF值、最大值、最小值的字典
        """
        return {
            'mtf_value': self.mtf_value,
            'IpatternMax': self.IpatternMax,
            'IpatternMin': self.IpatternMin,
            'res': self.res
        }


# 测试代码
if __name__ == "__main__":
    # 1. 创建MTF计算器实例
    mtf_calculator = MTFCalculator(
        pixel_size=0.5559,    # 像素大小（微米）
        linepair=500,      # 线对数量
        num_points=9000,    # 采样点数
    )

    # 2. 执行MTF计算
    mtf_value = mtf_calculator.calculate_mtf(sigma=1.159)  # 传入sigma参数

    # 3. 绘制结果图表
    mtf_calculator.plot_results()

    # 4. 获取详细结果
    results = mtf_calculator.get_results()
    print("\nDetailed Results:")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    