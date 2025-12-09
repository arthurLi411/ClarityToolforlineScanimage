import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # 从 scipy 导入误差函数（兼容所有版本）

def gaussian_distribution(x, mu, sigma, amplitude=1.0):
    """高斯分布函数（同之前）"""
    return amplitude * np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def gaussian_antiderivative(x, mu, sigma, amplitude=1.0):
    """单个高斯分布的不定积分原函数（基于误差函数erf）"""
    arg = (x - mu) / (sigma * np.sqrt(2))  # 推导得出的变量替换结果
    return (amplitude / 2) * erf(arg)  # 使用 scipy 的 erf 函数，避免版本兼容问题


def MTF_calc(sigma, res, num_points=7000, drawimg=False):

    try:
        # 4个高斯分布的参数设置（位置、标准差、振幅）
        # 保持标准差一致，位置均匀分布，振幅正负交替以形成明显的叠加效果
        
        mu1, sigma1, amp1 = 5, sigma, 100   # 第一个高斯（正振幅）
        mu2, sigma2, amp2 = mu1+res, sigma1, -amp1   # 第二个高斯（负振幅）
        mu3, sigma3, amp3 = mu2+res, sigma1, amp1   # 第三个高斯（正振幅）
        mu4, sigma4, amp4 = mu3+res, sigma1, -amp1  # 第四个高斯（负振幅）
        mu5, sigma5, amp5 = mu4+res, sigma1, amp1  # 第五个高斯（负振幅）

        x = np.linspace(mu1, mu5, num_points)
        # 单个高斯分布
        gauss1 = gaussian_distribution(x, mu1, sigma1, amp1)
        gauss2 = gaussian_distribution(x, mu2, sigma2, amp2)
        gauss3 = gaussian_distribution(x, mu3, sigma3, amp3)
        gauss4 = gaussian_distribution(x, mu4, sigma4, amp4)
        gauss5 = gaussian_distribution(x, mu5, sigma5, amp5)

        # 4个高斯分布叠加
        gauss_sum = np.round(gauss1 + gauss2 + gauss3 + gauss4 + gauss5).astype(int)

        # 单个高斯的不定积分原函数
        antideriv1 = gaussian_antiderivative(x, mu1, sigma1, amp1)
        antideriv2 = gaussian_antiderivative(x, mu2, sigma2, amp2)
        antideriv3 = gaussian_antiderivative(x, mu3, sigma3, amp3)
        antideriv4 = gaussian_antiderivative(x, mu4, sigma4, amp4)
        antideriv5 = gaussian_antiderivative(x, mu5, sigma5, amp5)

        # 叠加后的原函数（积分可加性）
        antideriv_sum = 0.5*amp1 + antideriv1 + antideriv2 + antideriv3 + antideriv4 + antideriv5

        numerical_deriv = np.gradient(antideriv_sum, x)  # 差分求导（验证正确性）

        # 计算MTF值
        # 找到antideriv_sum极值点
        extremum_indices = np.where(gauss_sum == 0)
        extremum_antideriv_sum = antideriv_sum[extremum_indices]
        # print("Extremum Antideriv Sum:", extremum_antideriv_sum)

        if extremum_antideriv_sum.size == 0:
            print("No extremum points found in the antiderivative sum.")
            mtf_value = 0
            
            IpatternMax = 0
            IpatternMin = 0

            print("MTF Value:", mtf_value)
        else:

            IpatternMax = np.max(extremum_antideriv_sum)
            IpatternMin = np.min(extremum_antideriv_sum)

            # 计算MTF值
            mtf_value = (IpatternMax - IpatternMin) / 100
            print("MTF Value:", mtf_value)
    except Exception as e:
        print("An error occurred during MTF calculation:", e)
        mtf_value = 0
        IpatternMax = 0
        IpatternMin = 0

    if drawimg:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100, sharex=True)


        # 子图1：单个高斯分布 + 叠加分布 + 数值导数（验证用）
        ax1.plot(x, gauss_sum, color='#F18F01', linewidth=3, linestyle='--', label='Sum of Gaussians', zorder=5)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_ylabel('Gray Gradients', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_ylim(min(min(gauss_sum), min(numerical_deriv)) * 1.2, max(max(gauss_sum), max(numerical_deriv)) * 1.2)


        # 子图2：单个原函数 + 叠加后原函数
        plt.axhline(y=IpatternMax, color='r', linestyle='--', label=f'Reference Line ({IpatternMax})')
        plt.axhline(y=IpatternMin, color='r', linestyle='--', label=f'Reference Line ({IpatternMin})')

        ax2.plot(x, antideriv_sum, color='#F18F01', linewidth=3, label='Sum of Antiderivatives', zorder=5)
        ax2.set_xlabel('Pixel', fontsize=12)
        ax2.set_ylabel('Gray', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.set_ylim(0, 160)

        plt.tight_layout()
        plt.show()

    return mtf_value



if __name__ == "__main__":
    pixel_size = 0.5  # 像素大小，单位微米
    sigma = 1.09*pixel_size*1  # 标准差参数
    linepair = 600
    res= 1000/(2*linepair)
    MTF_calc(sigma=sigma,res=res, drawimg=True)
