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



# ---------------------- 1. 参数设置（可自定义）----------------------
x_range = (-5, 20)  # 扩大x轴范围以容纳4个高斯分布
num_points = 1000

# 4个高斯分布的参数设置（位置、标准差、振幅）
# 保持标准差一致，位置均匀分布，振幅正负交替以形成明显的叠加效果
p = 2.2
mu1, sigma1, amp1 = -1, 0.5, 100   # 第一个高斯（正振幅）
mu2, sigma2, amp2 = mu1+p*sigma1, sigma1, -amp1   # 第二个高斯（负振幅）
mu3, sigma3, amp3 = mu2+p*sigma1, sigma1, amp1   # 第三个高斯（正振幅）
mu4, sigma4, amp4 = mu3+p*sigma1, sigma1, -amp1  # 第四个高斯（负振幅）
mu5, sigma5, amp5 = mu4+p*sigma1, sigma1, amp1  # 第五个高斯（负振幅）
mu6, sigma6, amp6 = mu5+p*sigma1, sigma1, -amp1  # 第六个高斯（负振幅）
mu7, sigma7, amp7 = mu6+p*sigma1, sigma1, amp1  # 第七个高斯（正振幅）
mu8, sigma8, amp8 = mu7+p*sigma1, sigma1, -amp1  # 第八个高斯（负振幅）
mu9, sigma9, amp9 = mu8+p*sigma1, sigma1, amp1  # 第九个高斯（正振幅）
mu10, sigma10, amp10 = mu9+p*sigma1, sigma1, -amp1  # 第十个高斯（负振幅）
mu11, sigma11, amp11 = mu10+p*sigma1, sigma1, amp1  # 第十一个高斯（正振幅）
# mu12, sigma12, amp12 = mu11+2*p*sigma1, sigma1, -amp1  # 第十二个高斯（负振幅）



x = np.linspace(x_range[0], x_range[1], num_points)

# ---------------------- 2. 计算核心数据 ----------------------
# 单个高斯分布
gauss1 = gaussian_distribution(x, mu1, sigma1, amp1)
gauss2 = gaussian_distribution(x, mu2, sigma2, amp2)
gauss3 = gaussian_distribution(x, mu3, sigma3, amp3)
gauss4 = gaussian_distribution(x, mu4, sigma4, amp4)
gauss5 = gaussian_distribution(x, mu5, sigma5, amp5)
gauss6 = gaussian_distribution(x, mu6, sigma6, amp6)
gauss7 = gaussian_distribution(x, mu7, sigma7, amp7)
gauss8 = gaussian_distribution(x, mu8, sigma8, amp8)
gauss9 = gaussian_distribution(x, mu9, sigma9, amp9)
gauss10 = gaussian_distribution(x, mu10, sigma10, amp10)
gauss11 = gaussian_distribution(x, mu11, sigma11, amp11)
# gauss12 = gaussian_distribution(x, mu12, sigma12, amp12)

# 4个高斯分布叠加
gauss_sum = gauss1 + gauss2 + gauss3 + gauss4 + gauss5 + gauss6 + gauss7 + gauss8 + gauss9 + gauss10 + gauss11

# 单个高斯的不定积分原函数
antideriv1 = gaussian_antiderivative(x, mu1, sigma1, amp1)
antideriv2 = gaussian_antiderivative(x, mu2, sigma2, amp2)
antideriv3 = gaussian_antiderivative(x, mu3, sigma3, amp3)
antideriv4 = gaussian_antiderivative(x, mu4, sigma4, amp4)
antideriv5 = gaussian_antiderivative(x, mu5, sigma5, amp5)
antideriv6 = gaussian_antiderivative(x, mu6, sigma6, amp6)
antideriv7 = gaussian_antiderivative(x, mu7, sigma7, amp7)
antideriv8 = gaussian_antiderivative(x, mu8, sigma8, amp8)
antideriv9 = gaussian_antiderivative(x, mu9, sigma9, amp9)
antideriv10 = gaussian_antiderivative(x, mu10, sigma10, amp10)
antideriv11 = gaussian_antiderivative(x, mu11, sigma11, amp11)
# antideriv12 = gaussian_antiderivative(x, mu12, sigma12, amp12)

# 叠加后的原函数（积分可加性）
antideriv_sum = 0.5*amp1 + antideriv1 + antideriv2 + antideriv3 + antideriv4 + antideriv5 + antideriv6 + antideriv7 + antideriv8 + antideriv9 + antideriv10 + antideriv11





# ---------------------- 3. 数值验证：原函数的导数 ≈ 叠加分布 ----------------------
numerical_deriv = np.gradient(antideriv_sum, x)  # 差分求导（验证正确性）

# ---------------------- 4. 绘图展示 ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=100, sharex=True)

# 子图1：单个高斯分布 + 叠加分布 + 数值导数（验证用）
ax1.plot(x, gauss_sum, color='#F18F01', linewidth=3, linestyle='--', label='Sum of 11 Gaussians', zorder=5)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax1.set_ylabel('Gray Gradients', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim(min(min(gauss_sum), min(numerical_deriv)) * 1.2, max(max(gauss_sum), max(numerical_deriv)) * 1.2)

# 子图2：单个原函数 + 叠加后原函数
ax2.plot(x, antideriv_sum, color='#F18F01', linewidth=3, label='Sum of Antiderivatives', zorder=5)
ax2.set_xlabel('Pixel', fontsize=12)
ax2.set_ylabel('Gray', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=9)
ax2.set_ylim(0, 160)

plt.tight_layout()
plt.show()

# ---------------------- 可选：定积分计算示例 ----------------------
def definite_integral(a, b, x, antideriv):
    """用原函数计算区间 [a,b] 的定积分（F(b) - F(a)）"""
    # 找到x中最接近a和b的索引
    idx_a = np.argmin(np.abs(x - a))
    idx_b = np.argmin(np.abs(x - b))
    return antideriv[idx_b] - antideriv[idx_a]

# 计算叠加分布在 [-2, 18] 区间的定积分（扩大积分区间以覆盖4个高斯）
a, b = -2, 18
integral_result = definite_integral(a, b, x, antideriv_sum)
print(f"4个高斯分布叠加后在区间 [{a}, {b}] 的定积分为：{integral_result:.4f}")

# 输出每个高斯分布的基本信息
print("\n4个高斯分布的参数：")
print(f"Gauss1: μ={mu1}, σ={sigma1}, A={amp1}")
print(f"Gauss2: μ={mu2}, σ={sigma2}, A={amp2}")
print(f"Gauss3: μ={mu3}, σ={sigma3}, A={amp3}")
print(f"Gauss4: μ={mu4}, σ={sigma4}, A={amp4}")