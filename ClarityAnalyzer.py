import cv2
import numpy as np



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
