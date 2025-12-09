import cv2
import numpy as np
import pandas as pd


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
