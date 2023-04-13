"""
HaMiHaMiHa
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class ImgProcess:
    def __init__(self, img) -> None:
        self.src = cv2.imread(img)  # 读取图像
        self.gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)  # 将彩色图片转化为灰色图片
        self.h, self.w = self.src.shape[:2]  # h：图像的高 w：图像的宽

    """ 
    毛玻璃特效思路：
    1、对于输入图像中的每个像素，随机选择一个周围像素进行采样，并将采样结果作为输出像素的值。采样窗口的大小和采样方式可以根据需求进行调整。
    2、重复上述操作，直到对整张图像完成采样。 
    """
    def glass(self, radius=2):  # 修改radius大小，可以调整效果
        glassImg = np.zeros((self.h, self.w, 3), np.uint8)  # 创建一个和原图相同大小的0图像
        for i in range(self.h):
            for j in range(self.w):
                # 随机选择一个采样位置
                i_rand, j_rand = np.random.randint(i - radius, i + radius + 1), np.random.randint(j - radius,
                                                                                                  j + radius + 1)
                # 对采样位置进行检查，防止超出图像的范围（防止出现黑边）
                i_rand = np.clip(i_rand, 0, self.h - 1)
                j_rand = np.clip(j_rand, 0, self.w - 1)
                glassImg[i, j] = self.src[i_rand, j_rand]  # 将采样结果作为输出像素的值
        return glassImg

    """
    浮雕特效思路：
    1、遍历图像中的每一个像素点，并计算该像素点与其右侧相邻像素点的灰度值之差。这个差值可以用来表示该像素点周围区域的边缘。
    2、将计算得到的边缘值加上一个固定的值（120），以产生一种立体感。如果这个值大于255，将它设为255，如果小于0，将它设为0。
    """
    def relief(self):
        reliefImg = np.zeros((self.h, self.w, 3), np.uint8)
        for i in range(self.h):
            for j in range(self.w - 2):
                edge = int(self.gray[i, j]) - int(self.gray[i, j + 1])  # 计算相邻像素之间的差值，作为边缘的值
                val = edge + 120  # 将边缘值加上120，使边缘变得更加明显，产生立体感

                # 确保像素值在0-255之间
                if val > 255:
                    val = 255
                if val < 0:
                    val = 0
                for k in range(3):
                    reliefImg[i, j, k] = val
        return reliefImg

    """
    油画特效思路：
    1.遍历输入图像的每个像素，以该像素为中心，计算周围相邻像素的颜色分布。这里使用一个4x4的卷积核，遍历周围16个像素。
    2.通过颜色量化，将颜色值划分为8个级别。遍历4x4的卷积核中的每个像素，将其颜色值量化为8个级别，并统计每个级别出现的次数。
    3.找到颜色分布最多的那个颜色，作为该像素的颜色值。在颜色量化中，每个级别代表一个颜色区间，颜色分布最多的区间就是该像素的最终颜色区间。在该区间内，
    选择一个像素作为该像素的颜色值。这里选择该区间内颜色值最接近区间中值的像素。
    4.将计算得到的颜色值设置为该像素的颜色，并将结果存储到油画特效的结果图像中。
    """
    def oil(self):
        oilImg = np.zeros((self.h, self.w, 3), np.uint8)
        for i in range(2, self.h - 2):
            for j in range(2, self.w - 2):
                # 量化向量，用于统计每个灰度级别的像素点数量
                quant = np.zeros(8, np.uint8)
                # 对于该像素的4x4区域中的每个像素，计算其灰度级别并量化
                for k in range(-2, 2):
                    for t in range(-2, 2):
                        # 将灰度值量化为8个级别
                        level = int(self.gray[i + k, j + t] / 32)
                        # 量化计数
                        quant[level] = quant[level] + 1

                # 找到数量最多的灰度级别，并返回它的索引
                valIndex = np.argmax(quant)

                # 对于该像素周围的16个像素，将其颜色平均值设置为属于最常见灰度级别的像素颜色平均值。
                for k in range(-2, 2):
                    for t in range(-2, 2):
                        # 如果该像素点属于最常见的灰度级别，则将其颜色赋值给该像素点
                        if (valIndex * 32) <= self.gray[i + k, j + t] <= ((valIndex + 1) * 32):
                            (b, g, r) = self.src[i + k, j + t]
                            oilImg[i, j] = (b, g, r)
        return oilImg

    """
    马赛克特效思路：
    1、通过参数size指定分块大小，将图像水平和垂直方向每个size个像素点分成一个小方块。
    2、对于每个小方块内的像素点，将其颜色设置为该小方块内的平均颜色，实现马赛克效果。
    """
    def mosaic(self, size=5):  # 修改size的大小，可以调整效果
        # 获得图像的高度和宽度
        h, w = self.h, self.w
        img = self.src
        # 将图像水平和垂直方向每个size个像素点分成一个小方块
        h_blocks = h // size
        w_blocks = w // size
        # 遍历每个小方块
        for row in range(h_blocks):
            for col in range(w_blocks):
                # 获取当前小方块的区域
                roi = img[row * size: (row + 1) * size, col * size: (col + 1) * size]
                # 计算小方块内的平均颜色
                b = int(cv2.mean(roi)[0])
                g = int(cv2.mean(roi)[1])
                r = int(cv2.mean(roi)[2])
                # 将小方块内所有像素点的颜色设置为小方块内的平均颜色
                cv2.rectangle(img, (col * size, row * size), ((col + 1) * size, (row + 1) * size), (b, g, r), -1)
        return img

    """
    素描特效思路：
    输入图像转换为灰度图像，反转并模糊该灰度图像，然后使用差分得到素描图像。
    """
    def sketch(self):
        temp = 255 - self.gray  # 对灰度图像进行反转
        gauss = cv2.GaussianBlur(temp, (21, 21), 0)  # 进行高斯模糊以减小噪声
        inverGauss = 255 - gauss
        return cv2.divide(self.gray, inverGauss, scale=127.0)  # 将灰度图像和模糊图像进行差分，得到素描图象。

    """
    怀旧特效思路：
    对于图像中的每一个像素点，将其BGR三个通道的值分别与一定的权重相乘，得到三个新的值，然后将这三个值作为像素点的新的BGR值。
    这里的权重是经过调整的，目的是使得转换后的图像具有一定的怀旧风格。
    """
    def old(self):
        oldImg = np.zeros((self.h, self.w, 3), np.uint8)
        # 遍历图像的每个像素
        for i in range(self.h):
            for j in range(self.w):
                # 分别赋予三个通道像素点新的值
                b = 0.272 * self.src[i, j][2] + 0.534 * self.src[i, j][1] + 0.131 * self.src[i, j][0]
                g = 0.349 * self.src[i, j][2] + 0.686 * self.src[i, j][1] + 0.168 * self.src[i, j][0]
                r = 0.393 * self.src[i, j][2] + 0.769 * self.src[i, j][1] + 0.189 * self.src[i, j][0]
                # 确保像素值不超过255
                if b > 255:
                    b = 255
                if g > 255:
                    g = 255
                if r > 255:
                    r = 255
                oldImg[i, j] = np.uint8((b, g, r))
        return oldImg

    """
    流年特效思路：
    对于每个像素点的蓝色通道，首先将其值进行开平方处理，然后将结果乘以一个常数14，这样可以强调蓝色通道的特征，赋予图像蓝色调的色彩风格。
    绿色和红色通道则直接赋值给新图像，保持不变。
    """
    def fleet(self):
        fleetImg = np.zeros((self.h, self.w, 3), np.uint8)
        for i in range(self.h):
            for j in range(0, self.w):
                b = math.sqrt(self.src[i, j][0]) * 14  # 对每个像素点的蓝色通道进行开平方处理，让后乘以14
                g = self.src[i, j][1]  # 绿色和红色通道像素值保持不变
                r = self.src[i, j][2]
                if b > 255:
                    b = 255
                fleetImg[i, j] = np.uint8((b, g, r))
        return fleetImg

    """
    卡通特效思路：
    1.进行双边滤波，保留图像边缘信息，同时去除噪声。
    2.对双边滤波的结果进行中值滤波，进一步消除噪声和不规则的边缘。
    3.使用自适应阈值处理来提取边缘。
    4.将提取出的边缘图像转换为RGB彩色空间。
    5.使用按位与运算符将原始图像与边缘图像相乘，使卡通化的效果更突出。
    """
    def cartoon(self):
        num = 7  # 双边滤波数目，可以调整
        for i in range(num):
            cv2.bilateralFilter(self.src, d=9, sigmaColor=5, sigmaSpace=3)  # 双边滤波
        median = cv2.medianBlur(self.gray, 7)  # 中值滤波
        # 自适应阈值处理提取边缘
        edge = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=5, C=2)
        # 转化为彩色图像
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        return cv2.bitwise_and(self.src, edge)  # 通过按位与运算与原始图像进行相乘


if __name__ == '__main__':
    process_types = ['glass', 'relief', 'oil', 'mosaic', 'sketch', 'old', 'fleet', 'cartoon']  # 定义8种处理效果的列表

    # 1、依次显示每种处理图像
    for process_type in process_types:
        process = ImgProcess('2.jpg')  # 创建ImgProcess对象，可以读取指定的图像文件  需要在此处修改图片名称
        processed_img = getattr(process, process_type)()  # 字符串转化为函数
        cv2.imshow(process_type, processed_img)  # 显示图像和名称
        cv2.waitKey(delay=0)  # 按下任意键执行下一项

    # # 2、单独显示一种处理图像
    # process = ImgProcess('1.png')  # 需要在此处修改图片名称
    # process_type = process_types[0]  # 输入0-7显示不同的处理结果
    # processed_img = getattr(process, process_type)()
    # cv2.imshow(process_type, processed_img)
    # cv2.waitKey(delay=0)

    # # 3、所有图像在一张图中显示
    # plt.rcParams['font.family'] = 'SimHei'  # 设置字体的格式
    # titles = ['原图', '毛玻璃特效', '浮雕特效', '油画特效', '马赛克特效', '素描特效', '怀旧特效', '流年特效', '卡通特效']
    # for i in range(9):
    #     if i == 0:
    #         img = cv2.imread('1.png')  # 需要在此处修改图片名称
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow(img)
    #         plt.title(titles[i])
    #         plt.xticks([])
    #         plt.yticks([])
    #     else:
    #         process = ImgProcess('1.png')  # 需要在此处修改图片名称
    #         process_type = process_types[i - 1]
    #         processed_img = getattr(process, process_type)()
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow(processed_img)
    #         plt.title(titles[i])
    #         plt.xticks([])
    #         plt.yticks([])
    # plt.suptitle('图像特效处理')
    # plt.show()
