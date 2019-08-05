# coding: utf-8
import cv2
import cv2.cv as cv
import numpy as np


class TargetDetection(object):
    '''
        Target Detection：目标检测基类，主要用于图像的预处理，以便后续检测更加精确
    '''
    def __init__(self, img):
        self.img = img

    def preProcess(self, img, object):
        '''
        Pre Process：预处理
        Arguments: 
            img：图像
            object：红球(redball)/足球(football)/黄杆(stick)
        Return:
            binImg：二值化后的图像
        '''
        if object == "redball":
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       # 转到HSV空间

            # HSV空间颜色判断，具体参见表格
            smin1, vmin1, hmax1, hmin2 = 9, 21, 39, 153         # 调用滑动条函数（sliderObjectHSV）得到理想值

            minHSV1 = np.array([0, smin1, vmin1])
            maxHSV1 = np.array([hmax1, 255, 255])
            minHSV2 = np.array([hmin2, smin1, vmin1])
            maxHSV2 = np.array([180, 255, 255])

            # 二值化处理
            binImg1 = cv2.inRange(HSVImg, minHSV1, maxHSV1)
            binImg2 = cv2.inRange(HSVImg, minHSV2, maxHSV2)
            binImg = np.maximum(binImg1, binImg2)

            # 图像滤波处理（腐蚀，膨胀，高斯）
            binImg = self.filter(binImg)

        elif object == "football":        
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       # 转到HSV空间   

            # HSV空间颜色判断，具体参见表格
            # vmin, smax, vmax = 184, 22, 255                     # 调用滑动条函数（sliderObjectHSV）得到理想值
            vmin, smax, vmax = 41, 34, 255
            # 二值化处理
            minHSV = np.array([0, 0, vmin])
            maxHSV = np.array([180, smax, vmax])
            binImg = cv2.inRange(HSVImg, minHSV, maxHSV)               

            # 图像滤波处理（腐蚀，膨胀，高斯）
            # binImg = self.filter(binImg)

        elif object == "stick":
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       # 转到HSV空间   

            # HSV空间颜色判断，具体参见表格
            hmin, hmax, smin, vmin = 26, 31, 47, 20             # 调用滑动条函数（sliderObjectHSV）得到理想值

            # 二值化处理
            minHSV = np.array([hmin, smin, vmin])
            maxHSV = np.array([hmax, 255, 255])
            binImg = cv2.inRange(HSVImg, minHSV, maxHSV)               

            # 图像滤波处理（腐蚀，膨胀，高斯）
            binImg = self.filter(binImg)

        else:
            print('''Please input "redball" or "football" or "stick" in preProcess()''')

        return binImg  

    def filter(self, img):
        '''
        图像滤波处理（腐蚀，膨胀，高斯）
        Arguments: 
            img：图像
        Return:
            resImg：处理后的图像
        '''
        kernelErosion = np.ones((3, 3), np.uint8)
        kernelDilation = np.ones((3, 3), np.uint8)
        resImg = cv2.erode(img, kernelErosion, iterations=2)
        resImg = cv2.dilate(resImg, kernelDilation, iterations=3)    
        resImg = cv2.GaussianBlur(resImg, (9, 9), 1.5)

        return resImg

    def sliderObjectHSV(self, object):
        '''
        HSV滑动条函数，为了获得理想的HSV阈值
        Arguments: 
            object：红球(redball)/足球(football)/黄杆(stick)
        '''
        if object == "redball":
            cv2.namedWindow("redball")
            # 创建滑动条
            cv2.createTrackbar("hmax1", "redball", 1, 20, self.nothing)
            cv2.createTrackbar("smin1", "redball", 30, 60, self.nothing)
            cv2.createTrackbar("vmin1", "redball", 30, 60, self.nothing)           
            cv2.createTrackbar("hmin2", "redball", 156, 175, self.nothing)

            img = self.img.copy()
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)        

            while True:
                srcImg = img.copy()
                # 获取滑动条的值
                hmax1 = cv2.getTrackbarPos("hmax1", "redball")
                smin1 = cv2.getTrackbarPos("smin1", "redball")
                vmin1 = cv2.getTrackbarPos("vmin1", "redball")
                hmin2 = cv2.getTrackbarPos("hmin2", "redball")

                # HSV空间颜色判断
                minHSV1 = np.array([0, smin1, vmin1])
                maxHSV1 = np.array([hmax1, 255, 255])

                minHSV2 = np.array([hmin2, smin1, vmin1])
                maxHSV2 = np.array([180, 255, 255])

                binImg1 = cv2.inRange(HSVImg, minHSV1, maxHSV1)
                binImg2 = cv2.inRange(HSVImg, minHSV2, maxHSV2)
                binImg = np.maximum(binImg1, binImg2)

                # 图像滤波处理
                binImg = self.filter(binImg)

                cv2.imshow("srcImg", img)
                cv2.imshow("redball", binImg)
                cv2.waitKey(1)
            cv2.destroyAllWindows() 

        elif object == "football":
            cv2.namedWindow("football")
            # 创建滑动条
            cv2.createTrackbar("vmin", "football", 20, 50, self.nothing)
            cv2.createTrackbar("smax", "football", 20, 250, self.nothing)
            cv2.createTrackbar("vmax", "football", 200, 255, self.nothing)

            img = self.img.copy()
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  

            while True:
                # 获取滑动条的值
                vmin = cv2.getTrackbarPos("vmin", "football")
                smax = cv2.getTrackbarPos("smax", "football")
                vmax = cv2.getTrackbarPos("vmax", "football")

                # HSV空间颜色判断
                minHSV = np.array([0, 0, vmin])
                maxHSV = np.array([180, smax, vmax])

                binImg = cv2.inRange(HSVImg, minHSV, maxHSV)

                # 图像滤波处理
                binImg = self.filter(binImg)

                cv2.imshow("srcImg", img)
                cv2.imshow("football", binImg)
                cv2.waitKey(1)
            cv2.destroyAllWindows() 

        elif object == "stick":
            cv2.namedWindow("stick")
            # 创建滑动条
            cv2.createTrackbar("hmin", "stick", 10, 30, self.nothing)
            cv2.createTrackbar("hmax", "stick", 31, 50, self.nothing)
            cv2.createTrackbar("smin", "stick", 20, 60, self.nothing)
            cv2.createTrackbar("vmin", "stick", 20, 60, self.nothing)

            img = self.img.copy()
            HSVImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  

            while True:
                # 获取滑动条的值
                hmin = cv2.getTrackbarPos("hmin", "stick")
                hmax = cv2.getTrackbarPos("hmax", "stick")
                smin = cv2.getTrackbarPos("smin", "stick")
                vmin = cv2.getTrackbarPos("vmin", "stick")

                # HSV空间颜色判断
                minHSV = np.array([hmin, smin, vmin])
                maxHSV = np.array([hmax, 255, 255])

                binImg = cv2.inRange(HSVImg, minHSV, maxHSV)

                # 图像滤波处理
                binImg = self.filter(binImg)

                cv2.imshow("srcImg", img)
                cv2.imshow("stick", binImg)
                cv2.waitKey(1)
            cv2.destroyAllWindows() 

        else:
            print('''Please input "redball" or "football" or "stick" in sliderObjectHSV()''')

    def nothing(self, x):
        pass


class HoughDetection(TargetDetection):
    '''
    Hough Detection：霍夫圆检测
    '''
    def __init__(self, img):
        super(HoughDetection, self).__init__(img)

    def houghDetection(self, img, minDist=100, minRadius=25, maxRadius=80, isShow=False):
        '''
        霍夫圆检测
        Arguments: 
            img：图像
            minDist：两圆之间最小间距
            minRadius：圆的最小半径
            maxRadius：圆的最大半径
            isShow：是否显示结果
        Return:
            circles：检测出来的圆
        '''
        SrcImg = self.img.copy()
        circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT, 1, minDist, 
                                   param1=100, param2=20, minRadius=minRadius, maxRadius=maxRadius)    
        if circles is None:
            circles = []
            print("no circle")

        else:
            circles = circles[0, ]
            if isShow is True:
                self.showHoughResult(SrcImg, circles)

        return circles

    def circle2Rect(self, circle, k=1):
        '''
        圆的信息转换为矩阵信息，以便后续处理
        Arguments: 
            circle：圆的信息：圆心坐标，半径
            k：放缩因子
        Return:
            rect：矩阵信息：左上角和右下角的坐标
        '''
        rect = []
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        initX, initY = x - k * r, y - k * r
        endX, endY = x + k * r, y + k * r
        rect = [initX, initY, endX, endY]

        return rect

    def showHoughResult(self, img, circles, timeMs=0):
        '''
        显示霍夫圆检测结果
        Arguments: 
            img：图像
            circles：圆
            timeMs：延迟时间，0表示一直显示
        '''
        for circle in circles:
            rect = self.circle2Rect(circle)
            initX, initY = rect[0], rect[1]
            endX, endY = rect[2], rect[3]
            cv2.rectangle(img, (initX, initY), (endX, endY), (0, 0, 255), 2)    # 画矩形

            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            cv2.circle(img, (x, y), r, (0, 0, 255), 2)                          # 画圆

        cv2.imshow("Hough Result", img)
        cv2.waitKey(timeMs)
        cv2.destroyAllWindows()

    def houghSlider(self, object):
        '''
        霍夫圆检测滑动条，为了获得minDist，minRadius和maxRadius理想值
        Arguments: 
            object：红球(redball)/足球(football)
        '''
        cv2.namedWindow("result")
        cv2.createTrackbar("minDist", "result", 50, 100, self.nothing)
        cv2.createTrackbar("minRadius", "result", 1, 50, self.nothing)
        cv2.createTrackbar("maxRadius", "result", 51, 100, self.nothing)
        img = self.img.copy()
        preImg = self.preProcess(img, object)        # 预处理

        while True:               
            srcImg = img.copy()          
            minD = cv2.getTrackbarPos("minDist", "result")
            minR = cv2.getTrackbarPos("minRadius", "result")
            maxR = cv2.getTrackbarPos("maxRadius", "result")

            circles = self.houghDetection(preImg, minD, minR, maxR)
            for circle in circles:
                rect = self.circle2Rect(circle)
                initX, initY = rect[0], rect[1]
                endX, endY = rect[2], rect[3]
                cv2.rectangle(srcImg, (initX, initY), (endX, endY), (0, 0, 255), 2) 

                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                cv2.circle(srcImg, (x, y), r, (0, 0, 255), 2)

            cv2.imshow("Hough Slider Result", srcImg)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    def nothing(self, x):
        '''
        滑动条需要的函数
        '''
        pass


class ContoursDetection(TargetDetection):
    def __init__(self, img):
        super(ContoursDetection, self).__init__(img)

    def contoursDetection(self, img, minPerimeter=300, minK=2, isShow=False):
        '''
        轮廓检测
        Arguments: 
            img：图像
            minPerimeter：轮廓最小周长
            isShow：是否显示结果
        Return:
            resultContours：检测出来的轮廓
        '''
        SrcImg = self.img.copy()
        rects = []
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        resultContours = []
        # 简单的轮廓周长及长宽比判断
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            k = h / w
            perimeter = cv2.arcLength(contour, True)
            if perimeter > minPerimeter and k >= minK:
                resultContours.append(contour)

        if resultContours == []:
            print("no contours")
        else:
            if isShow is True:
                self.showContourResult(SrcImg, resultContours)

        return resultContours

    def showContourResult(self, img, contours, timeMs=0):
        '''
        显示轮廓检测结果
        Arguments: 
            img：图像
            contours：轮廓
            timeMs：延迟时间，0表示一直显示
        '''
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        # 画出轮廓的外接矩阵
        for contour in contours:
            rect = self.contour2Rect(contour)     
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        cv2.imshow("Contour_result", img)
        cv2.waitKey(timeMs)
        cv2.destroyAllWindows()

    def contour2Rect(self, contour):
        '''
        轮廓的信息转换为矩阵信息，以便后续处理
        Arguments: 
            contour：轮廓的信息：若干个点组成的轮廓
        Return:
            rect：矩阵信息：左上角和右下角的坐标
        '''
        rect = []
        x, y, w, h = cv2.boundingRect(contour)      # 返回值为外接矩阵的顶点坐标和长宽

        rect = [x, y, x + w, y + h]
        return rect

    def contoursSlider(self, object):
        '''
        轮廓检测滑动条，为了获得minPer理想值
        Arguments: 
            object：黄杆(stick)
        '''
        cv2.namedWindow("result")
        cv2.createTrackbar("minPer", "result", 200, 500, self.nothing)
        img = self.img.copy()
        preImg = self.preProcess(img, object) 
        binImg = cv2.Canny(preImg, 200, 150)

        while True:               
            SrcImg = img.copy()          
            minPer = cv2.getTrackbarPos("minPer", "result")

            resultContours = self.contoursDetection(preImg, minPer)
            cv2.drawContours(SrcImg, resultContours, -1, (0, 0, 255), 2)
            for resultContour in resultContours:
                rect = self.contour2Rect(resultContour)
                cv2.rectangle(SrcImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

            cv2.imshow("result", binImg)
            cv2.imshow("Contours_result", SrcImg)
            cv2.waitKey(1)  
        cv2.destroyAllWindows()

    def nothing(self, x):
        pass


if __name__ == '__main__':
    Img = cv2.imread("stick.jpg")   # 注意替换照片

    # 测试HSV滑动条函数
    tarDet = TargetDetection(Img)
    tarDet.sliderObjectHSV("stick")

    # 测试霍夫圆检测滑动条(球类目标专用)
    # houghDet = HoughDetection(Img)
    # houghDet.houghSlider("redball")

    # 测试轮廓检测滑动条(黄杆专用)
    # ContoursDet = ContoursDetection(Img)
    # ContoursDet.contoursSlider("stick")
