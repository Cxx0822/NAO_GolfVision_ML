# coding uft-8
import xml.etree.ElementTree as ET
from TargetFeature import HogFeature, ColorFeature
from TargetDetection import HoughDetection, ContoursDetection
from Classifier import Logistic, KNN

import numpy as np
import cv2
import math
import os


def parseXml(xmlFile):
    """
    解析xml标注文件
    Arguments: 
        xmlFile：文件名字
    Return:
        labels：标签信息，包括矩形信息和类别信息
    """
    classes_num = {"ball": 1, "noball": 0}
    labels = []
    tree = ET.parse(xmlFile)
    root = tree.getroot()

    for item in root:
        if item.tag == "object":
            obj_name = item[0].text  
            obj_num = classes_num[obj_name]
            xmin = int(item[4][0].text)
            ymin = int(item[4][1].text)
            xmax = int(item[4][2].text)
            ymax = int(item[4][3].text)
            labels.append([xmin, ymin, xmax, ymax, obj_num])

    return labels


def reshapeBallRect(rawRect, numbers):
    """
    将球类矩形框分为若干等分
    Arguments: 
        rawRect：原来的大矩形框
        numbers：等分数量
    Return:
        newRect：新的若干个小矩形框
    """
    newRect = np.zeros((numbers, 4))
    initX, initY, endX, endY = rawRect[0], rawRect[1], rawRect[2], rawRect[3]    # 初始化参数

    newRect[0] = [initX, initY, initX + (endX - initX) / 2, initY + (endY - initY) / 2]
    newRect[1] = [initX + (endX - initX) / 2, initY, endX, initY + (endY - initY) / 2]
    newRect[2] = [initX, initY + (endY - initY) / 2, initX + (endX - initX) / 2, endY]
    newRect[3] = [initX + (endX - initX) / 2, initY + (endY - initY) / 2, endX, endY]

    return newRect


def reshapeStickRect(rawRect, numbers):
    """
    将黄杆类矩形框分为若干等分
    Arguments: 
        rawRect：原来的大矩形框
        numbers：等分数量
    Return:
        newRect：新的若干个小矩形框
    """
    newRect = np.zeros((numbers, 4))
    initX, initY, endX, endY = rawRect[0], rawRect[1], rawRect[2], rawRect[3]    # 初始化参数

    # 找出每个小矩阵的顶点坐标
    for i in range(numbers):
        newRect[i][0] = initX
        newRect[i][1] = initY + ((endY - initY) / numbers) * i
        newRect[i][2] = endX
        newRect[i][3] = initY + ((endY - initY) / numbers) * (i + 1)

    return newRect


def circle2Rect(circle, k=1):
    """
    将圆的信息转换为矩形框信息
    Arguments: 
        circle：圆心及半径
        k：放缩因子
    Return:
        矩形框信息
    """
    centerX, centerY, radius = circle[0], circle[1], circle[2]
    initX, initY = int(centerX - int(k * radius)), int(centerY - int(k * radius))
    endX, endY = int(centerX + int(k * radius)), int(centerY + int(k * radius))

    return [initX, initY, endX, endY]


def calColorFeature(img, number=16):
    """
    计算颜色特征
    Arguments: 
        img：输入图片
        number：特征向量维度
    Return:
        颜色特征信息
    """
    color = ColorFeature(img, number)
    result = color.colorExtract(img)

    return np.round(result, 4)


def calHOGFeature(img, cellSize):
    """
    计算HOG特征
    Arguments: 
        img：输入图片
        cellSize：元胞大小
    Return:
        HOG特征信息
    """
    rectBallArea = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    hog = HogFeature(rectBallArea, cellSize)        
    vector, image = hog.hog_extract() 

    return np.round(vector[0], 4)


def calPosVector(writeFilename):
    """
    计算正样本特征向量
    Arguments: 
        writeFilename：保存的特征向量txt文件名
    """   
    xmlPath = "./img_train_pos"
    labelNumbers = len(os.listdir(xmlPath))
    with open(writeFilename, 'w') as f:
        for i in range(labelNumbers):
            print('test ' + str(i))
            resultTotal = []
            xmlFile = "./label_train_pos/" + str(i) + ".xml"
            labels = parseXml(xmlFile)
            srcImg = cv2.imread("./img_train_pos/" + str(i) + ".jpg") 

            initX, initY, endX, endY = labels[0][0], labels[0][1], labels[0][2], labels[0][3]
            Rect = [initX, initY, endX, endY]    
            newRects = reshapeBallRect(Rect, 4)     # 注意更改函数

            for newRect in newRects:
                newInitX, newInitY = int(newRect[0]), int(newRect[1])
                newEndX, newEndY = int(newRect[2]), int(newRect[3])
                rectBallArea = srcImg[newInitY:newEndY, newInitX:newEndX, :]   # 矩形区域(宽，高，通道)                                 

                resultColor = calColorFeature(rectBallArea, 16)  
                cellSize = min(newEndX - newInitX, newEndY - newInitY)
                resultHOG = calHOGFeature(rectBallArea, cellSize / 2)              
                resultTotal.extend(resultColor)
                resultTotal.extend(resultHOG)   

                # cv2.rectangle(srcImg, (newInitX, newInitY), (newEndX, newEndY), (0, 0, 255), 2)  # 画矩形 

            # cv2.imshow("test " + str(i), srcImg)          
            # cv2.waitKey(300) 
            # cv2.destroyAllWindows()
            print('resultTotal', len(resultTotal)) 

            row = ' '.join(list(map(str, resultTotal))) + ' ' + str(labels[0][4]) + '\n'
            f.write(row)


def calNegVector(writeFilename):
    """
    计算负样本特征向量
    Arguments: 
        writeFilename：保存的特征向量txt文件名
    """   
    xmlPath = "./img_train_neg"
    labelNumbers = len(os.listdir(xmlPath))
    with open(writeFilename, 'w') as f:
        for i in range(labelNumbers):
            print('test ' + str(i))         
            srcImg = cv2.imread("./img_train_neg/" + str(i) + ".jpg") 
            hogDec = HoughDetection(srcImg)
            preImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY) 
            # preImg = hogDec.preProcess(srcImg, "football")
            circles = hogDec.houghDetection(preImg, minDist=100, minRadius=25, maxRadius=80)

            for circle in circles:
                resultTotal = []
                rect = hogDec.circle2Rect(circle)
                if rect[0] < 0 or rect[1] < 0 or rect[2] > 640 or rect[3] > 480:
                    continue 
                newRects = reshapeBallRect(rect, 4)      # 注意更改函数

                for newRect in newRects:
                    newInitX, newInitY = int(newRect[0]), int(newRect[1])
                    newEndX, newEndY = int(newRect[2]), int(newRect[3])
                    rectBallArea = srcImg[newInitY:newEndY, newInitX:newEndX, :]   # 矩形区域(宽，高，通道)                                                  

                    resultColor = calColorFeature(rectBallArea, 16)  
                    cellSize = min(newEndX - newInitX, newEndY - newInitY)
                    resultHOG = calHOGFeature(rectBallArea, cellSize / 2)             
                    resultTotal.extend(resultColor)
                    resultTotal.extend(resultHOG) 

                # cv2.rectangle(srcImg, (newInitX, newInitY), (newEndX, newEndY), (0, 0, 255), 2)  # 画矩形                  

            # cv2.imshow("test " + str(i), srcImg)          
            # cv2.waitKey(100) 
            # cv2.destroyAllWindows() 
                print('resultTotal', len(resultTotal)) 
                row = ' '.join(list(map(str, resultTotal))) + ' ' + str(0) + '\n'
                f.write(row)


def resultTest(method): 
    """
    分类结果测试
    Arguments: 
        method：分类器名字
    """   
    if method == "Logistic":
        trainingSet = []
        trainingLabels = []
        with open("data.txt", 'r') as f:
            for line in f.readlines():
                currLine = line.strip().split()
                lineArr = []
                for i in range(320):
                    lineArr.append(float(currLine[i]))
                trainingSet.append(lineArr)
                trainingLabels.append(float(currLine[320]))  

        log = Logistic("data.txt", 500)
        trainingWeights = log.gradDescent(trainingSet, trainingLabels)
        imgPath = "./img_test/"
        numbers = len(os.listdir(imgPath))
        for i in range(numbers):
            print("test" + str(i))
            srcImg = cv2.imread("./img_test/" + str(i) + ".jpg") 

            conDet = ContoursDetection(srcImg)
            preImg = conDet.preProcess(srcImg, "football")     # 注意更改名字
            rects = conDet.contoursDetection(preImg, minPerimeter=200, minK=0)         

            if rects == []:
                print("test" + str(i) + " no rects")

            for rect in rects:
                resultTotal = []
                rect = conDet.contour2Rect(rect)
                if rect[0] < 0 or rect[1] < 0 or rect[2] > 640 or rect[3] > 480:
                    print("out of bound")
                    continue 
                newRects = reshapeBallRect(rect, 4)

                for newRect in newRects:
                    newInitX, newInitY = int(newRect[0]), int(newRect[1])
                    newEndX, newEndY = int(newRect[2]), int(newRect[3])
                    rectBallArea = srcImg[newInitY:newEndY, newInitX:newEndX, :]                                   

                    resultColor = calColorFeature(rectBallArea, 16)  
                    cellSize = min(newEndX - newInitX, newEndY - newInitY)
                    resultHOG = calHOGFeature(rectBallArea, cellSize / 2)             
                    resultTotal.extend(resultColor)
                    resultTotal.extend(resultHOG)       

                resultTotal = np.array(resultTotal).reshape(1, -1)
                classify = log.classifyVector(resultTotal, trainingWeights)   

                if classify > 0.5:
                    classifyResult = "yes"
                    cv2.rectangle(srcImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)             

                else:
                    classifyResult = "no"
                    cv2.rectangle(srcImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)  

                print('classify', classifyResult)
            cv2.imshow("test " + str(i), srcImg)          
            cv2.waitKey(0) 

    elif method == "KNN":
        knn = KNN("data.txt")
        imgPath = "./img_test/"
        numbers = len(os.listdir(imgPath))
        for i in range(numbers):
            print("test" + str(i))
            srcImg = cv2.imread("./img_test/" + str(i) + ".jpg") 

            conDet = ContoursDetection(srcImg)
            preImg = conDet.preProcess(srcImg, "football")
            rects = conDet.contoursDetection(preImg, minPerimeter=200, minK=0)
            if rects == []:
                print("test" + str(i) + " no rects")

            for rect in rects:
                resultTotal = []
                rect = conDet.contour2Rect(rect)
                if rect[0] < 0 or rect[1] < 0 or rect[2] > 640 or rect[3] > 480:
                    print("out of bound")
                    continue 
                newRects = reshapeBallRect(rect, 4)

                for newRect in newRects:
                    newInitX, newInitY = int(newRect[0]), int(newRect[1])
                    newEndX, newEndY = int(newRect[2]), int(newRect[3])
                    rectBallArea = srcImg[newInitY:newEndY, newInitX:newEndX, :]                                   

                    resultColor = calColorFeature(rectBallArea, 16)  
                    cellSize = min(newEndX - newInitX, newEndY - newInitY)
                    resultHOG = calHOGFeature(rectBallArea, cellSize / 2)             
                    resultTotal.extend(resultColor)
                    resultTotal.extend(resultHOG)       

                resultTotal = np.array(resultTotal).reshape(1, -1).astype('float64')
                classify = knn.classifyVector(resultTotal)
                print(classify)

                if classify == 1:
                    classifyResult = "yes"
                    cv2.rectangle(srcImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)             

                else:
                    classifyResult = "no"
                    cv2.rectangle(srcImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)  

                print('classify', classifyResult)
            cv2.imshow("test " + str(i), srcImg)          
            cv2.waitKey(0) 


if __name__ == '__main__':
    # calPosVector("data_pos.txt")     # 计算正样本的特征向量
    # calNegVector("data_neg.txt")     # 计算负样本的特征向量
    resultTest("KNN")                # 测试分类结果
