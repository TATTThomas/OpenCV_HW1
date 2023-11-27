from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QFileInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QWidget, QFormLayout, QLineEdit, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QFileDialog, QDialog
from UI import Ui_MainWindow
import cv2 as cv 
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt
import torchsummary
import numpy as np

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.image1_path = None
        self.image1_name = None
        self.image1 = None
        self.image2_path = None
        self.image2_name = None
        self.image2 = None

    def setup_control(self):
        self.ui.pushButton_loadImage_1.clicked.connect(self.load_image_1)
        self.ui.pushButton_loadImage_2.clicked.connect(self.load_image_2)
        self.ui.pushButton_1_1.clicked.connect(self.pb_1_1)
        self.ui.pushButton_1_2.clicked.connect(self.pb_1_2)
        self.ui.pushButton_1_3.clicked.connect(self.pb_1_3)
        self.ui.pushButton_2_1.clicked.connect(self.pb_2_1)
        self.ui.pushButton_2_2.clicked.connect(self.pb_2_2)
        self.ui.pushButton_2_3.clicked.connect(self.pb_2_3)
        self.ui.pushButton_3_1.clicked.connect(self.pb_3_1)
        self.ui.pushButton_3_2.clicked.connect(self.pb_3_2)
        self.ui.pushButton_3_3.clicked.connect(self.pb_3_3)
        self.ui.pushButton_3_4.clicked.connect(self.pb_3_4)
        self.ui.pushButton_4_1.clicked.connect(self.pb_4_1)
        self.ui.pushButton_5_1.clicked.connect(self.pb_5_1)
        self.ui.pushButton_5_2.clicked.connect(self.pb_5_2)

    def load_image_1(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'open image', '*.jpg;;*.png;;All Files(*)')[0]
        if len(path) == 0:
            return
        else:
            self.image1_path = path
            self.image1_name = QFileInfo(path).fileName()
            #* set label
            self.ui.image1_label.setText(self.image1_name)
            #* read image
            self.image1 = cv.imread(self.image1_path)

    def load_image_2(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'open image', '*.jpg;;*.png;;All Files(*)')[0]
        if len(path) == 0:
            return
        else:
            self.image2_path = path
            self.image2_name = QFileInfo(path).fileName()
            #* set label
            self.ui.image2_label.setText(self.image2_name)
            #* read image
            self.image2 = cv.imread(self.image2_path)

    def pb_1_1(self):
        if self.image1 is not None:
            image = self.image1.copy()
            zeros = np.zeros(image.shape[:2], dtype="uint8")
            b, g, r = cv.split(image)
            b_channel = cv.merge([b, zeros, zeros])
            g_channel = cv.merge([zeros, g, zeros])
            r_channel = cv.merge([zeros, zeros, r])
            cv.imshow('blue', b_channel)
            cv.imshow('green', g_channel)
            cv.imshow('red', r_channel)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def pb_1_2(self):
        if self.image1 is not None:
            image = self.image1.copy()
            
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            b, g, r = cv.split(image)
            average_weighted = (b + g + r) // 3
                
            cv.imshow('grayscale', grayscale)
            cv.imshow('Average weighted', average_weighted)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def pb_1_3(self):
        if self.image1 is not None:
            image = self.image1.copy()
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            lower_bound = np.array([20, 20, 25])
            upper_bound = np.array([80, 255, 255])
            mask = cv.inRange(hsv , lower_bound, upper_bound)
            mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            rm_yellow_green = cv.bitwise_not(mask_bgr, image ,mask)
            cv.imshow('mask', mask)
            cv.imshow('Image without yellow and green', rm_yellow_green)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def pb_2_1(self):
        if self.image1 is not None:
            cv.imshow('gaussian_blur',self.image1)
            cv.createTrackbar('m', 'gaussian_blur', 1, 5, self.pb_2_1_control)
            cv.setTrackbarPos('m', 'gaussian_blur', 0)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
    def pb_2_1_control(self, m):
        cv.imshow('gaussian_blur', cv.GaussianBlur(self.image1, (2 * m + 1, 2 * m + 1), 0))

    def pb_2_2(self):
        if self.image1 is not None:
            cv.imshow('bilateral_filter',self.image1)
            cv.createTrackbar('m', 'bilateral_filter', 1, 5, self.pb_2_2_control)
            cv.setTrackbarPos('m', 'bilateral_filter', 0)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
    def pb_2_2_control(self, m):
        cv.imshow('bilateral_filter', cv.bilateralFilter(self.image1, 2 * m + 1, 90, 90))

    def pb_2_3(self):
        if self.image1 is not None:
            cv.imshow('median_filter',self.image1)
            cv.createTrackbar('m', 'median_filter', 1, 5, self.pb_2_3_control)
            cv.setTrackbarPos('m', 'median_filter', 0)
            cv.waitKey(0)
            cv.destroyAllWindows()
        
    def pb_2_3_control(self, m):
        cv.imshow('median_filter', cv.medianBlur(self.image1, 2 * m + 1))

    def convolution(self, image, filter):
        height, width = image.shape
        result = np.zeros((height - 2, width - 2), dtype=np.int32)

        for i in range(height - 2):
            for j in range(width - 2):
                pixal = image[i : i+3, j : j+3]
                result[i, j] = np.sum(pixal * filter)

        return result

    def pb_3_1(self):
        if self.image1 is not None:
            image = self.image1.copy()
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gaussian = cv.GaussianBlur(grayscale, (3, 3), 0)
            sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            cov_result = self.convolution(gaussian, sobel_x_filter)

            result = np.abs(cov_result)
            result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            
            cv.imshow('Sobel x', result)
            cv.waitKey(0)
            cv.destroyAllWindows()


    def pb_3_2(self):
        if self.image1 is not None:
            image = self.image1.copy()
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gaussian = cv.GaussianBlur(grayscale, (3, 3), 0)
            sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            cov_result = self.convolution(gaussian, sobel_y_filter)

            result = np.abs(cov_result)
            result = cv.normalize(result, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            
            cv.imshow('Sobel y', result)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def pb_3_3(self):
        if self.image1 is not None:
            image = self.image1.copy()
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gaussian = cv.GaussianBlur(grayscale, (3, 3), 0)
            sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_x = self.convolution(gaussian, sobel_x_filter)
            sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            sobel_y = self.convolution(gaussian, sobel_y_filter)

            combination = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            combination = cv.normalize(combination, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            
            cv.imshow('combination', combination)
            threshold = combination
            for i in range(0, combination.shape[0]):
                for j in range(0, combination.shape[1]):
                    if combination[i,j] < 128:
                        threshold[i,j] = 0
                    else:
                        threshold[i,j] = 255

            cv.imshow('threshold result', threshold)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def pb_3_4(self):
        if self.image1 is not None:
            image = self.image1.copy()
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            gaussian = cv.GaussianBlur(grayscale, (3, 3), 0)
            sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_x = self.convolution(gaussian, sobel_x_filter)
            sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            sobel_y = self.convolution(gaussian, sobel_y_filter)

            combination = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            combination = cv.normalize(combination, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

            mask1 = np.zeros((combination.shape[0], combination.shape[1]))
            mask2 = np.zeros((combination.shape[0], combination.shape[1]))
            for i in range(0, combination.shape[0]):
                for j in range(0, combination.shape[1]):
                    angle = np.degrees(np.arctan2(sobel_y[i][j], sobel_x[i][j]))
                    if (angle >= 120) & (angle <= 180):
                        mask1[i, j] = 255
                    else:
                        mask1[i, j] = 0
                    if (angle >= -150) & (angle <= -30):
                        mask2[i, j] = 255
                    else:
                        mask2[i, j] = 0

            cv.imshow('mask1', cv.bitwise_and(combination, mask1.astype(np.uint8)))
            cv.imshow('mask2', cv.bitwise_and(combination, mask2.astype(np.uint8)))
            cv.waitKey(0)
            cv.destroyAllWindows()

    def pb_4_1(self):
        rotation_matrix = cv.getRotationMatrix2D((240, 200), int(self.ui.lineEdit_rotation.text()), float(self.ui.lineEdit_scaling.text()))

        output = cv.warpAffine(self.image1, rotation_matrix,(1920, 1080))
        output = cv.warpAffine(output, np.array([[1, 0, float(self.ui.lineEdit_tx.text())], [0, 1, int(self.ui.lineEdit_ty.text())]]),(1920, 1080))
        cv.imshow('rotation', output)
        
        cv.waitKey(0)
        cv.destroyAllWindows()

    def pb_5_1(self):
        image_path = ["Q5_image/Q5_1/automobile.png", "Q5_image/Q5_1/bird.png", "Q5_image/Q5_1/cat.png", "Q5_image/Q5_1/deer.png", "Q5_image/Q5_1/dog.png", "Q5_image/Q5_1/frog.png", "Q5_image/Q5_1/horse.png", "Q5_image/Q5_1/ship.png", "Q5_image/Q5_1/truck.png"]
        images = []
        for path in image_path:
            images.append(Image.open(path))

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(30)])
        aug_images = []
        for image in images:
            aug_images.append(transform(image))
        
        plt.figure()
        count = 0
        label = {
                0:"automobile", 
                1:"bird", 
                2:"cat",
                3:"deer", 
                4:"dog", 
                5:"frog", 
                6:"horse", 
                7:"ship", 
                8:"truck"
                }
        
        for image in aug_images:
            plt.subplot(3, 3, count+1)
            plt.imshow(image)
            plt.title(label[count])
            plt.axis('off')
            count += 1

        plt.show()

    def pb_5_2(self):
        self.model = torchvision.models.vgg19_bn(num_classes=10)
        print(torchsummary.summary(self.model, (3, 32, 32)))