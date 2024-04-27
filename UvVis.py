import os
import time
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout,QComboBox, QGroupBox, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap, QImage ,QCursor
from PyQt5.QtCore import Qt, QTimer
from pypylon import pylon
import cv2
import sys

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        # Get the coordinates of the mouse in the label
        x, y = event.x(), event.y()
        # Calculate the real coordinates in the image if the image is scaled
        pixmap = self.pixmap()
        if pixmap:
            scale_w = pixmap.width() / self.width()
            scale_h = pixmap.height() / self.height()
            real_x, real_y = int(x * scale_w), int(y * scale_h)
            self.setToolTip(f"Coordinates: ({real_y}, {real_x})")
            self.setCursor(QCursor(Qt.CrossCursor))

class MockCamera:
    def __init__(self):
        self.isOpen = False

    def Open(self):
        self.isOpen = True
        print("Virtual camera opened")

    def Close(self):
        self.isOpen = False
        print("Virtual camera closed")

    def StartGrabbingMax(self, count):
        print("Started grabbing on virtual camera")

    def RetrieveResult(self, timeout, handling):
        # Simulate a successful grab result
        class GrabResult:
            def __init__(self):
                self.GrabSucceeded = lambda: True
                self.Array = self.simulate_image()
                self.Release = lambda: None

            def simulate_image(self):
                # Return a dummy image array (e.g., a black image for simplicity)
                #return np.zeros((480, 640), dtype=np.uint8)
                return np.random.randint(0, 255, (1280, 1530), dtype=np.uint8)

        return GrabResult()

    def IsOpen(self):
        return self.isOpen

class CameraControlApp(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateImageDisplay)
        self.initUI()

    def initUI(self):
        mainLayout = QVBoxLayout(self)

        # Camera Selection and Settings Group
        cameraSettingsGroup = QGroupBox("Camera Settings")
        cameraSettingsLayout = QGridLayout()

        # Camera Selection
        cameraSettingsLayout.addWidget(QLabel("Select Camera:"), 0, 0)
        self.cameraSelectionComboBox = QComboBox()
        self.cameraSelectionComboBox.addItems(self.listCameras())  # Assuming listCameras() is defined elsewhere
        cameraSettingsLayout.addWidget(self.cameraSelectionComboBox, 0, 1)

        # Exposure Time Input
        cameraSettingsLayout.addWidget(QLabel("Exposure Time (us):"), 1, 0)
        self.exposureTimeInput = QLineEdit("3000")  # Default exposure time in microseconds
        cameraSettingsLayout.addWidget(self.exposureTimeInput, 1, 1)

        self.initCameraBtn = QPushButton("Init Camera")
        self.closeCameraBtn = QPushButton("Close Camera")
        cameraSettingsLayout.addWidget(self.initCameraBtn, 2, 0)
        cameraSettingsLayout.addWidget(self.closeCameraBtn, 2, 1)
        cameraSettingsGroup.setLayout(cameraSettingsLayout)

        # Camera Controls Group
        cameraControlGroup = QGroupBox("Camera Controls")
        cameraControlLayout = QVBoxLayout()

        self.captureAqueousBtn = QPushButton("Capture Water")
        self.captureSampleBtn = QPushButton("Capture Sample")
        self.captureAqueousDeclineBtn = QPushButton("Capture Water Decline")
        self.captureSampleDeclineBtn = QPushButton("Capture Sample Decline")

        cameraControlLayout.addWidget(self.captureAqueousBtn)
        cameraControlLayout.addWidget(self.captureSampleBtn)
        cameraControlLayout.addWidget(self.captureAqueousDeclineBtn)
        cameraControlLayout.addWidget(self.captureSampleDeclineBtn)
        cameraControlGroup.setLayout(cameraControlLayout)

        # Directory Display
        self.directoryDisplayLabel = QLabel("Save Directories will be displayed here")
        #self.directoryDisplayLabel.clicked.connect(self.updateDirectoryDisplay())

        # Image Processing Controls Group
        imageProcessingGroup = QGroupBox("Image Processing")
        imageProcessingLayout = QVBoxLayout()
        self.averageSamplesBtn = QPushButton("Average Sample")
        self.averageWatersBtn = QPushButton("Average Water")
        self.averageSamplesDeclineBtn = QPushButton("Average Sample Decline")
        self.averageWatersDeclineBtn = QPushButton("Average Water Decline")
        self.findMaxSumAreasBtn = QPushButton("Find Max Sum Areas")
        self.processImagesBtn = QPushButton("Process Image Areas")
        imageProcessingLayout.addWidget(self.averageSamplesBtn)
        imageProcessingLayout.addWidget(self.averageWatersBtn)
        imageProcessingLayout.addWidget(self.averageSamplesDeclineBtn)
        imageProcessingLayout.addWidget(self.averageWatersDeclineBtn)
        imageProcessingLayout.addWidget(self.findMaxSumAreasBtn)
        imageProcessingLayout.addWidget(self.processImagesBtn)
        imageProcessingGroup.setLayout(imageProcessingLayout)

        # Spectral wavelength calibration
        spectralGroup = QGroupBox("Spectral wavelength calibration")
        spectralLayout = QGridLayout()

        # Exposure Time Input 需要改写变量名
        spectralLayout.addWidget(QLabel("437nm :"), 0, 0)
        self.spectral437nm = QLineEdit("200")  #
        self.spectral437nm.setFixedWidth(50)
        spectralLayout.addWidget(self.spectral437nm, 0, 1)

        spectralLayout.addWidget(QLabel("546nm:"), 0, 2)
        self.spectral546nm = QLineEdit("300")  #
        self.spectral546nm.setFixedWidth(50)
        spectralLayout.addWidget(self.spectral546nm, 0, 3)

        spectralLayout.addWidget(QLabel("577nm:"), 1, 0)
        self.spectral577nm = QLineEdit("400")  #
        self.spectral577nm.setFixedWidth(50)
        spectralLayout.addWidget(self.spectral577nm, 1, 1)

        spectralLayout.addWidget(QLabel("579nm:"), 1, 2)
        self.spectral579nm = QLineEdit("400",)  #
        self.spectral579nm.setFixedWidth(50)
        spectralLayout.addWidget(self.spectral579nm, 1, 3)
        spectralGroup.setLayout(spectralLayout)

        # Image Display Area with Scroll
        self.imageDisplayLabel = ImageLabel(self)
        self.imageDisplayLabel.setScaledContents(True)
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(self.imageDisplayLabel)

        # Layout Organization
        topLayout = QHBoxLayout()
        topLayout.addWidget(cameraSettingsGroup)
        topLayout.addWidget(cameraControlGroup)
        topLayout.addWidget(imageProcessingGroup)
        topLayout.addWidget(spectralGroup)
        mainLayout.addLayout(topLayout)
        mainLayout.addWidget(self.directoryDisplayLabel)
        mainLayout.addWidget(scrollArea)

        self.setWindowTitle("Camera Control")

        # connect
        self.initCameraBtn.clicked.connect(self.initCamera)
        self.captureAqueousBtn.clicked.connect(lambda: self.captureImage("water_picture"))
        self.captureSampleBtn.clicked.connect(lambda: self.captureImage("sample_picture"))
        self.captureAqueousDeclineBtn.clicked.connect(lambda: self.captureImage("water_decline_picture"))
        self.captureSampleDeclineBtn.clicked.connect(lambda: self.captureImage("sample_decline_picture"))
        self.closeCameraBtn.clicked.connect(self.closeCamera)
        self.averageSamplesBtn.clicked.connect(lambda: self.averageImages("sample_picture"))
        self.averageWatersBtn.clicked.connect(lambda: self.averageImages("water_picture"))
        self.averageSamplesDeclineBtn.clicked.connect(lambda: self.averageImages("sample_decline_picture"))
        self.averageWatersDeclineBtn.clicked.connect(lambda: self.averageImages("water_decline_picture"))
        self.findMaxSumAreasBtn.clicked.connect(self.findMaxSumAreas)
        self.processImagesBtn.clicked.connect(self.image2Spectral)

    def listCameras(self):
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if not devices:
            return ["Virtual Camera"]  # No real cameras found, return virtual camera option
        return [device.GetSerialNumber() for device in devices] + ["Virtual Camera"]


    def initCamera(self):
        selectedCamera = self.cameraSelectionComboBox.currentText()
        if selectedCamera == "Virtual Camera":
            self.camera = MockCamera()  # Use the mock camera
        else:
            # Initialize the real camera based on the selected serial number
            devices = pylon.TlFactory.GetInstance().EnumerateDevices()
            for device in devices:
                if device.GetSerialNumber() == selectedCamera:
                    self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
                    break
        self.camera.Open()
        if self.camera.IsOpen():
            self.timer.start(100)  # Start updating the display every 100 ms
        exposureTime = int(self.exposureTimeInput.text())
        if selectedCamera != "Virtual Camera":
            self.camera.ExposureTimeAbs.SetValue(exposureTime)
        print(f"Camera initialized with exposure time: {exposureTime}us")
        self.updateDirectoryDisplay(f"Camera initialized with exposure time: {exposureTime}us")

    def openCamera(self, imageType):
        self.camera.Open()
        if self.camera.IsOpen():
            self.timer.start(100)  # Start updating the display every 100 ms

    def captureImage(self, imageType):
        if not self.camera or not self.camera.IsOpen:
            print("Camera is not initialized.")
            self.updateDirectoryDisplay("Camera is not initialized.")
            return

        # Determine save directory
        save_directory = imageType
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.timer.stop()
        # Capture and save the image
        self.camera.StartGrabbingMax(1)
        grabResult = self.camera.RetrieveResult(5000,None)
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{imageType}_{timestamp}.png"
            cv2.imwrite(os.path.join(save_directory, filename), img)
            imagePath = os.path.join(save_directory, filename)
            image = QImage(grabResult.Array.data, grabResult.Array.shape[1], grabResult.Array.shape[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(image)
            self.imageDisplayLabel.setPixmap(pixmap)
            self.directoryDisplayLabel.setText(f"Last saved in: {os.path.abspath(save_directory)}")
            print(f"Image saved in {imagePath}")
            self.updateDirectoryDisplay(f"Image saved in {imagePath}，image maxpixe is {img.max()}")
        grabResult.Release()

    def updateImageDisplay(self):
        if self.camera.IsOpen():
            result = self.camera.RetrieveResult(5000, None)
            result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if result.GrabSucceeded():
                image = QImage(result.Array.data, result.Array.shape[1], result.Array.shape[0], QImage.Format_Grayscale8)
                #image = QImage(result.Array.data, result.Array.shape[1], result.Array.shape[0], QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.imageDisplayLabel.setPixmap(pixmap.scaled(self.imageDisplayLabel.size(), Qt.KeepAspectRatio))
    def updateDirectoryDisplay(self, message):
        self.directoryDisplayLabel.setText(message)

    def closeCamera(self):
        if self.camera and self.camera.IsOpen:
            self.camera.Close()
            self.updateDirectoryDisplay(f"Camera closed")
            print("Camera closed")

    def averageImages(self, imageType):
        # Example to average images in the 'sample_picture' folder
        folder_path = imageType
        average_image = self.average_images_in_folder(folder_path)
        if average_image is not None:
            height, width = average_image.shape
            bytesPerLine = width
            qImg = QImage(average_image.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            self.imageDisplayLabel.setPixmap(pixmap)
            # Save the averaged image
            cv2.imwrite(os.path.join(folder_path, f'average_{folder_path}.png'), average_image)
            print("Average image saved")
            self.updateDirectoryDisplay(f"Average image saved in {folder_path}/average_{folder_path}.png")

    @staticmethod
    def average_images_in_folder(folder_path):
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return None

        # Initialize variables to store the sum of images and the count of images
        sum_of_images = None
        image_count = 0

        # Iterate through files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Check if the file is an image (you can add more image extensions if needed)
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                # Read the image
                image = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)

                # Check if the image was read successfully
                if image is not None:
                    # Initialize the sum_of_images on the first iteration
                    if sum_of_images is None:
                        sum_of_images = np.zeros_like(image, dtype=np.float64)

                    # Add the image to the sum
                    sum_of_images += image.astype(np.float64)
                    image_count += 1

        # Check if at least one image was found
        if image_count > 0:
            # Calculate the average by dividing the sum by the count
            average_image = (sum_of_images / image_count).astype(np.float64)
            return average_image.astype(np.uint8)
        else:
            print(f"No valid images found in '{folder_path}'.")
            return None

    def findMaxSumAreas(self):
        # Example usage with an averaged image
        imagePath = 'sample_picture/average_sample_picture.png'
        average_image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        if average_image is not None:
            max_sums, starting_rows = self.find_non_overlapping_max_sum_areas(average_image)
            message = ""
            for i, start_row in enumerate(starting_rows, start=1):
                end_row = start_row + 50  # Assuming 50 as rows_per_area
                message = message + "   " + f"Area {i}: Rows {start_row} to {end_row}" + "  "
                self.directoryDisplayLabel.setText(message)
                print(f"Area {i}: Rows {start_row} to {end_row}")
                # Update resultDisplayLabel or draw on the image as needed

    @staticmethod
    def find_non_overlapping_max_sum_areas(average_image, num_areas=3, rows_per_area=50, min_gap=200):
        # Initialize variables to store the maximum sums and their starting rows
        max_sums = []
        starting_rows = []

        for _ in range(num_areas):
            # Calculate the sum of pixel values for each row
            row_sums = np.sum(average_image, axis=1)

            # Find the starting row for the current area with the maximum sum
            start_row = np.argmax(row_sums)

            # Extract the rows for the current area
            end_row = start_row + rows_per_area
            area_range = (start_row, end_row)
            area_image = average_image[start_row:end_row, :]

            # Calculate the maximum sum for the current area
            max_sum = np.sum(area_image)

            # Set the maximum sum and starting row for the current area
            max_sums.append(max_sum)
            starting_rows.append(start_row)

            # Set the rows in the current area and a buffer above and below it to zero
            buffer = min_gap + rows_per_area
            average_image[max(0, start_row - buffer):min(end_row + buffer, average_image.shape[0]), :] = 0
        return max_sums, starting_rows


    def process_image_areas(self):
        directories = ['water_picture', 'sample_picture']
        results = []

        for dir_name in directories:
            # Construct path to the average image
            image_path = os.path.join(dir_name, 'average_image.png')
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming grayscale for simplicity
            if image is not None:
                # Assuming find_non_overlapping_max_sum_areas is already defined
                _, starting_rows = self.find_non_overlapping_max_sum_areas(image)

                # Extract and process the specified areas
                for start_row in starting_rows[:3]:  # Assuming we need the first three areas
                    end_row = start_row + 50  # Assuming the area height is 50 rows
                    area = image[start_row:end_row, :]

                    # Example operation: calculate the sum of the area
                    area_sum = np.sum(area)
                    results.append(area_sum)

                    # Perform other operations as needed

        # Save the results to a text file
        with open('image_processing_results.txt', 'w') as file:
            for result in results:
                file.write(f"{result}\n")

        print("Image processing complete. Results saved to image_processing_results.txt.")

    @staticmethod
    def splitImage(path, first_area, second_area, third_area, length):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        t = img[first_area:first_area + length, :]
        s = img[second_area:second_area + length, :]
        c = img[third_area:third_area + length, :]
        return t, s, c

    @staticmethod
    def averageIntense(t, s, c):
        t_avg = np.average(t, axis=0)
        s_avg = np.average(s, axis=0)
        c_avg = np.average(c, axis=0)
        return t_avg, s_avg, c_avg

    @staticmethod
    def spectralCalibration(self):
        # 汞氩灯特征峰
        l1, l2, l3, l4 = 437, 546, 577, 579
        # 对应的序列数
        n1, n2, n3, n4 = int(self.spectral437nm.text()), int(self.spectral546nm.text()), int(self.spectral577nm.text()), int(self.spectral579nm.text())
        y_train = np.array([l1, l2, l3, l4])
        x_train = np.array([n1, n2, n3, n4])
        poly = PolynomialFeatures(degree=2)
        x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
        model = LinearRegression()
        model.fit(x_train_poly, y_train)
        image = cv2.imread("sample_picture/average_sample_picture.png",cv2.IMREAD_GRAYSCALE)
        rows = image.shape[1]
        x_test = np.arange(rows)
        x_test_poly = poly.fit_transform(x_test.reshape(-1, 1))
        y_pred = model.predict(x_test_poly)
        print(y_pred)
        print("模型的系数:", model.coef_)
        print("模型的截距:", model.intercept_)
        # plt.plot(y_pred,column_sum)
        return y_pred

    def image2Spectral(self):
        average_sample_picture = cv2.imread("sample_picture/average_sample_picture.png",cv2.IMREAD_GRAYSCALE)
        # get three area
        max_sums, starting_rows = self.find_non_overlapping_max_sum_areas(average_sample_picture)
        first, second, third = starting_rows

        # average area
        water_toushe, water_sanshe, water_canbi = self.splitImage("water_picture/average_water_picture.png", first, second, third, 50)
        water_avg_t, water_avg_s, water_avg_c = self.averageIntense(water_toushe, water_sanshe, water_canbi)

        water_decline_toushe, water_decline_sanshe, water_decline_canbi = self.splitImage("water_decline_picture/average_water_decline_picture.png", first, second, third, 50)
        water_decline_avg_t, water_decline_avg_s, water_decline_avg_c = self.averageIntense(water_decline_toushe, water_decline_sanshe, water_decline_canbi)

        sample_toushe, sample_sanshe, sample_canbi = self.splitImage("sample_picture/average_sample_picture.png", first, second, third, 50)
        sample_avg_t, sample_avg_s, sample_avg_c = self.averageIntense(sample_toushe, sample_sanshe, sample_canbi)

        sample_decline_toushe, sample_decline_sanshe, sample_decline_canbi = self.splitImage("sample_decline_picture/average_sample_decline_picture.png", first, second, third, 50)
        sample_decline_avg_t, sample_decline_avg_s, sample_decline_avg_c = self.averageIntense(sample_decline_toushe, sample_decline_sanshe, sample_decline_canbi)

        # c_w为水溶液参比光路衰减系数
        c_w = (water_avg_c * 200) / water_decline_avg_c
        c_w = list(c_w)
        c_yp = (sample_avg_c * 200) / sample_decline_avg_c
        c_yp = list(c_yp)
        # t为透射光路的衰减系数
        t_w = (water_avg_t * 200) / water_decline_avg_t
        t_w = list(t_w)
        t_yp = (sample_avg_t * 200) / sample_decline_avg_t
        t_yp = list(t_yp)

        # 得到考虑衰减系数后样品和水溶液三条光路的光强
        sample_avg_t = sample_avg_t * t_yp
        sample_avg_c = sample_avg_c * c_yp
        water_avg_c = water_avg_c * c_w
        water_avg_t = water_avg_t * t_w

        # 透射和参比校准
        t2c = water_avg_t / water_avg_c
        # 校正后样品透射
        sample_avg_t = sample_avg_t / t2c

        #计算吸收和散射
        A = np.log10(sample_avg_c / sample_avg_t)
        S = np.log10((sample_avg_s + sample_avg_t) / sample_avg_t)
        spectral = self.spectralCalibration(self)

        with open('output/xiaoguangdu.txt', 'w') as f:
            for i in range(len(A)):
                # 将两个列表中的元素用逗号隔开，然后写入文件中
                f.write(f"{spectral[i]},{A[i]}\n")

        with open('output/sanshedu.txt', 'w') as f:
            for i in range(len(S)):
                # 将两个列表中的元素用逗号隔开，然后写入文件中
                f.write(f"{spectral[i]},{S[i]}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CameraControlApp()
    ex.show()
    sys.exit(app.exec_())
