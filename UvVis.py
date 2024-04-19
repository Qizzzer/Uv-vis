import os
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout,QComboBox, QGroupBox, QGridLayout, QScrollArea
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt
from pypylon import pylon
import cv2
import sys

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

        cameraSettingsGroup.setLayout(cameraSettingsLayout)

        # Camera Controls Group
        cameraControlGroup = QGroupBox("Camera Controls")
        cameraControlLayout = QVBoxLayout()
        self.initCameraBtn = QPushButton("Open Camera")
        self.captureAqueousBtn = QPushButton("Capture Water")
        self.captureSampleBtn = QPushButton("Capture Sample")
        self.closeCameraBtn = QPushButton("Close Camera")
        cameraControlLayout.addWidget(self.initCameraBtn)
        cameraControlLayout.addWidget(self.captureAqueousBtn)
        cameraControlLayout.addWidget(self.captureSampleBtn)
        cameraControlLayout.addWidget(self.closeCameraBtn)
        cameraControlGroup.setLayout(cameraControlLayout)

        # Directory Display
        self.directoryDisplayLabel = QLabel("Save Directories will be displayed here")
        #self.directoryDisplayLabel.clicked.connect(self.updateDirectoryDisplay())

        # Image Processing Controls Group
        imageProcessingGroup = QGroupBox("Image Processing")
        imageProcessingLayout = QVBoxLayout()
        self.averageSamplesBtn = QPushButton("Average Sample")
        self.averageWatersBtn = QPushButton("Average Water")
        self.findMaxSumAreasBtn = QPushButton("Find Max Sum Areas")
        imageProcessingLayout.addWidget(self.averageSamplesBtn)
        imageProcessingLayout.addWidget(self.averageWatersBtn)
        imageProcessingLayout.addWidget(self.findMaxSumAreasBtn)
        imageProcessingGroup.setLayout(imageProcessingLayout)

        # Image Display Area with Scroll
        self.imageDisplayLabel = QLabel()
        self.imageDisplayLabel.setScaledContents(True)
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setWidget(self.imageDisplayLabel)

        # Layout Organization
        topLayout = QHBoxLayout()
        topLayout.addWidget(cameraSettingsGroup)
        topLayout.addWidget(cameraControlGroup)
        topLayout.addWidget(imageProcessingGroup)

        mainLayout.addLayout(topLayout)
        mainLayout.addWidget(self.directoryDisplayLabel)
        mainLayout.addWidget(scrollArea)

        self.setWindowTitle("Camera Control")

        # connect
        self.initCameraBtn.clicked.connect(self.initCamera)
        self.captureAqueousBtn.clicked.connect(lambda: self.captureImage("water_picture"))
        self.captureSampleBtn.clicked.connect(lambda: self.captureImage("sample_picture"))
        self.closeCameraBtn.clicked.connect(self.closeCamera)
        self.averageSamplesBtn.clicked.connect(lambda: self.averageImages("sample_picture"))
        self.averageWatersBtn.clicked.connect(lambda: self.averageImages("water_picture"))
        self.findMaxSumAreasBtn.clicked.connect(self.findMaxSumAreas)

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
        exposureTime = int(self.exposureTimeInput.text())
        if selectedCamera != "Virtual Camera":
            self.camera.ExposureTimeAbs.SetValue(exposureTime)
        print(f"Camera initialized with exposure time: {exposureTime}us")
        self.updateDirectoryDisplay(f"Camera initialized with exposure time: {exposureTime}us")

    def captureImage(self, imageType):
        if not self.camera or not self.camera.IsOpen:
            print("Camera is not initialized.")
            self.updateDirectoryDisplay("Camera is not initialized.")
            return

        # Determine save directory
        save_directory = imageType
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Capture and save the image
        self.camera.StartGrabbingMax(1)
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.png"
            cv2.imwrite(os.path.join(save_directory, filename), img)
            imagePath = os.path.join(save_directory, filename)
            self.updateImageDisplay(imagePath)
            self.directoryDisplayLabel.setText(f"Last saved in: {os.path.abspath(save_directory)}")
            print(f"Image saved in {imagePath}")
            self.updateDirectoryDisplay(f"Image saved in {imagePath}")
        grabResult.Release()
    def averageImages(self, imageType):
        # Example to average images in the 'sample_picture' folder
        folder_path = imageType
        average_image = self.average_images_in_folder(folder_path)
        if average_image is not None:
            # Convert to QPixmap for display
            # Convert the averaged image from float64 to uint8 before color conversion
            average_image_uint8 = cv2.normalize(average_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

            average_image = cv2.cvtColor(average_image_uint8, cv2.COLOR_BGR2RGB)
            height, width, channel = average_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(average_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.imageDisplayLabel.setPixmap(pixmap)
            # Save the averaged image
            cv2.imwrite(os.path.join(folder_path, 'average_image.png'), average_image)
            print("Average image saved")
            self.updateDirectoryDisplay(f"Average image saved in {folder_path}")

    @staticmethod
    def average_images_in_folder(folder_path,):
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
                image = cv2.imread(file_path)

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
            return average_image
        else:
            print(f"No valid images found in '{folder_path}'.")
            return None

    def findMaxSumAreas(self):
        # Example usage with an averaged image
        imagePath = 'sample_picture/average_image.png'
        average_image = cv2.imread(imagePath)
        if average_image is not None:
            max_sums, starting_rows = self.find_non_overlapping_max_sum_areas(average_image)
            for i, start_row in enumerate(starting_rows, start=1):
                end_row = start_row + 50  # Assuming 50 as rows_per_area
                print(f"Area {i}: Rows {start_row} to {end_row}")
                # Update resultDisplayLabel or draw on the image as needed

    @staticmethod
    def find_non_overlapping_max_sum_areas(average_image, num_areas=3, rows_per_area=50, min_gap=200):
        average_image = average_image[:, :, 0]
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

    def updateImageDisplay(self, imagePath):
        pixmap = QPixmap(imagePath)
        self.imageDisplayLabel.setPixmap(
            pixmap.scaled(self.imageDisplayLabel.width(), self.imageDisplayLabel.height(), Qt.KeepAspectRatio))


    def updateDirectoryDisplay(self, message):
        self.directoryDisplayLabel.setText(message)


    def closeCamera(self):
        if self.camera and self.camera.IsOpen:
            self.camera.Close()
            print("Camera closed")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CameraControlApp()
    ex.show()
    sys.exit(app.exec_())
