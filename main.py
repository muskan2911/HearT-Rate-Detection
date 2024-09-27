import numpy as np
import cv2
import sys
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

class HeartRateMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.realWidth = 320
        self.realHeight = 240
        self.videoWidth = 160
        self.videoHeight = 120
        self.videoChannels = 3
        self.videoFrameRate = 15

        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIndex = 0

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.loadingTextLocation = (20, 30)
        self.bpmTextLocation = (self.videoWidth // 2 + 5, 30)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)
        self.lineType = 2
        self.boxColor = (0, 255, 0)
        self.boxWeight = 3

        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(3, self.realWidth)
        self.webcam.set(4, self.realHeight)

        self.firstFrame = np.zeros((self.videoHeight, self.videoWidth, self.videoChannels))
        self.firstGauss = self.buildGauss(self.firstFrame, self.levels + 1)[self.levels]
        self.videoGauss = np.zeros((self.bufferSize, self.firstGauss.shape[0], self.firstGauss.shape[1], self.videoChannels))
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        self.frequencies = (1.0 * self.videoFrameRate) * np.arange(self.bufferSize) / (1.0 * self.bufferSize)
        self.mask = (self.frequencies >= self.minFrequency) & (self.frequencies <= self.maxFrequency)

        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        self.i = 0
        self.monitoring = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)

        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Prepare CSV file for writing heart rate data
        self.csv_file = open('heart_rate_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Heart Rate (BPM)'])

    def initUI(self):
        self.setWindowTitle('Heart Rate Monitor')
        self.setStyleSheet("background-color: #2E3440; color: white;")

        # Create main layout
        self.layout = QVBoxLayout()

        # Video layout
        self.videoLayout = QHBoxLayout()
        self.videoLabel = QLabel()
        self.videoLabel.setFixedSize(320, 240)
        self.videoLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.videoLabel.setStyleSheet("border: 2px solid #4C566A;")
        self.videoLayout.addWidget(self.videoLabel)

        # Heart rate layout
        self.hrLayout = QVBoxLayout()
        self.hrLabel = QLabel('HR: Not Available')
        self.hrLabel.setFont(QFont('Arial', 24))
        self.hrLabel.setStyleSheet("color: #88C0D0;")
        self.hrLayout.addWidget(self.hrLabel, alignment=Qt.AlignCenter)

        # Heart rate trend plot
        self.hrTrendPlot = pg.PlotWidget(title="Heart Rate Trend")
        self.hrTrendPlot.setBackground('#3B4252')
        self.hrTrendPlot.getAxis('left').setPen(pg.mkPen('white'))
        self.hrTrendPlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.hrTrendPlot.setYRange(40, 180)
        self.hrCurve = self.hrTrendPlot.plot(pen=pg.mkPen('r', width=2))
        self.hrData = []

        self.hrLayout.addWidget(self.hrTrendPlot)
        self.videoLayout.addLayout(self.hrLayout)

        # Add the video and HR layout to the main layout
        self.layout.addLayout(self.videoLayout)

        # Pulse signal plot
        self.pulsePlot = pg.PlotWidget(title="Pulse Signal")
        self.pulsePlot.setBackground('#3B4252')
        self.pulsePlot.getAxis('left').setPen(pg.mkPen('white'))
        self.pulsePlot.getAxis('bottom').setPen(pg.mkPen('white'))
        self.pulsePlot.setYRange(0, 5000)
        self.pulseCurve = self.pulsePlot.plot(pen=pg.mkPen('b', width=2))
        self.layout.addWidget(self.pulsePlot)

        # Start/Stop buttons
        self.buttonLayout = QHBoxLayout()
        self.startButton = QPushButton('Start')
        self.stopButton = QPushButton('Stop')
        self.startButton.setStyleSheet("background-color: #5E81AC; color: white;")
        self.stopButton.setStyleSheet("background-color: #BF616A; color: white;")

        self.startButton.clicked.connect(self.startMonitoring)
        self.stopButton.clicked.connect(self.stopMonitoring)

        self.buttonLayout.addWidget(self.startButton)
        self.buttonLayout.addWidget(self.stopButton)
        self.layout.addLayout(self.buttonLayout)

        # Set the main layout
        self.setLayout(self.layout)

    def buildGauss(self, frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(self, pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:self.videoHeight, :self.videoWidth]
        return filteredFrame

    def startMonitoring(self):
        self.monitoring = True
        self.timer.start(1000 // self.videoFrameRate)

    def stopMonitoring(self):
        self.monitoring = False
        self.timer.stop()

    def normalizeSkinColor(self, frame):
        normalizedFrame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return normalizedFrame

    def applyAdaptiveHistogramEqualization(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        equalizedFrame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return equalizedFrame

    def saveToCSV(self, bpm):
        import time
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.csv_writer.writerow([timestamp, bpm])
        self.csv_file.flush()  # Ensure data is written to the file

    def update(self):
        if not self.monitoring:
            return

        ret, frame = self.webcam.read()
        if not ret:
            return

        # Transform to RGB color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = self.applyAdaptiveHistogramEqualization(frame)
        frame = self.normalizeSkinColor(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            self.hrLabel.setText("No face detected")
            return

        x, y, w, h = faces[0]
        detectionFrame = frame[y:y + h, x:x + w]

        # Focusing on the forehead region
        forehead = detectionFrame[0:int(0.3 * h), 0:w]

        detectionFrame = cv2.resize(forehead, (self.videoWidth, self.videoHeight))
        self.videoGauss[self.bufferIndex] = self.buildGauss(detectionFrame, self.levels + 1)[self.levels]
        fourierTransform = np.fft.fft(self.videoGauss, axis=0)

        fourierTransform[self.mask == False] = 0

        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            self.i += 1
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            bpm = 60.0 * hz
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize
            self.hrData.append(bpm)
            self.hrCurve.setData(self.hrData[-50:])  # Show the last 50 data points

            # Save heart rate to CSV file
            self.saveToCSV(bpm)

        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * self.alpha

        filteredFrame = self.reconstructFrame(filtered, self.bufferIndex, self.levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize

        frame[y:y + int(0.3 * h), x:x + w] = cv2.resize(outputFrame, (w, int(0.3 * h)))
        cv2.rectangle(frame, (x, y), (x + w, y + int(0.3 * h)), self.boxColor, self.boxWeight)
        if self.i > self.bpmBufferSize:
            bpm_text = "HR: %.1f" % self.bpmBuffer.mean()
        else:
            bpm_text = "Calculating HR..."

        self.hrLabel.setText(bpm_text)

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width, channel = img.shape
        step = channel * width
        qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

        self.pulseCurve.setData(self.frequencies, self.fourierTransformAvg)

    def closeEvent(self, event):
        self.timer.stop()
        
        self.webcam.release()
        self.csv_file.close()  # Close the CSV file when the application closes
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HeartRateMonitor()
    ex.show()
    sys.exit(app.exec_())
