import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime
from skimage import io
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import time
from PyQt5 import QtCore, QtGui, QtWidgets

#cnt = 0
#;gui.pbar.setValue(cnt);cnt += 1

def findEncodings(images,cnt):

    encodeList = [];gui.pbar.setValue(cnt);cnt += 1

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);gui.pbar.setValue(cnt);cnt += 1
        encode = face_recognition.face_encodings(img)[0];gui.pbar.setValue(cnt);cnt += 1
        encodeList.append(encode);gui.pbar.setValue(cnt);cnt += 1
    return encodeList


def markAttendance(name,cnt):
    with open('Attendance.csv', 'r+', newline='') as f:
        csvFile = csv.reader(f);gui.pbar.setValue(cnt);cnt += 1
        t = 0;gui.pbar.setValue(cnt);cnt += 1
        for i in csvFile:
            # print(i)
            if (len(i) == 0):
                gui.pbar.setValue(cnt);cnt += 1
                continue
            else:
                # print(i[0])
                if (i[0] == name):
                    t = 1;gui.pbar.setValue(cnt);cnt += 1
                    break
        if t == 0:
            now = datetime.now();gui.pbar.setValue(cnt);cnt += 1
            dtString = now.strftime('%H:%M:%S');gui.pbar.setValue(cnt);cnt += 1
            list = [name, dtString];gui.pbar.setValue(cnt);cnt += 1
            csvwriter = csv.writer(f);gui.pbar.setValue(cnt);cnt += 1
            csvwriter.writerow(list);gui.pbar.setValue(cnt);cnt += 1
            
#gui.pbar.setValue(cnt);cnt += 1
def makeMenu(pt, nm, ch):
    cnt = 0
    #print(nm)
    #print(pt)
    #print("MENU:\n1. Encode List\n2. Get Attendance\n")
    #while True:
    path = 'Trained_images';gui.pbar.setValue(cnt);cnt += 1
    images = [];gui.pbar.setValue(cnt);cnt += 1
    classNames = [];gui.pbar.setValue(cnt);cnt += 1
    dr = os.getcwd();gui.pbar.setValue(cnt);cnt += 1
    str = '\%s' % path;gui.pbar.setValue(cnt);cnt += 1
    dr1 = dr + str;gui.pbar.setValue(cnt);cnt += 1
    #print(dr)
    myList = os.listdir(path);gui.pbar.setValue(cnt);cnt += 1
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}');gui.pbar.setValue(cnt);cnt += 1
        images.append(curImg);gui.pbar.setValue(cnt);cnt += 1
        # print(images)
        classNames.append(os.path.splitext(cl)[0]);gui.pbar.setValue(cnt);cnt += 1
        # ch = int(input("Enter choice: "))
    gui.pbar.setValue(cnt);cnt += 1

    if (ch == 1):
        gui.pbar.setValue(cnt);cnt += 1
        im = cv2.imread(pt);gui.pbar.setValue(cnt);cnt += 1
        gui.pbar.setValue(cnt);cnt += 1
        os.chdir(dr1);gui.pbar.setValue(cnt);cnt += 1
        #print(nm)
        cv2.imwrite(nm, im);gui.pbar.setValue(cnt);cnt += 1
        os.chdir(dr);gui.pbar.setValue(cnt);cnt += 1
        images = [];gui.pbar.setValue(cnt);cnt += 1
        classNames = [];gui.pbar.setValue(cnt);cnt += 1
        myList = os.listdir(path);gui.pbar.setValue(cnt);cnt += 1
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}');gui.pbar.setValue(cnt);cnt += 1
            images.append(curImg);gui.pbar.setValue(cnt);cnt += 1
            classNames.append(os.path.splitext(cl)[0]);gui.pbar.setValue(cnt);cnt += 1
        #print(classNames)
        encodeListKnown = findEncodings(images,cnt);gui.pbar.setValue(cnt);cnt += 1
        np.save('file', encodeListKnown);gui.pbar.setValue(cnt);cnt += 1
        #print("Data Encoded");
        gui.pbar.setValue(100);cnt += 1
    
    elif ch == 2:
        encodeListKnown = np.load('file.npy');gui.pbar.setValue(cnt);cnt += 1
        # nm = input("Enter name of image: ")
        # success, img = cap.read()
        img = cv2.imread(pt);gui.pbar.setValue(cnt);cnt += 1
        # imgS = cv2.resize(img, (0, 0), None, 0.25,0.25)
        # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(img);gui.pbar.setValue(cnt);cnt += 1
        encodesCurFrame = face_recognition.face_encodings(img, facesCurFrame);gui.pbar.setValue(cnt);cnt += 1
        c = 0;gui.pbar.setValue(cnt);cnt += 1
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace);gui.pbar.setValue(cnt);cnt += 1
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace);gui.pbar.setValue(cnt);cnt += 1
            # print(faceDis)
            matchIndex = np.argmin(faceDis);gui.pbar.setValue(cnt);cnt += 1

            if matches[matchIndex]:
                name = classNames[matchIndex].upper();gui.pbar.setValue(cnt);cnt += 1
            # print(name)
                y1, x2, y2, x1 = faceLoc;gui.pbar.setValue(cnt);cnt += 1
                # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2);gui.pbar.setValue(cnt);cnt += 1
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2),(0, 255, 0), cv2.FILLED);gui.pbar.setValue(cnt);cnt += 1
                cv2.putText(img, name, (x1 + 6, y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2);gui.pbar.setValue(cnt);cnt += 1
                # c += 1
                markAttendance(name,cnt);gui.pbar.setValue(cnt);cnt += 1
            # print(c)
        # cv2.imshow('Webcam', img)
        dim = img.shape;gui.pbar.setValue(cnt);cnt += 1
        h = dim[0];gui.pbar.setValue(cnt);cnt += 1
        w = dim[1];gui.pbar.setValue(cnt);cnt += 1
        h1 = 1366;gui.pbar.setValue(cnt);cnt += 1
        w1 = 768;gui.pbar.setValue(cnt);cnt += 1
        # print(h," ",w," ",h1," ",w1)
        # if (h > h1 or w > w1):
        #     cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("window",min(w,1366),min(h,768))
        #     #cv2.imshow("window", img)
        # else:
        #     cv2.namedWindow("img")
        #     cv2.moveWindow("img", 0, 0)
        #     #cv2.imshow("img", img)
        cv2.imwrite("output.jpg", img);gui.pbar.setValue(cnt);cnt += 1
        obj = Template();gui.pbar.setValue(cnt);cnt += 1
        gui.open_image1("output.jpg");gui.pbar.setValue(cnt);cnt += 1
        #cv2.waitKey()
        gui.pbar.setValue(100);cnt += 1


class PhotoLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
        QLabel {
            border: 4px dashed #aaa;
        }''')

    def setPixmap(self, *args, **kwargs):
        super().setPixmap(*args, **kwargs)
        self.setStyleSheet(
            "QLabel" "{"
                "border: 2px solid black;"
            "}"
        )





class Template(QWidget):

    def __init__(self):
        super().__init__()
        self.resize(1300, 700)
        self.pbar = QProgressBar(self)
  
        # setting its geometry
        self.pbar.move(20,250)
        self.pbar.resize(200, 25)
        
        
        self.path = ''
        self.name = ''
        self.path1 = ''
        self.name1 = ''
        title = "Title of window"

        # set the title
        self.setWindowTitle('Face Recognition Attendance System')

        # setting  the geometry of window
        #self.setGeometry(0, 0, 1300, 800)

        # show all the widgets
        self.show()
        self.photo = PhotoLabel()
        self.photo1 = PhotoLabel(self)
        
        # self.btn = QtWidgets.QPushButton(self)
        # self.btn2 = QtWidgets.QPushButton(self)
        # self.btn3 = QtWidgets.QPushButton(self)
        # self.btn4 = QtWidgets.QPushButton(self)
        
        self.btn = QPushButton('Browse Image',self)
        self.btn2 = QPushButton('Browse Test Image',self)
        self.btn3 = QPushButton('Encode Image',self)
        self.btn4 = QPushButton('Get Attendance',self)
        self.btn.move(20,100)
        self.btn2.move(700,10)
        self.btn3.move(20,160)
        self.btn4.move(700,660)
        
        self.btn.clicked.connect(self.open_image)
        self.btn2.clicked.connect(self.open_image1)
        self.btn3.clicked.connect(self.upload)
        self.btn4.clicked.connect(self.detect)
        
        self.btn.show()
        self.btn2.show()
        self.btn3.show()
        self.btn4.show()
        
        # _translate = QtCore.QCoreApplication.translate
        # self.btn.setText(_translate(self, "Submit Name"))
        # self.btn2.setText(_translate(self, "Submit Name"))
        # self.btn3.setText(_translate(self, "Submit Name"))
        # self.btn4.setText(_translate(self, "Submit Name"))
        
        # self.initUI()
        #grid = QGridLayout(self)
        # grid.addWidget(btn, 0, 0, Qt.AlignHCenter)
        # grid.addWidget(btn2, 0, 1, Qt.AlignHCenter)
        # grid.addWidget(btn3, 1, 0, Qt.AlignHCenter)
        # grid.addWidget(btn4, 2, 1, Qt.AlignHCenter)
        self.photo1.setFixedWidth(1000) #height of image box
        self.photo1.setFixedHeight(600)
        #self.photo.setFixedWidth(250)
        #self.photo.setFixedHeight(250)
        #grid.addWidget(self.photo, 1, 0)
        #grid.addWidget(self.photo1, 1, 1)
        
        self.photo1.move(250,50)
        self.photo1.show()
        
        self.setAcceptDrops(True)
        self.resize(1300, 700)   #size of window
        
        self.label = QLabel('Enter Name of Person: ', self)
        self.label.move(3, 0)
        self.label.resize(200, 30)
        self.label.setFont(QFont('Segoe UI (TrueType)',10))
        self.label.show()
        
        self.label1 = QLabel('Status Bar: ', self)
        self.label1.move(20, 200)
        self.label1.resize(200, 30)
        self.label1.setFont(QFont('Segoe UI (TrueType)',10))
        self.label1.show()        
        
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 30)
        self.textbox.resize(100, 20)
        self.name = self.textbox.text()
        self.textbox.setText("")
        self.textbox.show()
        self.button = QPushButton('Submit Name', self)
        self.button.move(20, 60)

        # connect button to function on_click
        self.button.clicked.connect(self.getText)
        self.button.show()

    def getText(self):
        cnt = 0;gui.pbar.setValue(cnt);cnt += 1
        self.name = self.textbox.text()+'.jpg';gui.pbar.setValue(cnt);cnt += 1
        gui.pbar.setValue(100);cnt += 1
        #print(self.name)

    # def initUI(self):
        # Create textbox

        # Create a button in the window

        self.textbox.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            filename = event.mimeData().urls()[0].toLocalFile()
            event.accept()
            self.open_image(filename)
        else:
            event.ignore()

    def open_image(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
        self.path = filename
        self.label2 = QLabel(filename, self)
        self.label2.move(20, 110)
        self.label2.resize(200, 45)
        self.label2.setFont(QFont('Segoe UI (TrueType)',8))
        self.label2.show()  

    def open_image1(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
        self.path1 = filename
        self.photo1.setPixmap(QPixmap(filename).scaledToHeight(600))
    
    @staticmethod
    def test(filename):
        PhotoLabel.setPixmap(QPixmap(filename).scaledToHeight(600))

    def upload(self):
        if (self.path == ''):
            print('Image not uploaded')
        else:
            makeMenu(self.path, self.name, 1)

    def detect(self):
        if (self.path1 == ''):
            print('Image not uploaded')
        else:
            # obj = doWork()
            makeMenu(self.path1, self.name, 2)


# class doWork:



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Template()
    gui.show()
    sys.exit(app.exec_())
