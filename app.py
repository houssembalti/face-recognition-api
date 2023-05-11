from flask import Flask, jsonify, request
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
import base64
from flask_cors import CORS
from io import BytesIO

app = Flask(__name__)
CORS(app)
path = "C:/Users/Houssem/Desktop/FaceRecognitionAPI/face-recognition-api/images"
images = []
classNames = []
personsList = os.listdir("C:/Users/Houssem/Desktop/FaceRecognitionAPI/face-recognition-api/images")

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodeings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodeings(images)
print('Encoding Complete.')


@app.route('/face-recognition/num-faces', methods=['POST'])
def fr():
    file = request.form.get('image')
    file = file[23:]
    files=base64.b64decode(file)


    img = np.array(Image.open(BytesIO(files)))

    img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(img)
    encodeCurentFrame = face_recognition.face_encodings(img, faceCurentFrame)

    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name.lower())



    return name

#@app.route('/register',methods=['POST'])
#def register():
 #   file = request.form.get('image')

  #  files = base64.b64decode(file)

