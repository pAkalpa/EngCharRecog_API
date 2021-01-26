'''
/*
 * @Author: Pasindu Akalpa 
 * @Date: 2021-01-26 22:00:50 
 * @Last Modified by: Pasindu Akalpa
 * @Last Modified time: 2021-01-26 22:33:32
 */
'''

import numpy as np
import cv2
from flask import Flask, request, jsonify
import joblib
from PIL import Image

app = Flask(__name__)

model = joblib.load('src/app.src/English_Char_SVC.sav')

@app.route('/', methods=['POST'])
def feedModel():
    file = request.files['image']
    img = Image.open(file.stream)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array,(28,28))
    img_array = np.reshape(img_array,(1,784))
    prediction = np.array2string(model.predict(img_array)[0])
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5000,debug=False,threaded=True)   
