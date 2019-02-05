import sys
import os

import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, Response, jsonify
import numpy as np 
import json
import tensorflow as tf

from ssd_keras.predict import Predict
from resources import plotbox

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=['GET', 'POST'])
def output():
    return render_template('index.html')


@app.route("/upload_image", methods=["POST"])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if 'image' in file.content_type:
            print('ok image')
            f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            img = os.path.join('static/images/', file.filename)
            file.save(img)
            print(file.filename)
            filename = 'box88_' + file.filename
            print(filename)
            image_path = os.path.join('static/images/', filename)
            result = predict(img, image_path)
            result = result[0].tolist()
            # plotbox.plotBoxToImage(result, img, image_path)
            # img = os.path.join('static/images/plot2.png')

            # return json.dumps({'status': 'OK', 'data':result, 'img_path': image_path})
            resp = jsonify(result)
            resp.status_code = 200
            return resp

        else:
            print('not ok image')
            return json.dumps({'data': 'false', 'err':'NOT_IMG'})
    else:
        print('error')
        return json.dumps({'data':'false'})

# @app.route("/predict", methods=["POST", "GET"])

def predict(image_pat, img_pth):
    img = plt.imread(image_pat)
    predict = Predict(image=img, image_path=img_pth)
    result = predict.predict()
    
    return result


@app.route('/hello', methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':

        dataset = request.values
        dataset = dataset.to_dict(flat=False)
        print(dataset)
        print(len(dataset))
        print(dataset["img"])
        print(type(dataset))

        return 'hello post'
    else:
        return 'hello get'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
