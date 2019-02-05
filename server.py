import sys
import os
import datetime
import matplotlib.pyplot as plt
import base64
import io
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


# @app.route("/upload_image", methods=["POST"])
# def upload():
#     if request.method == 'POST':
#         file = request.files['file']
#
#         if 'image' in file.content_type:
#             print('ok image')
#             f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             img = os.path.join('static/images/', file.filename)
#             file.save(img)
#             print(file.filename)
#             filename = 'box88_' + file.filename
#             print(filename)
#             image_path = os.path.join('static/images/', filename)
#             result = predict(img, image_path)
#             result = result[0].tolist()
#             # plotbox.plotBoxToImage(result, img, image_path)
#             # img = os.path.join('static/images/plot2.png')
#
#             # return json.dumps({'status': 'OK', 'data':result, 'img_path': image_path})
#             resp = jsonify(result)
#             resp.status_code = 200
#             return resp
#
#         else:
#             print('not ok image')
#             return json.dumps({'data': 'false', 'err':'NOT_IMG'})
#     else:
#         print('error')
#         return json.dumps({'data':'false'})
#
#
# @app.route("/predict", methods=["POST"])
# def predict2():
#     data = {}
#     try:
#         data = request.get_json()['data']
#
#     except Exception:
#         return jsonify(status_code='400', msg='bad request'), 400
#
#     print(data)
#     data = base64.b64decode(data)
#
#     image = io.BytesIO(data)
#
#     print(image)
#
#     predictions = [{'label': 'test', 'description': 'test', 'probability': 0.1 * 100.0}]
#
#     return 'xxxxxxx'
#    # return jsonify(predictions = predictions)
#
#
#
# def priedict(image_pat, img_pth):
#     img = plt.imread(image_pat)
#     predict = Predict(image=img, image_path=img_pth)
#     result = predict.predict()
#
#     return result

@app.route('/predict', methods=['POST'])
def predict():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month)
    day = str(now.day)
    temp = "{}_{}_{}".format(year, month, day)

    data = {}

    if request.method == 'POST':
        try:
            data = request.get_json()

        except Exception:
            return jsonify({'bad request': '400'})
    else:
        print("GET")

    image = base64.b64decode(data['data'])
    filename = temp
    with open('uploads/'+filename, 'wb') as f:
        f.write(image)

    predict = Predict(image=image, image_path=filename)
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
        now = datetime.datetime.now()
        year = str(now.year)
        month = str(now.month)
        day = str(now.day)
        temp = "{}_{}_{}".format(year, month, day)
        return temp


if __name__ == '__main__':
    app.run('0.0.0.0', '5010')