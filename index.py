from flask import Flask, flash, redirect, url_for, render_template, request, jsonify
import pickle
import numpy as np
from od import face_mask_prediction
import os

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PREDICT_FOLDER'] = 'static/predicts/'

@app.route('/', methods=['POST', 'GET'])
def form():
    if request.method == 'POST':
        if request.files:
            file = request.files['formFile']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            print('upload_image filename: ' + filename)
            return render_template('result.html', filename=filename)
    else:
        return render_template('form.html')

@app.route('/display_img/<filename>')
def display_pred_img(filename):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    raw_filename = filename
    filename = 'static/uploads/' + filename
    sample_img = filename
    img = cv2.imread(sample_img)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)
    for (x,y,w,h) in faces:
        xmin, ymin, w, h = x, y, w, h
        cropped_img = img[ymin:ymin+w, xmin:xmin+h]
        prediction = face_mask_prediction(cropped_img)
        edgecolor = 'r'
        img_label = 'without_mask'
        if prediction == 0:
            edgecolor = 'b'
            img_label = 'mask_weared_incorrect'
        if prediction == 1:
            edgecolor = 'g'
            img_label = 'with_mask'
        rectangle = patches.Rectangle([xmin, ymin], w, h, linewidth=2, edgecolor=edgecolor, facecolor='none')
        rx, ry = rectangle.get_xy()
        axis.annotate(img_label, (rx, ry), color=edgecolor, weight='bold', fontsize=10, ha='left', va='baseline')
        axis.add_patch(rectangle)
    axis.set_axis_off()
    axis.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig.savefig('static/predicts/' + raw_filename)
    
    return redirect(url_for('static', filename='/predicts/' + raw_filename), code=301)

if __name__ == '__main__':
    debug_mode = True
    app.run(debug=debug_mode)