import cv2
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
from matplotlib.figure import Figure



def face_mask_prediction(cropped_img):
    # Load face-mask prediction model
    model = tf.keras.models.load_model('models/model_cnn.hdf5', compile=False)
    img = cv2.resize(cropped_img, (32, 32))
    img = img.astype(np.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    predict = model.predict(img)
    predict = np.argmax(predict)
    return predict