from flask import Flask, request, jsonify
import logging.config
import telepot
import matplotlib
import random
matplotlib.use('TkAgg')
import tensorflow as tf
import os
import time
import cv2
from keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

global graph, model
graph = tf.get_default_graph()

logging.config.fileConfig('logconfig.ini')

detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = 'trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

bot = telepot.Bot('Input your bot key')
IMG_PATH = 'pic_server_side/'


def predict(image_path):
    # loading images
    start = time.time()
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
    with graph.as_default():
        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                rgb_face = cv2.resize(rgb_face, (gender_target_size))
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
            rgb_face = preprocess_input(rgb_face, False)
            rgb_face = np.expand_dims(rgb_face, 0)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion_text = emotion_labels[emotion_label_arg]

            if gender_text == gender_labels[0]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, gender_text, color, 0, -10, 1, 2)
            draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -40, 1, 2)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(bgr_image, (1024, 769))
    file_path = 'result/' + image_path.split('/')[-1]
    cv2.imwrite(file_path, resized_image)
    end = time.time()
    logging.info('[Time] for prediction(): {}'.format(end - start))
    return file_path


def get_filename(chat_id):
    time_stamp = time.strftime('%Y%m%d%H%M%S') + str(random.randint(1, 100))
    return str(chat_id) + '_' + time_stamp + '.png'


app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def classify():
    start_all = time.time()
    if request.method == 'POST':
        data = request.get_json()
        file_id = data['file_id']
        chat_id = data['chat_id']
        logging.info('Request data from chat_id : {}'.format(chat_id))
        img_name = get_filename(chat_id)
        file_name = IMG_PATH + img_name
        start_download = time.time()
        bot.download_file(file_id, file_name)
        end_download = time.time()
        logging.info('[Time] download_from_telegram: {}'.format(end_download - start_download))
        logging.info('Request data from chat_id : {}'.format(chat_id))
        logging.info('The pid & ppid : {}, {}'.format(os.getpid(), os.getppid()))
        bot.sendPhoto(chat_id, photo=open(predict(file_name), 'rb'))
        end_all = time.time()
        logging.info('[Time] overall: {}'.format(end_all - start_all))
        return jsonify({'success': 1})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=9001)
    # gunicorn -w 4 app_image:app -b localhost:8000
