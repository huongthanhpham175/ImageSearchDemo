import os.path

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import urllib.request
from tensorflow import keras
from .ClassName import class_names


np.os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
new_model = tf.keras.models.load_model(os.path.abspath('webs/static/model/my_model.h5'))



class TestModel:
    def __init__(self):
        self.src = "";
    def predict(self):
        image_input = Image.open(self.src)
        # Check its architecture
        # new_model.summary()
        size = (224, 224)
        image = ImageOps.fit(image_input, size, Image.ANTIALIAS)
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        # result = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
        result = "{}".format(
            class_names[np.argmax(score)])
        print(result)
        return result
