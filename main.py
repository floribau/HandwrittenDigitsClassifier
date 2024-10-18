import os
import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('handwritten-digits-nn.keras')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The digit in digit{image_number}.png is probably a {np.argmax(prediction)}")
    image_number += 1
