import tensorflow as tf
import gradio as gr
import numpy as np

# model 불러오기
my_model = tf.keras.models.load_model('/Users/pablo/Documents/GitHub/machine_learning/Tensorflow/my_model.h5')

def test_digit(image):
    image = image.reshape(1, 28, 28, 1)
    image2 = image / 255.0
    prediction = my_model.predict(image2)
    pred=np.argmax(prediction)
    return pred

gr.Interface(
    fn = test_digit,
    inputs = "sketchpad",
    outputs = "label",
    title = "MNIST TEST"
    ).launch(debug=True)