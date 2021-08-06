import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Location Image Classifier")
st.text("Provide URL of Location Image for image classification")
st.text("Make sure you get the url right so that the app will not produce an error and let it end with jpg")
@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('models/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
st.sidebar.info('This app is created to predict buildings, forest, glacier, ,mountain,, sea, street')
def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = tf.image.resize(img,[150,150])
  return np.expand_dims(img, axis=0)

#path = st.text_input('Enter Image URL to Classify.. ')
path = st.text_input('Enter Image URL to Classify.. ','https://github.com/osabobo/stock_price/blob/master/5.jpg?raw=true')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    if st.button("Predict"):
        label =model.predict(decode_img(content))
        st.write(classes[np.argmax(label)])
        st.write("")
        image = Image.open(BytesIO(content))
        st.image(image, caption='Classifying Image', use_column_width=True)
