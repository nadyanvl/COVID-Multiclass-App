import tensorflow
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import efficientnet.tfkeras
from tensorflow.keras.models import load_model

# Load the trained model
model = tensorflow.keras.models.load_model('efficientnet_model.hdf5')

# Make the header for web app
st.write("""
         # COVID Multiclass Image Prediction
         """
         )
st.write("This is a simple image classification web app to predict COVID Multiclass CT Scans Image")
st.write("The type of Lungs CT Scan that can be predicted as follows: "
         "Covid, Healthy, Others")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

# processing function
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It has Covid!")
    elif np.argmax(prediction) == 1:
        st.write("It is Healthy!")
    else:
        st.write("It has others pulmonary directions!")
    
    st.text("Probability (0: Covid, 1: Healty, 2: Others")
    st.write(prediction)