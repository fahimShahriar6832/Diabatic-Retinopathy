import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import joblib

#st.header("Image Predictor")

html_temp = """
<div style="background-color:#00008B ;padding:10px">
<h2 style="color:white;text-align:center;">Image Predictor</h2>

</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a fundas image")


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    im = img.resize((224,224))
    im = np.array(im)
    im = im/255
    im = np.expand_dims(im,axis=0)
    st.image(img, caption='Query Image')
    
# load model
loaded_model = load_model('St_DR_MobileNet.h5')

result = loaded_model.predict(im)

st.subheader('Prediction Percentages:') 
st.write("Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][0]*100)))
st.write("NO Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][1])*100))

