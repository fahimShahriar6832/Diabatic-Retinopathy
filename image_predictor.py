import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import joblib

st.header("Image Predictor")

html_temp = """
<div style="background-color:#00008B ;padding:10px">
<h2 style="color:white;text-align:center;"><b>What is Diabetic Retinopathy?</b></h2>
<h3 style="color:white;text-align:center;">Diabetic retinopathy (DR) is an illness occurring in the eye due to increase in blood glucose level.Diabetic Retinopathy are diseases called after the parts of the body that are affected by an increase in blood glucose levels (retina of the eye is affected).</h3>

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
    st.image(im, caption='Query Image')
    
# load model
loaded_model = load_model('St_DR_MobileNet.h5')

result = loaded_model.predict(im)


 color1 = st.color_picker('try', '#1aa3ff',key=1)
 st.write(f"again{color1}")

st.subheader('Prediction Percentages:') 


st.write("Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][0]*100)))
st.write("NO Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][1])*100))

