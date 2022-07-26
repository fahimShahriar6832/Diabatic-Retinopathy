import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import joblib
from streamlit_option_menu import option_menu

st.title('Welcome')

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabatic-Retinopathy',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)
    

if (selected == 'Diabatic-Retinopathy'):
    
        # page title
    st.title('Diabetes Retinopathy using ML')

    #st.header("Image Predictor")

    html_temp = """
    <div style="background-color:#00008B ;font-size:24px;padding:24px">
    <h2 style="color:white;text-align:center;"><b>What is Diabetic Retinopathy?</b></h2>
    <h3 style="color:white;text-align:center;">Diabetic retinopathy (DR) is an illness occurring in the eye due to increase in blood glucose level.Diabetic Retinopathy are diseases called after the parts of the body that are affected by an increase in blood glucose levels (retina of the eye is affected).</h3>

    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a fundas image")

    # Diabetes Prediction Page


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

        if result[0][0] > result[0][1]:
          st.write("Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][0]*100)))
        else:
          st.write("NO Diabetic Retinopathy [{:.2f}% accuracy]".format((result[0][1])*100))

