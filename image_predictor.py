import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import joblib
from streamlit_option_menu import option_menu
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Diabatic-Retinopathy',
                          
                          ['Fundas Image',
                           'OCT Image'],
                          icons=['activity','heart'],
                          default_index=0)
st.title('Diabetes Retinopathy using ML')  
st.write("Diabetic retinopathy (DR) is an illness occurring in the eye due to increase in blood glucose level.")
st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)


if (selected == 'Fundas Image'):
    image = Image.open('Fundus_Image.jpeg')
    st.image(image, caption='Fundus_Image')
    
    st.header("Upload a Fundus Image")
    #st.header("Image Predictor")

    uploaded_file = st.file_uploader("")

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

        
        
        
        
if (selected == 'OCT Image'):
    image = Image.open('oct.jpg')
    st.image(image, caption='OCT_Image')
    st.header("Upload a OCT Image")

    uploaded_file = st.file_uploader("")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        im = img.resize((224,224))
        im = np.array(im)
        im = im/255
        im = np.expand_dims(im,axis=0)
        st.image(im, caption='Query Image')

        # load model
        loaded_model = load_model('oct_MobileNet_1.h5')

        result = loaded_model.predict(im)

        if (result[0][0] > result[0][1]) and (result[0][0] > result[0][2]) and (result[0][0] > result[0][3]) :
             st.write("NORMAL [{:.2f}% accuracy]".format((result[0][0]*100)))
        elif (result[0][1] > result[0][0]) and (result[0][1] > result[0][2]) and (result[0][1] > result[0][3]) :
             st.write("CNV [{:.2f}% accuracy]".format((result[0][1]*100)))
        elif (result[0][2] > result[0][1]) and (result[0][2] > result[0][0]) and (result[0][2] > result[0][3]) :
             st.write("DME [{:.2f}% accuracy]".format((result[0][2]*100)))
        elif (result[0][3] > result[0][1]) and (result[0][3] > result[0][2]) and (result[0][3] > result[0][0]) :
             st.write("DRUSEN [{:.2f}% accuracy]".format((result[0][3]*100)))

