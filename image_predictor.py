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



if (selected == 'Fundas Image'):

    html_temp = """
    <div style="background-color:#00008B ;font-size:24px;padding:24px">
    <h2 style="color:white;text-align:center;"><b>What is meant by fundus images?</b></h2>
    <h3 style="color:white;text-align:center;">Fundus imaging is defined as the process whereby reflected light is used to form a two dimensional representation of the three dimensional retina, the semi-transparent, layered tissue lining the interior of the eye projected onto an imaging plane </h3>

    </div>
        """
    #st.markdown(html_temp, unsafe_allow_html=True)

    image = Image.open('Fundus_Image.jpeg')
    st.image(image, caption='Fundus_Image')
    image_2 = Image.open('Diabatic.jpg')
    st.image(image_2, caption='Fundus_Image')
    
    
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
    html_temp = """
    <div style="background-color:#00008B ;font-size:24px;padding:24px">
    <h2 style="color:white;text-align:center;"><b>What is an OCT image?</b></h2>
    <h3 style="color:white;text-align:center;">Optical coherence tomography (OCT) is an emerging technology for performing high-resolution cross-sectional imaging. OCT is analogous to ultrasound imaging, except that it uses light instead of sound. OCT can provide cross-sectional images of tissue structure on the micron scale in situ and in real time.</h3>

    </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
   
    image_2 = Image.open('image.jpg')
    st.image(image_2, caption='OCT_Image')
    
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

