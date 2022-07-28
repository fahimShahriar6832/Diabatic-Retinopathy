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

html_temp = """
<div style="background-color:#00008B ;font-size:24px;padding:24px">
<h3 style="color:white;text-align:center;">Fundus photographs are ocular documentation that record the appearance of a patient's retina. Optometrists, ophthalmologists, orthoptists and other trained medical professionals use fundus photography for monitoring the progression of certain eye condition/diseases.</h3>

</div>
"""

st.markdown(html_temp, unsafe_allow_html=True)


if (selected == 'Fundas Image'):
    
    html_temp = """
    <div style="background-color:#00008B ;font-size:24px;padding:24px">
    <h2 style="color:white;text-align:center;"><b>What is Diabetic Retinopathy?</b></h2>
    <h3 style="color:white;text-align:center;">Diabetic retinopathy (DR) is an illness occurring in the eye due to increase in blood glucose level.What does a fundus photograph show? It's able to provide a picture of the retina, the retinal vasculature (blood vessels), and the optic nerve head, where retinal blood vessels enter the eye. It can also show drusen, abnormal bleeding, scar tissue, and areas of atrophy.</h3>

    </div>
    """


    #st.header("Image Predictor")

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

        
        
        
        
if (selected == 'OCT Image'):
    st.header("Upload a OCT image")

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

