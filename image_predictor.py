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

<img src=”(data:image/jpeg;base64,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)”>

<h2 style="color:white;text-align:center;"><b>What is Diabetic Retinopathy?</b></h2>
<h3 style="color:white;text-align:center;">Diabetic retinopathy (DR) is an illness occurring in the eye due to increase in blood glucose level.</h3>

</div>
    """


st.markdown(html_temp, unsafe_allow_html=True)


if (selected == 'Fundas Image'):
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

