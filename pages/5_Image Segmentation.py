# Import the necessary libraries
import streamlit as st

st.set_page_config(layout="wide")
st.title("Image Segmentation")
import streamlit as st
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image



uploaded_file = st.file_uploader("Choose an image or drag and drop it here", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    st.write("Image Segmetation Using Threholding")
   
    # Decode the image using OpenCV
    img = cv2.imdecode(file_bytes, 1)

    # Convert the image from BGR (OpenCV format) to RGB (for display purposes)
    CV2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    CV2image = cv2.resize(CV2image, (500,500))
    CV2image_copy = CV2image.copy()
    gray_image = cv2.cvtColor(CV2image, cv2.COLOR_RGB2GRAY)
    _, thresholded = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    col0 = st.columns(2)
    
    with col0[0]:
        n1 =st.number_input("Enter val 1",value=128, min_value=0, max_value=255)
        n2 =st.number_input("Enter val 2",value=255, min_value=n1+1, max_value=255)
        _, thresholded = cv2.threshold(gray_image, n1, n2, cv2.THRESH_BINARY)
    
    with col0[1]:
        st.image(thresholded, caption='Modifide Image', use_column_width=0)
        st.image(CV2image_copy, caption='Original Image', use_column_width=0)
        

   
        

