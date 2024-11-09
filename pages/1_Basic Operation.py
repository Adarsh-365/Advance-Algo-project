# Import the necessary libraries
import streamlit as st

st.set_page_config(layout="wide")
from PIL import Image
import streamlit as st
from PIL import Image
import cv2
import numpy as np
st.title("BASIC OPERATION")

# Divide the window into two columns
cols = st.columns(2)
cols = st.columns([6, 4])  # 80% and 20%
Greyscale = False
Red_color = False
Green_color = False
Blue_color = False
image_flip = False

with cols[0]:
    # Drag and drop or file uploader for images
    uploaded_file = st.file_uploader("Choose an image or drag and drop it here", type=["jpg", "jpeg", "png"])

    # Add a line between the two columns
    st.markdown("<hr style='border:1px solid black;'>", unsafe_allow_html=True)
    
    
    Greyscale = st.checkbox("Greyscale")
    colorcols1 = st.columns(3,gap="Large")
    
    if not Greyscale:
        with colorcols1[0]:
            Red_color = st.checkbox("Red")
        with colorcols1[1]:
            Green_color = st.checkbox("Green")
       
    

        with colorcols1[2]:
            Blue_color = st.checkbox("Blue")
   
    
        
    colorcols = st.columns(3,gap="Large")
    
    if not Greyscale:
    
        with colorcols[0]:
            Red_color_number =  st.slider("Red intensity", 0, 255, 25)
        with colorcols[1]:
            Green_color_number = st.slider("Green intensity", 0, 255, 25)
        

        with colorcols[2]:
            Blue_color_number =  st.slider("Blue intensity", 0, 255, 25)
    
    image_flip_list = st.multiselect(
            "Flip an Image",
                [ "DOWN","SIDE", "SIDE DOWN"],
            
        )
    print(image_flip_list)
    if len(image_flip_list)>0:
        image_flip = True 

    

with cols[1]:
    if uploaded_file is not None:
        # Open the image
       
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
        # Decode the image using OpenCV
        img = cv2.imdecode(file_bytes, 1)

        # Convert the image from BGR (OpenCV format) to RGB (for display purposes)
        CV2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        CV2image = cv2.resize(CV2image, (500,500))
        CV2image_copy = CV2image.copy()
        
        
        if Blue_color:
            CV2image[:,:,2]=Blue_color_number
        
        if Green_color:
            CV2image[:,:,1]=Green_color_number
            
        if Red_color:
            CV2image[:,:,0]=Red_color_number
        
        if image_flip:
            if "DOWN" in image_flip_list:
                CV2image = cv2.flip(CV2image,0)
            if "SIDE" in image_flip_list:
                CV2image = cv2.flip(CV2image,1)
            if "SIDE DOWN" in image_flip_list:
                CV2image = cv2.flip(CV2image,-1)
         
            
        
        
        if Greyscale:
            CV2image = cv2.cvtColor(CV2image,cv2.COLOR_BGR2GRAY)
        st.image(CV2image, caption='Modifide Image', use_column_width=0)

       
        
        
            
     
        st.image(CV2image_copy, caption='Original Image', use_column_width=0)
       

   
        
