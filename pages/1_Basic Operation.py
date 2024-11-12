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


sharpened = False
blurred =False
Inverted = False
Resized = False
Rescale = False
resize_width = 0
resize_row = 0
rescale_slider =0.1

# Function to sharpen the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Function to blur the image
def blur_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

# Function to invert (negative) the image
def invert_image(image):
    inverted = cv2.bitwise_not(image)
    return inverted

# Function to resize the image
def resize_image(image, width, height):
    resized = cv2.resize(image, (width, height))
    return resized

# Function to rescale the image
def rescale_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    rescaled = cv2.resize(image, (width, height))
    return rescaled


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
    
    col1 = st.columns(3)
    with col1[0]:
        sharpened = st.checkbox("Sharpen Image")
    with col1[1]:
        blurred = st.checkbox("Blurred Image")
    with col1[2]:
         Inverted = st.checkbox("Inverted Image")
    with col1[0]:
         Resized = st.checkbox("Resized Image")
    with col1[1]:
         Rescale = st.checkbox("Rescale Image")
         
    if Resized:
        resize_width =  st.number_input("Resize width",step=1,value=500)
        resize_row =  st.number_input("Resize Height",step=1,value=500)
        
    if Rescale:
        rescale_slider =st.slider("Rescale Slider", 0, 100, 100)
        # rescale_slider= rescale_slider//100
        print(rescale_slider)
       
       
    
    # Invert the image (Negative)
    # inverted = invert_image(CV2image)
    # st.image(inverted, caption="Inverted Image")

    # Resize the image (example 100x100)
    # resized = resize_image(CV2image, 100, 100)
    # st.image(resized, caption="Resized Image (100x100)")

    # Rescale the image (example 0.5 scale factor)
    # rescaled = rescale_image(CV2image, 0.5)
    # print(image_flip_list)
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
        
        
        if sharpened:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            CV2image = cv2.filter2D(CV2image, -1, kernel)
            
        if blurred:
             CV2image = cv2.GaussianBlur(CV2image, (5, 5), 0)
            
        if Inverted:
            CV2image = cv2.bitwise_not(CV2image)
        if Resized:
            CV2image = cv2.resize(CV2image, (resize_width, resize_row))
            
        if Rescale:
            width = int(CV2image.shape[1] * rescale_slider)
            height = int(CV2image.shape[0] * rescale_slider)
            rescaled = cv2.resize(CV2image, (width, height))
            
            
            
        st.image(CV2image, caption='Modifide Image', use_column_width=0)
        st.image(CV2image_copy, caption='Original Image', use_column_width=0)
       
        
        _, buffer = cv2.imencode('.jpg', CV2image)
        lowered_contrast_image_bytes = buffer.tobytes()

        st.download_button(
            label="Download updated Image",
            data=lowered_contrast_image_bytes,
            file_name=".jpg",
            mime="image/jpeg"
        )

       
        
        
            
     
        



# Upload image

