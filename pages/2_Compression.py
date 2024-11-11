# Import the necessary libraries
import streamlit as st
import numpy as np
import cv2
import pandas as pd
st.title("Image Compression")
import os
from PIL import Image

uploaded_file = st.file_uploader("Choose an image or drag and drop it here", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode the image using OpenCV
    CV2image = cv2.imdecode(file_bytes, 1)
    CV2imageorg = cv2.cvtColor(CV2image, cv2.COLOR_BGR2RGB)
    CV2imageorg_real = CV2imageorg.copy()
    CV2image_resize = cv2.resize(CV2imageorg, (500,500))
colm = st.columns(2)

with colm[0]:
    if uploaded_file is not None:
        # Open the image
        Color_Selectd = st.radio(
            "Select Radio button for see value in matrix",
            [":red[Red]", ":green[Green]", ":blue[Blue]"],horizontal=1
            
        )
        resol = CV2imageorg.shape
        temp_resol = (resol[1],resol[0],resol[2])
        st.write("The Resolution of an image ", temp_resol)
        colm2 = st.columns(2)
        with colm2[0]:
            row_number = st.number_input("Insert row number",step=1,min_value=0,max_value=resol[1]-1)
            st.write("The Row number is ", row_number)
        with colm2[1]:
            colm_number = st.number_input("Insert Colm number",step=1,min_value=0,max_value=resol[0]-1)
            st.write("The Colm number is ", colm_number)
        # print(CV2imageorg)
        Dict1 ={}
        if Color_Selectd == ":red[Red]":
            set_color_no = 0
        elif Color_Selectd == ":green[Green]":
             set_color_no = 1
        else:
            set_color_no = 2
            
            
        _20_rows = row_number+20
        
        if _20_rows>resol[1]:
            _20_rows = resol[1]
        # print(row_number,_20_rows,resol[1])
        for i in range(row_number,_20_rows):
            try:
                Dict1[i]= [CV2imageorg[j][i][set_color_no] for j in range(colm_number,colm_number+10)]
            except:
                Dict1[i]= [CV2imageorg[j][i][set_color_no] for j in range(colm_number,resol[0])]



        # # 
        df = pd.DataFrame.from_dict(Dict1)

       # print(df)

        st.table(df)

        

        
    
with colm[1]:
    if uploaded_file is not None:
        st.write("On left hand side the value of matrix is the value of image pixel of color")
        
        
        CV2imageorg1 = cv2.rectangle(CV2imageorg,pt1=(row_number,colm_number),pt2=(row_number+20,colm_number+20),color = (0, 0, 255),thickness = 1)
        CV2image_resize1 = cv2.resize(CV2imageorg1, (500,500))
        CV2image_copy = CV2image.copy()
        st.image(CV2image_resize1, caption='Imported Image', use_column_width=0)
        
st.divider()

_3_clumn = st.columns(3)
caption_for_three_image = ["Blue color only","Red color only","Green color only"]
for i in range(3):
    with _3_clumn[i]:
        if uploaded_file is not None:
            seprate_color_image = CV2image_resize.copy()
            seprate_color_image[:,:,i]=0
            seprate_color_image[:,:,(i+1)%3]=0
            st.image(seprate_color_image, caption=caption_for_three_image[i], use_column_width=0)
                
st.divider()               
if uploaded_file is not None:
    st.write("If the size of each pixel is 8bit i.e. 1 BYTE")
    _2_columns = st.columns(2)
    with _2_columns[0]:
        
            st.write(f"Then the size should be {resol[1]} x {resol[0]} x {resol[2]}  =  ",round(resol[0]*resol[1]*resol[2]/(1024),2) , " KILOBYTE")
    with _2_columns[0]:
            file_size_bytes =   uploaded_file.size
          
            st.write(f"But actual size is   =  ",round(file_size_bytes/(1024),2) , " KILOBYTE")
st.divider() 


if uploaded_file is not None:
    st.header("RGB to YCbCr")
    CV2image_resize_small = cv2.resize(CV2imageorg_real, (250,250))
    _0_5_col = st.columns(5)
    
    _3_clumn1 = st.columns(3)
    caption_for_three_image = ["Blue","Red","Green"]
    for i in range(3):
        with _3_clumn1[i]:
            if uploaded_file is not None:
                seprate_color_image = CV2image_resize_small.copy()
                seprate_color_image[:,:,i]=0
                seprate_color_image[:,:,(i+1)%3]=0
                st.image(seprate_color_image, caption=caption_for_three_image[i], use_column_width=0)
    _3_clumn2 = st.columns(3)
    image_ycrcb = cv2.cvtColor(CV2image_resize_small, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(image_ycrcb)
    ybr_image = [Y, Cr, Cb]
    caption_for_YBR_three_image = [" Luminance"," Chrominance Blue (Cb)"," Chrominance red (Cr)"]
    for i in range(3):
        with _3_clumn1[i]:
                if uploaded_file is not None:
                    st.image(ybr_image[i], caption=caption_for_YBR_three_image[i], use_column_width=0)
                    
    st.header("Croma Subsampling")
    
    st.divider() 
    st.header("contrast sensitivity vs spatial frequency plot")
    
    col_CON_VS_SPA = st.columns(2) 
    with col_CON_VS_SPA[1]:
            img = Image.open("IMAGES/CON_VS_SPA.png")
            st.image(img, caption='Uploaded Image', use_column_width='auto')
    
    st.divider() 
    st.header("contrast sensitivity vs spatial frequency plot")
    
    col_CON_VS_SPA = st.columns(2) 
    with col_CON_VS_SPA[1]:
            img = Image.open("IMAGES/CON_VS_SPA_i.jpeg")
            img= img.resize((500,500))
            st.image(img, caption='Uploaded Image', use_column_width='auto')
            
    
    

    st.divider() 
    
    image_copy = CV2imageorg_real.copy()
    image_copy= cv2.resize(image_copy,(1000,1000))
    # image_copy
    for i in range(100):
        for j in range(100):
            image_copy = cv2.line(image_copy,pt1=(i*10,0),pt2=(i*10,1000),color=(0,0,0),thickness=2)
            image_copy = cv2.line(image_copy,pt1=(0,i*10),pt2=(1000,i*10),color=(0,0,0),thickness=2)
    
    st.image(image_copy, caption='Imported Image', use_column_width=0)
    st.divider()
    
    img = Image.open("IMAGES/DCT.png")
    img= img.resize((500,500))
    st.image(img, caption='Uploaded Image', use_column_width='auto')
            