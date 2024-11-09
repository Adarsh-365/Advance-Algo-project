import streamlit as st
import json
import cv2
import numpy as np
import pandas as pd
st.set_page_config(layout="wide")
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt


# Create and start the listener thread








st.title("Image Enhancement")
for i in range(20):
             st.write("")

st.header("A COMPARATIVE STUDY OF HISTOGRAM EQUALIZATION BASED IMAGE ENHANCEMENT TECHNIQUES FOR BRIGHTNESS PRESERVATION AND CONTRAST ENHANCEMENT",divider="red")
for i in range(25):
    st.write("")




st.subheader("Histogram Equalization is a contrast enhancement technique in the image processing which uses the histogram of image")

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the Lottie animation from a local JSON file
lottie_animation = load_lottie_file("anime/q1.json")

st.markdown("""
    <style>
    .right-align {
        text-align: right;
        margin-left: 0px;  /* Adjust as needed */
    }
    </style>
    """, unsafe_allow_html=True)


col1 = st.columns(2)
with col1[0]:
    for i in range(25):
        st.write("")
    st.markdown('<h2 class="right-align">So What is Image Contrast ??</h2>', unsafe_allow_html=True)


with col1[1]:
    # with st.echo():
        for i in range(20):
             st.write("")
        st_lottie(lottie_animation,width=200,height=200,)
        

for i in range(20):
             st.write("")
st.markdown('<h3> Image contrast is the difference in brightness between the light and dark areas of an image. Its the amount of differentiation in color or grayscale between different parts of an image</h3> ', unsafe_allow_html=True)

# E

for i in range(20):
             st.write("")
uploaded_file = st.file_uploader("Choose an image or drag and drop it here", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode the image using OpenCV
    CV2image = cv2.imdecode(file_bytes, 1)
    CV2imageorg = cv2.cvtColor(CV2image, cv2.COLOR_BGR2RGB)
    CV2image_resize1 = cv2.resize(CV2imageorg, (300,300))
    contrast_factor = 0.5  # 1.0 means original contrast, lower than 1.0 reduces contrast

    # Adjust the contrast
    
    col1 =  st.columns(3)
    with col1[0]:
        contrast_factor = st.slider("Contrast %",min_value=0,max_value=200,value=100)
    with col1[1]:
        st.image(CV2image_resize1, caption='Original Image', use_column_width='auto')
    with col1[2]:
        lowered_contrast_image = cv2.convertScaleAbs(CV2image_resize1, alpha=contrast_factor/100, beta=0)

        st.image(lowered_contrast_image, caption='Modified Image', use_column_width='auto')
        
    
    
    _, buffer = cv2.imencode('.jpg', lowered_contrast_image)
    lowered_contrast_image_bytes = buffer.tobytes()

    st.download_button(
        label="Download updated Image",
        data=lowered_contrast_image_bytes,
        file_name=".jpg",
        mime="image/jpeg"
    )
    
    
    for i in range(20):
        st.write("")
    
    col2 = st.columns(2)
    CV2image_resize1 = cv2.resize(CV2imageorg, (500,500))
   
    CV2image_resize1 = cv2.cvtColor(CV2image_resize1,cv2.COLOR_BGR2GRAY)
    
    with col2[0]:
        resol = CV2image_resize1.shape
        temp_resol = (resol[1],resol[0])
        st.write("The Resolution of an image ", temp_resol)
        colm2 = st.columns(2)
        with colm2[0]:
            row_number = st.number_input("Insert col number",step=1,min_value=0,max_value=resol[1]-1)
            st.write("The Row number is ", row_number)
        with colm2[1]:
            colm_number = st.number_input("Insert row number",step=1,min_value=0,max_value=resol[0]-1)
            st.write("The Colm number is ", colm_number)
        # print(CV2imageorg)
        Dict1 ={}
        
            
            
        _20_rows = row_number+20
        
        if _20_rows>resol[1]:
            _20_rows = resol[1]
        # print(row_number,_20_rows,resol[1])
        for i in range(row_number,_20_rows):
          
            
            try:
                Dict1[i]= [CV2image_resize1[j][i] for j in range(colm_number,colm_number+10)]
            except:
                Dict1[i]= [CV2image_resize1[j][i] for j in range(colm_number,resol[0])]



        # # 
        df = pd.DataFrame.from_dict(Dict1)

       # print(df)

        st.table(df)
    with col2[1]:
        st.markdown('<h3> Convert image to grayscale </h3> ', unsafe_allow_html=True)
        CV2image_resize12 =CV2image_resize1.copy()
        CV2image_resize12 = cv2.rectangle(CV2image_resize12,pt1=(row_number,colm_number),pt2=(row_number+20,colm_number+20),color = (255, 255, 255),thickness = 1)
        st.image(CV2image_resize12, caption='Modifide Image', use_column_width=0)
    

    for i in range(20):
        st.write("")
        
    
    
   

    
# Streamlit app
    start_cal = st.button("Start Calculation")
    if start_cal:
        st.title("Histogram of image")

        # Generate x values from 0 to 255
        x_values = np.arange(256)

        y_values = [0 for i in range(256)]
    
        y_val = []
        matrix = [[0 for i in range(500)] for i in range(500)]
        for i in range(500):
            for j in range(500):
                try:
                    y_values[int(CV2image_resize1[i][j])] +=1
                    y_val.append(CV2image_resize1[i][j])
                    matrix[i][j] =CV2image_resize1[i][j]
                
                except:
                    pass
                
                
                    
                
        
                

        # Create the bar graph
        plt.figure(figsize=(6, 3))
        # plt.bar(x_values, y_values, color='blue')
        plt.hist(y_val, bins=255, alpha=0.9, color='pink', edgecolor='black')


        # Set the title and labels
        plt.title('Histogram of grey Image')
        plt.xlabel('X Grey shade (0-255)')
        plt.ylabel('Y Grey shade count')

        # Show the grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Display the plot in Streamlit
        st.pyplot(plt)
        
        
        
    
        small_df = {}
        small_matrix=[]
        small_matrix_= []
        for i in range(0,500):
            
                    small_df[i]= [CV2image_resize1[j][i] for j in range(0,500)]
                    small_matrix.append([CV2image_resize1[j][i] for j in range(0,500)])
                
        df = pd.DataFrame.from_dict(small_df)
    
        # print(df)
        
        # st.table(df)
        # print(CV2image_resize1[20][20])
        uniqu_df ={}
        matrix_flat = np.array(small_matrix).flatten()
        for ui in np.unique(matrix_flat):
            uniqu_df[ui] =0 
            for val in matrix_flat:
                if ui==val:
                    uniqu_df[ui] +=1 
        
        st.table(uniqu_df)
        
       
        
        
        
        minimum = min(matrix_flat)
        maximum = max(matrix_flat)
        HISTOGRAM_EQU = {}
        lsit_of_hist = [i for i in range(minimum,maximum+1)]
        HISTOGRAM_EQU["Grey_Level"] = lsit_of_hist
        grey_count = [0 for i in range(minimum,maximum+1)]
        for value in matrix_flat:
            grey_count[value-minimum]+=1
        
        pdf =[i/sum(grey_count) for i in grey_count]
        
        HISTOGRAM_EQU["No of pixel"] =grey_count
        HISTOGRAM_EQU["PDF"] =pdf
        
        cdf = [pdf[0]]
        for i in range(1,len(pdf)):
            cdf.append(pdf[i]+cdf[i-1])
        
        HISTOGRAM_EQU["CDF"] =cdf
        HISTOGRAM_EQU["Sk*255"] =[i*maximum for i in cdf]
        hist_eqlise = [round(i*maximum) for i in cdf]
        HISTOGRAM_EQU["Hist equl"] = hist_eqlise
        
        dfq = pd.DataFrame.from_dict(HISTOGRAM_EQU)
        swaf_dict = {}
        for i in range(len(grey_count)):
            swaf_dict[lsit_of_hist[i]] =hist_eqlise[i] 
        #
        st.table(dfq)
        matrix_flat2 =  matrix_flat.copy()
        
        
        for i in range(len(matrix_flat2)):
            
                matrix_flat2[i]=swaf_dict[matrix_flat2[i]]
                
        # matrix_flat2.reshape(10, 10) 
        #matrix_flat2
        
        matrix_flat2c = matrix_flat2.reshape(500,500)
        # matrix_flat2 = np.flip(matrix_flat2)
        fig, ax = plt.subplots()
        matrix_flat2c =cv2.rotate(matrix_flat2c,0)
        matrix_flat2c =cv2.rotate(matrix_flat2c,1)
        # matrix_flat2c =cv2.rotate(matrix_flat2c,-1)
        for i in range(len(matrix_flat2c)):
            matrix_flat2c[i] = np.flip( matrix_flat2c[i])
        col3 = st.columns(2)
        with col3[0]:
              matrix_flat2c = cv2.cvtColor(matrix_flat2c,cv2.COLOR_GRAY2RGB)
              matrix_flat2c =cv2.rotate(matrix_flat2c,1)
              
              st.image(matrix_flat2c, caption="Modified image", channels="RGB")
        with col3[1]:
            st.image(CV2image_resize1, caption='Original Image', use_column_width='auto')

        # Display the plot in Streamlit
        
        
        plt.figure(figsize=(6, 3))
        # plt.bar(x_values, y_values, color='blue')
        plt.hist(matrix_flat2, bins=255, alpha=0.9, color='pink', edgecolor='black')


        # Set the title and labels
        plt.title('Histogram of grey Image')
        plt.xlabel('X Grey shade (0-255)')
        plt.ylabel('Y Grey shade count')

        # Show the grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Display the plot in Streamlit
        st.pyplot(plt)
        
        
        
        
        
        