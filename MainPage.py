


from PIL import Image
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import base64
from pathlib import Path
# from utilities import lo

st.set_page_config(
   
    layout="wide",  # Use 'wide' to increase the app's width
    initial_sidebar_state="collapsed"
    
)

# Streamlit title
# st.title("Image PreProcesing Techniques Project")
st.sidebar.image: st.sidebar.image("logo.png", use_column_width=True) 

# Title Page with Styling
st.markdown(
    """
    <style>
    .h1head{
        font-size: 20px;
        font-weight: bold;
        color: #2e86de;
        text-align: center;
        
        
    }
    .title {
        
        font-weight: bold;
        color: #2e86de;
        text-align: center;
    }
    .subtitle {
        font-size: 24px;
        color: #555;
        text-align: center;
        margin-top: 0px;
    }
    
    .subtitle1 {
        font-size: 24px;
        color: #555;
        text-align: center;
        margin-top: -10px;
         font-style: italic;
    }
   
    .team {
        font-size: 20px;
        color: #333;
        text-align: center;
    }
    .section-header {
        font-size: 22px;
        color: #fff;
        text-align: center;
        margin-top: 10px;
    }
    .section-header1 {
        font-size: 20px;
        color: #fff;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0;
        
    }
    .section-header2 {
        font-size: 15px;
        color: #F3F3E0;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0;
        
    }
    .section-header3 {
        font-size: 15px;
        color: #CBDCEB;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0;
        
    }
    .img-fluid  {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px; /* Adjust as needed */
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Project Title and Topic
st.markdown('<h1 class="title">Image Processing Techniques</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advance Algorithms Project</p>', unsafe_allow_html=True)

# Team Members
st.markdown('<p class="section-header">Master of Technology</p>', unsafe_allow_html=True)
st.markdown('<p class="section-header"> In </p>', unsafe_allow_html=True)
st.markdown('<p class="section-header"> COMPUTER SCIENCE AND INFORMATION TECHNOLOGY </p>', unsafe_allow_html=True)


def img_to_bytes(img_path):
      img_bytes = Path(img_path).read_bytes()
      encoded = base64.b64encode(img_bytes).decode()
      return encoded

def img_to_html(img_path):
      img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
        img_to_bytes(img_path)
      )
      return img_html

st.markdown(img_to_html('logo.png'), unsafe_allow_html=True)
    

st.markdown('<p class="subtitle1"> Subject :- Advance Algorithms </p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle1"> Submitted to : Prof. Manish Kumar Bajpai </p>', unsafe_allow_html=True)
st.markdown('<p class="h1head"> DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING </p>', unsafe_allow_html=True)
st.markdown('<p class="h1head"> NATIONAL INSTITUTE OF TECHNOLOGY WARANGAL </p>', unsafe_allow_html=True)
st.markdown('<p class="h1head">November, 2024 </p>', unsafe_allow_html=True)

st.divider()


st.markdown('<h1 class="title">Group Member</p>', unsafe_allow_html=True)

col1 =  st.columns(5)

with col1[0]:
   st.image("IMAGES/adarsh.png")
   st.markdown('<p class="section-header1">Adarsh Tayde</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header2">24CSM2R03</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header3">MTECH CSIS</p>', unsafe_allow_html=True)
   
   

with col1[2]:
   st.image("IMAGES/Tushar.png")
   st.markdown('<p class="section-header1">Tushar Singh</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header2">24CSM2R21</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header3">MTECH CSIS</p>', unsafe_allow_html=True)
   
   

with col1[4]:
   st.image("IMAGES/ashish.jpg")
   st.markdown('<p class="section-header1">Aashish Vishwakarma</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header2">24CSM2R01</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header3">MTECH CSIS</p>', unsafe_allow_html=True)
   
   
st.divider()
# Title of the project
st.title("Technologies Used in Our Project")

# Technologies and their logos
technologies = {
    "Python": "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",
    "NumPy": "https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg",
    "Pandas": "https://pandas.pydata.org/static/img/pandas_mark.svg",
    "OpenCV": "https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg",
    "Streamlit": "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png",
    "Pillow": "https://python-pillow.org/assets/images/pillow-logo-248x250.png",
    "Matplotlib": "https://matplotlib.org/_static/images/logo2.svg"
}

libraries = [
    {"Python": "High-level programming language for versatile and dynamic application development."},
    {"NumPy": "Library for numerical computing, offering array objects and mathematical functions."},
    {"Pandas": "Data manipulation tool for structured data analysis and dataframes."},
    {"OpenCV": "Open-source computer vision library for real-time image and video processing."},
    {"Streamlit": "Framework for creating interactive web applications with Python scripts."},
    {"Pillow": "Image processing library, enabling opening, manipulation, and saving image files."},
    {"Matplotlib": "Plotting library for creating static, animated, and interactive visualizations."}
]

# Create two columns
col1, col2 = st.columns(2)

# List of technology names and their column assignment
tech_list = list(technologies.items())

# Display technologies in two columns
for i in range(len(tech_list)):
    tech, logo_url = tech_list[i]
    
    # Display even-indexed technologies in the first column
    with col1 if i % 2 == 0 else col2:
        st.image(logo_url, width=200)  # Display the logo
        st.write(f"**{tech}**") 
        st.write(libraries[i][tech])# Display the technology name
        st.write("This project leverages " + tech + " for various functionalities.")  # Description placeholder

# Optional: Add a summary
st.write("In this project, we have used various libraries and technologies to process, analyze, and visualize data effectively.")

