


from PIL import Image
import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
   
    layout="wide",  # Use 'wide' to increase the app's width
    
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
    .stImage > img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px; /* Adjust as needed */
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Project Title and Topic
st.markdown('<h1 class="title">Image Processing Techniques</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advance Algo Project</p>', unsafe_allow_html=True)

# Team Members
st.markdown('<p class="section-header">Master of Technology</p>', unsafe_allow_html=True)
st.markdown('<p class="section-header"> In </p>', unsafe_allow_html=True)
st.markdown('<p class="section-header"> COMPUTER SCIENCE AND INFORMATION TECHNOLOGY </p>', unsafe_allow_html=True)
col0 =  st.columns(5)
with col0[2]:
    st.image("logo.png")  # Replace "logo.png" with the actual filename or URL
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
   st.image("IMAGES/tushar.png")
   st.markdown('<p class="section-header1">Tushar Singh</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header2">24CSM2R21</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header3">MTECH CSIS</p>', unsafe_allow_html=True)
   
   

with col1[4]:
   st.image("IMAGES/adarsh.png")
   st.markdown('<p class="section-header1">Aashish Vishwakarma</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header2">24CSM2R01</p>', unsafe_allow_html=True)
   st.markdown('<p class="section-header3">MTECH CSIS</p>', unsafe_allow_html=True)
   