# Import the necessary libraries
import streamlit as st

st.set_page_config(layout="wide")

from PIL import Image
from PIL import Image
import cv2
import numpy as np

st.title("Image Noise")
Mean_GN = 0
signa_GN = 25
Mean_SN = 0
signa_SN = 25

# Divide the window into two columns
cols = st.columns(2)
cols = st.columns([6, 4])  # 80% and 20%
Salt_Prob = 0
Paper_Prob = 0

def add_speckle_noise(image, mean=0, var=0.1):
    """
    Add speckle noise to an image.
    
    Args:
    image (numpy array): Input image.
    mean (float): Mean of the speckle noise (typically 0).
    var (float): Variance of the noise. The higher the value, the more noise.
    
    Returns:
    noisy_image (numpy array): Image with speckle noise.
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean/100, var ** 0.1, (row, col, ch))  # Gaussian noise
    noisy_image = image + image * gauss
    
    # Clip the values to stay in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    
    return noisy_image.astype(np.uint8)



def add_salt_and_pepper_noise(img, salt_prob, pepper_prob):
      # Getting the dimensions of the image 
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    row , col = img.shape 
 
    number_of_pixels = 100* salt_prob
    for i in range(number_of_pixels): 
        y_coord=np.random.randint(0, row - 1) 
        x_coord=np.random.randint(0, col - 1) 
        img[y_coord][x_coord] = 255

    
    number_of_pixels = 100* pepper_prob
    for i in range(number_of_pixels): 
        y_coord=np.random.randint(0, row - 1) 
        x_coord=np.random.randint(0, col - 1) 
        img[y_coord][x_coord] = 0

    return img 



def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    print(noise)
    noisy_image = cv2.add(image, noise)
    return noisy_image
gaussian_noise = False
salt_and_pepper_noise = False
Speckle_Noise = False


with cols[0]:
    # Drag and drop or file uploader for images
    uploaded_file = st.file_uploader("Choose an image or drag and drop it here", type=["jpg", "jpeg", "png"])
    if not Speckle_Noise and not salt_and_pepper_noise:
        gaussian_noise =  st.checkbox("Gaussian Noise")
    if not Speckle_Noise and not gaussian_noise:
        salt_and_pepper_noise =  st.checkbox("Salt and Pepper Noise")
    if not salt_and_pepper_noise and not gaussian_noise:
        Speckle_Noise =  st.checkbox("Speckle Noise")
    
    if Speckle_Noise:
        Mean_SN = st.slider("Mean for Speckle",0,100,0 )
        signa_SN =  st.slider("Variance for Speckle",0,100,25 )
    
    if gaussian_noise:
        st.latex(r'''
                N(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
            ''')

        Mean_GN = st.slider("Mean",0,100,0 )
        signa_GN =  st.slider("Sigma",0,100,25 )
    if salt_and_pepper_noise:
        Salt_Prob = st.slider("Salt Probability",0,100,50 )
        Paper_Prob =  st.slider("Paper Probability",0,100,50 )
        
    # Add a line between the two columns
    st.markdown("<hr style='border:1px solid black;'>", unsafe_allow_html=True)

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
        
        if Speckle_Noise:
             CV2image = add_speckle_noise(CV2image, mean=Mean_SN, var=signa_SN)
            

        if gaussian_noise:
            CV2image = add_gaussian_noise(CV2image,mean=Mean_GN,sigma=signa_GN)
        
        if salt_and_pepper_noise:
            CV2image = add_salt_and_pepper_noise(CV2image,salt_prob=Salt_Prob,pepper_prob=Paper_Prob)
        
            
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
