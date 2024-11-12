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

def possion_noise(image):
    
            # image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)  # Read in grayscale for simplicity
            img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
            # Generate Poisson noise based on image intensity
            poisson_noise = np.random.poisson(image / 255.0 * 60) / 60 * 255  # Adjust intensity by multiplying with a scalar

            # Convert to uint8 to match the original image format
            noisy_image = image + poisson_noise.astype(np.uint8)
            return noisy_image

gaussian_noise = False
salt_and_pepper_noise = False
Speckle_Noise = False
possion_noise_var = False


with cols[0]:
    # Drag and drop or file uploader for images
    uploaded_file = st.file_uploader("Choose an image or drag and drop it here", type=["jpg", "jpeg", "png"])
    if not Speckle_Noise and not salt_and_pepper_noise:
        gaussian_noise =  st.checkbox("Gaussian Noise")
    if not Speckle_Noise and not gaussian_noise:
        salt_and_pepper_noise =  st.checkbox("Salt and Pepper Noise")
    if not salt_and_pepper_noise and not gaussian_noise:
        Speckle_Noise =  st.checkbox("Speckle Noise")
    possion_noise_var = st.checkbox("possion noise")
    
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
        
        
        if possion_noise_var:
              CV2image = possion_noise(CV2image)
        

            
        st.image(CV2image, caption='Modifide Image', use_column_width=0)
        st.image(CV2image_copy, caption='Original Image', use_column_width=0)
        
        rgb_image = cv2.cvtColor(CV2image,cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', rgb_image)
        lowered_contrast_image_bytes = buffer.tobytes()

        st.download_button(
            label="Download updated Image",
            data=lowered_contrast_image_bytes,
            file_name=".jpg",
            mime="image/jpeg"
        )




import numpy as np

# Load an example image (you can replace this with any image you want to test)
# In a real use case, allow the user to upload their own image for testing
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image")

    # --- 1. Gaussian Blurring ---
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    st.image(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB), caption="Gaussian Blurring")
    st.markdown("""
    **Gaussian Blurring** is effective for reducing Gaussian noise, a common noise pattern in digital images.
    It applies a Gaussian kernel, which creates a weighted average of surrounding pixels, 
    reducing high-frequency noise. Gaussian blur is best used when noise follows a bell-curve distribution.
    """)

    # --- 2. Median Blurring ---
    median_blur = cv2.medianBlur(image, 5)
    st.image(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB), caption="Median Blurring")
    st.markdown("""
    **Median Blurring** is effective for removing salt-and-pepper noise, where individual pixels appear 
    very bright or dark. It replaces each pixel with the median of neighboring pixels, which preserves edges 
    better than Gaussian blurring, making it suitable for high-contrast noise.
    """)

    # --- 3. Bilateral Filtering ---
    bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
    st.image(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB), caption="Bilateral Filtering")
    st.markdown("""
    **Bilateral Filtering** reduces noise while preserving edges, making it ideal for images where edges 
    are important. It takes into account both spatial closeness and intensity differences, selectively 
    blurring pixels to keep edges sharp.
    """)

    # --- 4. Non-Local Means Denoising ---
    non_local_means = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    st.image(cv2.cvtColor(non_local_means, cv2.COLOR_BGR2RGB), caption="Non-Local Means Denoising")
    st.markdown("""
    **Non-Local Means Denoising** uses a weighted average of pixels across the image based on similarity. 
    This technique is effective for images with Gaussian noise and is more computationally intensive, 
    but it often produces cleaner results with fewer artifacts.
    """)

    # --- 5. Morphological Opening ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphological_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    st.image(cv2.cvtColor(morphological_opening, cv2.COLOR_BGR2RGB), caption="Morphological Opening")
    st.markdown("""
    **Morphological Opening** is a two-step operation combining erosion and dilation. 
    It's especially useful for reducing noise in binary or thresholded images, 
    where small noise artifacts can be removed while maintaining the overall structure.
    """)

else:
    st.warning("Please upload an image to see the noise removal techniques in action.")
