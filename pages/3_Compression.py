# Import the necessary libraries
import streamlit as st
import numpy as np
import cv2
import pandas as pd
st.title("Image Compression")
import os
from PIL import Image
import io
from PIL import Image
import streamlit as st
import io

def jpeg_compression(input_image, compression_percentage):
    # Ensure compression percentage is between 0 and 100

    # Convert the compression percentage to JPEG quality level
    # quality = max(10, 100 - compression_percentage * 2)  # Limit quality to 10 for high compression
    quality = max(5, 100 - compression_percentage * 3)  # More aggressive compression, lower quality
    
    # Compress and save image to bytes
    img_byte_arr = io.BytesIO()
    # Compress and save image to bytes
    img_byte_arr = io.BytesIO()
    input_image.save(img_byte_arr, format="JPEG", quality=quality)
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

# Streamlit app to test the compression
st.title("JPEG Compression Based on Compression Percentage")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
compression_percentage = st.slider("Select compression percentage:", 0, 100, 50)

if uploaded_file is not None:
    # Open the uploaded image
    input_image = Image.open(uploaded_file)
    
    # Get the size of the original image in bytes
    original_image_bytes = uploaded_file.getvalue()
    original_size = len(original_image_bytes)//1000
    st.write(f"Original Image Size: {original_size} Kilobytes")
    
    # Apply compression
    compressed_image_bytes = jpeg_compression(input_image, compression_percentage)
    compressed_size = len(compressed_image_bytes)//1000
    st.write(f"Compressed Image Size: {compressed_size} Kilobytes")
    
    # Display original and compressed images
    
    col1 = st.columns(2)
    
    with col1[0]:
        st.image(input_image, caption="Original Image")
    
    with col1[1]:
        compressed_image = Image.open(io.BytesIO(compressed_image_bytes))
        st.image(compressed_image, caption="Compressed Image")

    # Add a download button for the compressed image
    st.download_button(
        label="Download Compressed Image",
        data=compressed_image_bytes,
        file_name="compressed_image.jpg",
        mime="image/jpeg"
    )




st.title("Study of JPEG Compression ")

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
    st.subheader("If the size of each pixel is 8bit i.e. 1 BYTE")
    _2_columns = st.columns(2)
    with _2_columns[0]:
        
            st.subheader(f"Then the size should be {resol[1]} x {resol[0]} x {resol[2]}  =  " +str(round(resol[0]*resol[1]*resol[2]/(1024),2) )+ " KILOBYTE")
    with _2_columns[0]:
            file_size_bytes =   uploaded_file.size
          
            st.subheader(f"But actual size is   =  "+str(round(file_size_bytes/(1024),2)) + " KILOBYTE")
st.divider() 


if uploaded_file is not None:
    st.title("Step 1: Input image is divided into a small block which is having 8x8 Pixel dimensions")
    image_copy = CV2imageorg_real.copy()
    image_copy= cv2.resize(image_copy,(1000,1000))
    # image_copy
    for i in range(100):
        for j in range(100):
            image_copy = cv2.line(image_copy,pt1=(i*10,0),pt2=(i*10,1000),color=(0,0,0),thickness=2)
            image_copy = cv2.line(image_copy,pt1=(0,i*10),pt2=(1000,i*10),color=(0,0,0),thickness=2)
    
    st.image(image_copy, caption='Imported Image', use_column_width=0)
    st.divider()
    st.title("Step 1: Transform Color")
   
    
    
    
    
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
    st.write(
    """
    ## What is Chroma Subsampling?
    
    Chroma Subsampling is a technique used in image and video compression that reduces the resolution of the color information (chrominance) while keeping the brightness information (luminance) at full resolution. This exploits the fact that the human eye is more sensitive to brightness than to color, allowing significant compression without a noticeable loss in image quality.

    ### Color Space Conversion
    Chroma subsampling works in the **YCbCr** color space:
    - **Y (Luminance)**: Represents brightness or intensity.
    - **Cb and Cr (Chrominance)**: Represent the color components (blue and red differences).

    ### Common Chroma Subsampling Ratios
    - **4:4:4**: No subsampling. All components (Y, Cb, Cr) have full resolution. Highest quality, but large file size.
    - **4:2:2**: Chrominance channels (Cb, Cr) are sampled at half the resolution horizontally. The luminance channel (Y) retains full resolution.
    - **4:2:0**: Chrominance channels (Cb, Cr) are sampled at half the resolution both horizontally and vertically. This is the most common format for video and image compression, offering high compression at the cost of some color detail.

    ### Why Chroma Subsampling?
    - **Human Vision Sensitivity**: The human eye is less sensitive to color detail compared to brightness, making this technique effective for reducing file size without significant visual degradation.
    - **File Size Reduction**: Chroma subsampling helps compress images and videos, making them easier to store and transmit.

 
    ### Visual Quality vs Compression
    - **4:4:4**: No loss in color detail, but the largest file size.
    - **4:2:2**: Good color fidelity with reduced file size.
    - **4:2:0**: Maximum compression with slight loss in color quality, often not noticeable to viewers.

    ### Summary
    Chroma Subsampling is a crucial technique for reducing file sizes in image and video compression. By reducing color information while retaining brightness at full resolution, it ensures efficient compression with minimal perceptual loss in quality.
    """
)
    st.image("IMAGES/croma.png")
    
    st.divider() 
   
    
    st.title("2D Discrete Cosine Transform (DCT)")

    # Introduction text
    st.write("""
    The **2D Discrete Cosine Transform (DCT)** is a technique used to transform an image from the spatial domain to the frequency domain. It is widely used in image compression algorithms, such as JPEG, where it helps represent an image in terms of its frequency components. 

    In the 2D DCT, the image is divided into smaller blocks (often 8x8 pixels), and the DCT is applied first along the rows, then along the columns. This results in a set of DCT coefficients that represent the contributions of different frequency components of the image.

    The formula for the 2D DCT of an image block is given by:

    """)

    # LaTeX code for 2D DCT formula
    dct_formula = r"""
    F(u, v) = \alpha(u) \alpha(v) \sum_{x=0}^{N-1} \sum_{y=0}^{M-1} f(x, y) 
    \cos\left[ \frac{\pi (2x + 1)u}{2N} \right] \cos\left[ \frac{\pi (2y + 1)v}{2M} \right]
    """

    # Display the LaTeX formula in Streamlit
    st.latex(dct_formula)

    # Further explanation
    st.latex(r"""
    \text{Where:}
    \quad
    - f(x, y) \text{ is the pixel value at position } (x, y) \text{ in the image block.}
    \quad""")
    st.latex(r"""
    - F(u, v) \text{ is the DCT coefficient at frequency } u \text{ and } v.
    \quad """)
    st.latex(r"""
    - \alpha(u) \text{ and } \alpha(v) \text{ are normalization factors, ensuring proper scaling of the coefficients.}
    """)
    st.header("contrast sensitivity vs spatial frequency plot")
    
    col_CON_VS_SPA = st.columns(2) 
    with col_CON_VS_SPA[1]:
            img = Image.open("IMAGES/CON_VS_SPA_i.jpeg")
            img= img.resize((500,500))
            st.image(img)
    
    

    # st.divider() 
    
   
    col16 = st.columns(2)
    
    with col16[0]:
        img = Image.open("IMAGES/DCT.png")
        img= img.resize((500,500))
    
        st.image(img, use_column_width=0)
    with col16[1]:
        img = Image.open("IMAGES/qunt.png")
        img= img.resize((500,500))
    
        st.image(img, use_column_width=0)
    
    st.write("DCT is done by comparing them with an 8x8, 64-frequency pattern where spatial frequency increases from left to right and top to bottom.")
    st.write("This process converts an image from its frequency components, converting an 8x8 block where each pixel represents a brightness level into another level where each block represents the presence of a particular frequency component.")
    img = Image.open("IMAGES/zigzag.jpg")
    img= img.resize((500,500))
    
    st.image(img, use_column_width=0)
    # import streamlit as st

    # Title
    st.title("Image Compression: Quantization and Encoding")

    # Quantization section
    st.header(" Quantization")
    st.write("""
    **Purpose**: Reduce the precision of the DCT coefficients to save storage space.

    **Process**: Quantization compresses the data by rounding off less significant (usually high-frequency) DCT coefficients. This step introduces some loss of information, but because the human eye is less sensitive to high-frequency details, the visual quality of the image remains mostly intact.

    **Implementation**: In JPEG, a quantization matrix is applied to the DCT-transformed block. The matrix has lower values in the top-left (low-frequency) area and higher values in the bottom-right (high-frequency) area. Each DCT coefficient is divided by its corresponding quantization matrix value and then rounded, causing many high-frequency coefficients to become zero.
    """)

    # LaTeX for Quantization formula
    st.latex(r"""
    Q(u, v) = \text{round}\left( \frac{F(u, v)}{\text{Quantization Matrix}(u, v)} \right)
    """)

    st.latex(r"""
Q(u, v) = \text{round}\left( \frac{F(u, v)}{\text{Quantization Matrix}(u, v)} \right)
""")

    # Encoding section
    st.header(" Encoding (Compression)")
    st.write("""
    **Purpose**: Further compress the quantized data using a lossless encoding algorithm.

    **Process**:
    - **Zig-Zag Scanning**: The quantized 8x8 block is scanned in a zig-zag order, which prioritizes lower-frequency coefficients first and clusters the zero values at the end.
    - **Run-Length Encoding (RLE)**: The sequence is encoded using RLE, which compresses consecutive zeroes efficiently.
    - **Huffman or Arithmetic Encoding**: The RLE output is further encoded using Huffman or arithmetic encoding to minimize the file size.

    **Outcome**: This step produces a compact binary representation of the image data, achieving the final compression.
    """)

    
    
   

    
    st.title("Steps to Read a JPEG Image and Convert to RGB")
    # Title
    
    img = Image.open("IMAGES/FOWRD.png")
    # img= img.resize((500,500))

    st.image(img, use_column_width=0)

    img = Image.open("IMAGES/BACK.png")
    # img= img.resize((500,500))

    st.image(img, use_column_width=0)

    # Step 1: File Decoding and Decompression
    st.header("1. File Decoding and Decompression")
    st.write("""
    The JPEG file is stored in a compressed format that contains information about how the image should look, but not the actual pixel values in a straightforward way.

    When you read a JPEG file, the first step is to **decode** this compressed data. This includes:
    - **Reconstructing the frequency coefficients** that were created during the Discrete Cosine Transform (DCT) step.
    - **Performing inverse quantization** on these coefficients to get back the approximate frequency components of the original image.
    - **Applying the Inverse DCT (IDCT)** to transform these frequency components back into the spatial domain (pixel values).
    """)

    # Step 2: Rebuilding the Color Channels
    st.header("2. Rebuilding the Color Channels")
    st.write("""
    JPEG often uses the **YCbCr color space** instead of RGB, which separates the brightness (Y) from the color information (Cb and Cr).

    After decompression, the image data is still in this YCbCr format. To display the image as RGB, the software needs to **convert the YCbCr channels back to RGB**.
    """)

    # Step 3: Combining Channels into an RGB Image
    st.header("3. Combining Channels into an RGB Image")
    st.write("""
    The **Y, Cb, and Cr channels** are then combined to create the final RGB image. Each pixel is represented as a triplet of **(R, G, B)** values, ready for display or further processing.
    """)


