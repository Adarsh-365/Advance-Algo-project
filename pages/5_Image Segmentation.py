# Import the necessary libraries
import streamlit as st

st.set_page_config(layout="wide")
st.title("Image Segmentation")
import streamlit as st
import cv2
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

# st.sidebar.image: st.sidebar.image("logo.png", use_column_width=True) 


ch1 = st.checkbox("Threholding")

if ch1:

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


    st.divider()

ch2 = st.checkbox("watershad Algorithm")

if ch2:
    watershad_file = st.file_uploader("Choose an image or drag and drop it here for watershad", type=["jpg", "jpeg", "png"])
    if watershad_file is not None:
        
        st.sidebar.header("Parameter Adjustment")
        threshold_value = st.sidebar.number_input("Binary Threshold Value", min_value=0, max_value=255, value=120)
        # max_value = st.sidebar.number_input("Max Value for Threshold", min_value=0, max_value=255, value=255)
        dist_threshold = st.sidebar.number_input("Distance Transform Threshold", min_value=0, max_value=255, value=15)

        file_bytes = np.asarray(bytearray(watershad_file.read()), dtype=np.uint8)
        
        st.write("Image Segmetation Using Threholding")
    
        # Decode the image using OpenCV
        img = cv2.imdecode(file_bytes, 1)
        figsize = (6, 4)
    


        # img = cv2.imread("testcoin.jpg")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Streamlit app
        st.title("Image Processing Workflow")

            
        st.header("1. Grayscale Image Conversion")
        st.write("Convert the original image from RGB to grayscale for easier processing and analysis.")
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.imshow(img_gray, cmap="gray")
        # ax.set_title("Grayscale Image")
        # st.pyplot(fig)
        st.image(img_gray)

        st.header("2. Binary Inverse Thresholding")
        st.write("Thresholding is applied to convert the grayscale image to binary. Binary inverse makes the object white and background black.")
        _, threshold_img = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.imshow(threshold_img, cmap="gray")
        # ax.set_title("Binary Inverse Threshold")
        # st.pyplot(fig)
        st.image(threshold_img)

        st.header("3. Morphological Dilation")
        st.write("Dilation enlarges the boundaries of the object. This helps in connecting broken parts of the object in the binary image.")
        kernel = np.ones((3, 3), np.uint8)
        imp_dilate = cv2.morphologyEx(threshold_img, cv2.MORPH_DILATE, kernel)
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.imshow(imp_dilate, cmap="gray")
        # ax.set_title("Dilated Image")
        # st.pyplot(fig)
        st.image(imp_dilate)

        st.header("4. Distance Transform")
        st.write("Calculates the distance from each foreground pixel to the nearest background pixel. The center of objects will have higher distance values.")
        dist_transform = cv2.distanceTransform(imp_dilate, cv2.DIST_L2, 5)
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.imshow(dist_transform, cmap="gray")
        # ax.set_title("Distance Transform")
        # st.pyplot(fig)
        st.image(imp_dilate)

        st.header("5. Thresholded Distance Transform")
        st.write("Apply a threshold to the distance-transformed image to separate the foreground objects more clearly.")
        _, dist_thresh = cv2.threshold(dist_transform, dist_threshold, 255, cv2.THRESH_BINARY)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(dist_thresh, cmap="gray")
        ax.set_title("Thresholded Distance Transform")
        st.pyplot(fig)
        # st.image(dist_thresh)

        st.header("6. Connected Components")
        st.write("Label the connected components in the binary image, which assigns different labels to each unique object.")
        dist_thresh_uint8 = np.uint8(dist_thresh)
        _, labels = cv2.connectedComponents(dist_thresh_uint8)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(labels, cmap="nipy_spectral")
        ax.set_title("Connected Components")
        st.pyplot(fig)
        # st.image(labels)

        st.header("7. Watershed Segmentation")
        st.write("Use the watershed algorithm to separate overlapping objects. Boundaries are marked with red.")
        labels = np.int32(labels)
        labels = cv2.watershed(img_rgb, labels)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Slightly larger for the final side-by-side comparison
        ax1.imshow(labels, cmap="nipy_spectral")
        ax1.set_title("Watershed Labels")
        # st.image(labels)

        # Display the segmented image with boundaries
        img_rgb[labels == -1] = [255, 0, 0]  # Red boundary
        ax2.imshow(img_rgb)
        ax2.set_title("Segmented Image with Boundaries")
        st.pyplot(fig)
        st.image(img_rgb)
        
        
ch3 = st.checkbox("Contours")    
if ch3:
    counter_file = st.file_uploader("Choose an image or drag and drop it here for counter", type=["jpg", "jpeg", "png"])
    if counter_file is not None:
        
    
        file_bytes = np.asarray(bytearray(counter_file.read()), dtype=np.uint8)
        
        st.write("Image Segmetation Using Threholding")
    
        # Decode the image using OpenCV
        img = cv2.imdecode(file_bytes, 1)
        figsize = (6, 4)
    


        # img = cv2.imread("testcoin.jpg")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image_rgb = cv2.resize(image_rgb,(500,500))

        # Step 1: Original Image
        st.subheader("Step 1: Original Image")
        st.image(image_rgb, caption="Original Image")
        st.write("This is the original image loaded in BGR format and converted to RGB for display.")

        # Convert to grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Step 2: Grayscale Image
        st.subheader("Step 2: Grayscale Image")
        st.image(gray, caption="Grayscale Image", channels="GRAY")
        st.write("The image is converted to grayscale to simplify the data, making it easier to detect contours.")

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 3: Blurred Image
        st.subheader("Step 3: Blurred Image")
        st.image(blurred, caption="Blurred Image", channels="GRAY")
        st.write("Gaussian blur is applied to reduce noise, which helps in detecting more accurate edges.")

        # Use Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Step 4: Edge Detection
        st.subheader("Step 4: Edge Detection")
        st.image(edges, caption="Edges Detected", channels="GRAY")
        st.write("Canny edge detection is applied to highlight the edges in the image.")

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        contour_image = image_rgb.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Green color contours

        # Convert contour image to RGB for Streamlit display
        contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

        # Step 5: Contours
        st.subheader("Step 5: Contours")
        st.image(contour_image_rgb, caption="Contours on Original Image")
        st.write("Contours are drawn on the original image, highlighting the boundaries of objects detected.")

        # Summary
        st.subheader("Summary")
        st.write("""
        This Streamlit app demonstrates the process of image segmentation using contours:
        1. The original image is converted to grayscale.
        2. Gaussian blur is applied to reduce noise.
        3. Canny edge detection is used to detect edges in the image.
        4. Finally, contours are drawn to segment and highlight objects in the image.
        """)

ch4 = st.checkbox("K-means Clutering")  

if ch4:
    # Function to perform KMeans segmentation
    k = st.sidebar.slider("Number of Segments (k)", min_value=2, max_value=10, value=3)

    def kmeans_segmentation(image, k):
        # Convert image to RGB and reshape it
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape((-1, 3))
        
        # Convert pixels to float32 for kmeans
        pixels = np.float32(pixels)
        
        # KMeans criteria and apply KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers back to 8 bit values
        centers = np.uint8(centers)
        
        # Map labels to the centers to create the segmented image
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        
        return segmented_image

    # Streamlit app
    st.title('K-Means Image Segmentation')

    # Upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Number of clusters (segments)

    if uploaded_image is not None:
        # Open the image using PIL
        image = Image.open(uploaded_image)
        
        # Show original image
        
        
        # Apply KMeans segmentation
        segmented_image = kmeans_segmentation(image, k)
        
        # Display segmented image
        col1 = st.columns(2)
        
        with col1[0]:
            st.image(image, caption="Original Image")
        
        with col1[1]:
          st.image(segmented_image, caption=f"Segmented Image (k={k})")
        
        # Show additional info
        st.write(f"Image dimensions: {image.size}")
        st.write(f"Number of segments: {k}")