# Import the necessary libraries
import streamlit as st

st.set_page_config(layout="wide")
st.title("Image Segmentation")
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Helper function to convert image to a downloadable format
def convert_image(img):
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Header and description
st.title('Enhanced KMeans Image Segmentation')
st.write('Upload an image, select the number of clusters, and get a segmented output.')

# Image uploader
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Load image
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Display the original image
    st.image(image_np, caption="Original Image", use_column_width=True)

    # Preprocessing options (resize the image to speed up clustering if it's too large)
    max_size = st.slider("Max image dimension (resize to speed up processing)", 100, 1000, 500)
    if max(image_np.shape) > max_size:
        scale_factor = max_size / max(image_np.shape)
        new_size = (int(image_np.shape[1] * scale_factor), int(image_np.shape[0] * scale_factor))
        image_np = cv2.resize(image_np, new_size, interpolation=cv2.INTER_AREA)

    st.write(f"Image resized to: {image_np.shape}")

    # Option to apply image filters (optional preprocessing)
    apply_filter = st.checkbox("Apply Gaussian Blur (optional)", value=False)
    if apply_filter:
        image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Reshape the image into a 2D array of pixels
    pixels = image_np.reshape(-1, 3)

    # Choose the number of clusters (with slider)
    k = st.slider("Select number of clusters (regions)", 2, 20, 4)

    # Option to calculate an optimal K (Elbow method suggestion)
    optimal_k = st.checkbox("Suggest optimal number of clusters (Elbow method)", value=False)

    if optimal_k:
        distortions = []
        K = range(2, 11)
        for k_value in K:
            kmeans_model = KMeans(n_clusters=k_value, random_state=0).fit(pixels)
            distortions.append(kmeans_model.inertia_)

        # Plot the elbow curve
        plt.figure(figsize=(6, 4))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title('Elbow Method For Optimal k')
        st.pyplot(plt)
        st.write("The elbow point suggests the optimal number of clusters.")

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)

    # Replace each pixel with the center of its cluster
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image_np.shape)
    segmented_img = segmented_img.astype(np.uint8)

    # Post-processing option to smoothen the result (optional)
    smoothen = st.checkbox("Apply post-processing (Bilateral Filter)", value=False)
    if smoothen:
        segmented_img = cv2.bilateralFilter(segmented_img, 9, 75, 75)

    # Display segmented image
    st.image(segmented_img, caption=f'Segmented Image with {k} clusters', use_column_width=True)

    # Option to download the segmented image
    segmented_image_data = convert_image(segmented_img)
    st.download_button(
        label="Download Segmented Image",
        data=segmented_image_data,
        file_name="segmented_image.png",
        mime="image/png"
    )

else:
    st.write("Please upload an image to start the segmentation process.")
