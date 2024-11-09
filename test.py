from oct2py import Oct2Py
import numpy as np
import matplotlib.pyplot as plt

# Initialize Oct2Py
oc = Oct2Py()

# Read image using Octave's imread function
image_path = 'anime.jpg'  # Replace with the path to your image file
image_data = oc.imread(image_path)

# Convert the image to a numpy array if needed (oct2py automatically converts it to numpy)
image_array = np.array(image_data)

# Display the image using matplotlib
plt.imshow(image_array)
plt.axis('off')  # Hide axes
plt.show()

# Close Oct2Py instance
oc.exit()
