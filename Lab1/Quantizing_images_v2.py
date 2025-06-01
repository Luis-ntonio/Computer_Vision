import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def quantize_image(image, n_colors):
    #Convert image to black and white
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert the image to a 2D numpy array
    image = np.array(image, dtype=np.float64) / 255

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 1)

    # Fit the model
    model = KMeans(n_clusters=n_colors)
    labels = model.fit_predict(pixels)

    # Get the colors of the clusters
    colors = model.cluster_centers_

    # Replace the pixels in the image with the colors of the clusters
    quantized_image = colors[labels].reshape(image.shape)

    return quantized_image


def main(image):
    # Load the image
    image = cv2.imread(image)

    # Quantize the image
    quantized_image = quantize_image(image, 64)

    plt.hist(quantized_image.ravel(), 256, [0, 1])
    plt.show()
    
    # Display the original and quantized images
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(quantized_image, cmap='gray')
    plt.title('Quantized Image')
    plt.axis('off')

    plt.show()

main("./image.png")