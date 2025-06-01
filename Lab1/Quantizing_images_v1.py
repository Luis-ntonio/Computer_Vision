import cv2
import numpy as np
import matplotlib.pyplot as plt


def quantize_image(image, n_colors):
    #Convert image to black and white
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert the image to a 2D numpy array
    image = np.array(image, dtype=np.float64) / 255

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 1)

    #Divide sections for n_colors
    n = image.max() / n_colors
    for i in range(n_colors):
        pixels = np.where((pixels >= n * i) & (pixels <= n * (i + 1)), i, pixels)
    print(pixels)
    # Replace the pixels in the image
    quantized_image = pixels.reshape(image.shape)

    return quantized_image


def main(image):
    # Load the image
    image = cv2.imread(image)

    # Quantize the image
    quantized_image = quantize_image(image, 16)
    
    # Plot distribution of quantized image
    plt.hist(quantized_image.ravel(), 256, [0, 20])
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