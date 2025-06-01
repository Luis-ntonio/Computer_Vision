import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def update_image(val):
    brightness = cv2.getTrackbarPos('Brightness', 'Interactive Image')
    contrast = cv2.getTrackbarPos('Contrast', 'Interactive Image')

    # Adjust the image using the trackbar values (remains in BGR)
    adjusted = cv2.convertScaleAbs(image, alpha=contrast/50, beta=brightness-50)

    # Resize the adjusted image to reduce its size
    scale_percent = 50  # Adjust this percentage to control the size
    width = int(adjusted.shape[1] * scale_percent / 100)
    height = int(adjusted.shape[0] * scale_percent / 100)
    resized_adjusted = cv2.resize(adjusted, (width, height), interpolation=cv2.INTER_AREA)

    # Create a matplotlib figure for the histogram
    fig = Figure(figsize=(5, 3))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # Plot the histogram (you can specify a color if desired)
    ax.hist(resized_adjusted.ravel(), 256, [0, 256], color='blue')
    ax.set_title('Histogram')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.axis('tight')

    # Render the histogram to a numpy array (this is in RGB)
    canvas.draw()
    histogram_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    histogram_image = histogram_image.reshape(canvas.get_width_height()[::-1] + (3,))

    # Convert the histogram image from RGB to BGR for correct display with OpenCV
    histogram_image = cv2.cvtColor(histogram_image, cv2.COLOR_RGB2BGR)

    # Combine the resized image and histogram side by side
    combined_height = max(resized_adjusted.shape[0], histogram_image.shape[0])
    combined_width = resized_adjusted.shape[1] + histogram_image.shape[1]
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Place the resized image (BGR) and histogram (converted to BGR) side by side
    combined_image[:resized_adjusted.shape[0], :resized_adjusted.shape[1]] = resized_adjusted
    combined_image[:histogram_image.shape[0], resized_adjusted.shape[1]:] = histogram_image

    # Display the combined image
    cv2.imshow('Interactive Image', combined_image)

# Load the image (remains in BGR)
image_path = '../Gato-apropiado.jpg'  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Create a window for the trackbars
cv2.namedWindow('Interactive Image')

# Create trackbars for brightness and contrast
cv2.createTrackbar('Brightness', 'Interactive Image', 50, 100, update_image)
cv2.createTrackbar('Contrast', 'Interactive Image', 50, 100, update_image)

# Display the initial image and histogram
update_image(0)

# Wait until the user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()
