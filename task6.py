import cv2
import numpy as np
from matplotlib import pyplot as plt

def convolve(image, kernel):
    """Perform convolution on a grayscale image with a given kernel."""
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    output = np.zeros((img_h, img_w), dtype="float32")

    for y in range(img_h):
        for x in range(img_w):
            roi = padded[y:y + k_h, x:x + k_w] # AI used for this formula
            output[y, x] = np.sum(roi * kernel)

    # Clip values to 0–255
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output

def main():
    image_path = "images/classroom.png"
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file not found. Place 'sample.jpg' in the same folder.")
    
    # TO DO: Define a simple 5x5 averaging kernel (all ones / 9)
    kernel = np.ones((5, 5)) * (1/9)

    # Define a simple 3x3 sobel operators  
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
        
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    # Apply convolution 
    # (IN REALITY IS CROSS CORRELATION, we are not flipping the kernel)
    blurred = convolve(image, kernel)

    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    edges = np.sqrt(grad_x.astype(np.float32)**2 + grad_y.astype(np.float32)**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # TO DO: Display results (original and blurred)
    image = cv2.resize(image, dsize = (0,0), fx = 0.4, fy = 0.4)
    grad_x = cv2.resize(grad_x, dsize = (0,0), fx = 0.4, fy = 0.4)
    grad_y = cv2.resize(grad_y, dsize = (0,0), fx = 0.4, fy = 0.4)
    edges = cv2.resize(edges, dsize = (0,0), fx = 0.4, fy = 0.4)
    
    top = np.hstack([image, grad_x])
    bottom = np.hstack([grad_y, edges])
    grid = np.vstack([top, bottom])

    cv2.putText(grid, "Original", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(grid, "Sobel x", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(grid, "Sobel y", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(grid, "Combined", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Results Sobel Edge Detection', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
