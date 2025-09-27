import cv2
import numpy as np

def convolve(image, kernel):
    """Perform convolution on a grayscale image with a given kernel."""
    # TO DO: Get image and kernel dimensions
    # Hint: .shape
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape

    # TO DO: Pad the image with zeros around the border
    # Hint: use np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='')
    # Choose the type of padding to work with using the mode parameter
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # TO DO: Create an empty output image
    # Hint: matrix the same size of the current image
    output = np.zeros((img_h, img_w), dtype="float32")

    # TO DO: Loop over each pixel (y, x)
    # Extract the region of interest (ROI) from padded image
    # Multiply by kernel and sum up values
    # Assign to output[y, x]
    # Hint: np.sum(region * kernel)
    for y in range(img_h):
        for x in range(img_w):
            roi = padded[y:y + k_h, x:x + k_w] # AI used for this formula
            output[y, x] = np.sum(roi * kernel)

    # Clip values to 0–255
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output  

def gaussia_kernel(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def main():
    image_path = "images/flower.png"
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file not found. Place 'sample.jpg' in the same folder.")
    
    # TO DO: Define a simple 5x5 averaging kernel (all ones / 9)
    kernel = np.ones((5, 5)) * (1/9)

    # TO DO: Apply convolution
    # (IN REALITY IS CROSS CORRELATION, we are not flipping the kernel)
    blurred = convolve(image, kernel)

    # TO DO: Create gasussian kernel and apply to image
    gaussian_kernel = gaussia_kernel(5,1)
    gaussina_blur = convolve(image, gaussian_kernel)

    # TO DO: Display results (original and blurred)
    cv2.imshow("Original", image)
    cv2.imshow("Box Blur (Manual Convolution)", blurred)
    cv2.imshow("Gaussian Blur (Manual Convolution)", gaussina_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
