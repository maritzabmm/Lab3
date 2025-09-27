import cv2
import numpy as np
import os

def apply_rgb_filters(image):
    """Apply RGB color filtering"""
    # TO DO: Define BGR color ranges for red, green, and blue
    # Example: 
    ranges = {'blue': ([150, 0, 0],  [255, 100, 90]), 
              'red': ([0, 0, 150], [50, 100, 255]),
              'green': ([0,200,0], [100,255,25]),
              'yellow': ([0, 230, 100], [30, 255, 255]),
              'purple': ([200, 0, 100], [250, 20, 150])
        }

    results = {}
    for color, (lower, upper) in ranges.items():
        # TO DO: Convert lists to NumPy arrays
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        # TO DO: Create mask with cv2.inRange()
        mask = cv2.inRange(image, lower_bound, upper_bound)
        # TO DO: Apply cv2.bitwise_and() to extract the color region
        result = cv2.bitwise_and(image, image, mask=mask)
        filtered = result  # Replace with masked image
        results[color] = {'mask': mask, 'filtered': filtered}
    
    return results

def apply_hsv_filters(image):
    """Apply HSV color filtering"""
    # TO DO: Convert image from BGR to HSV using cv2.cvtColor()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # TO DO: Define HSV ranges for red, green, and blue
    # Hint: Red requires TWO ranges (low 0–10 and high 170–180)
    ranges = {
        'red': [([0, 30, 30], [10, 255, 255]), ([170, 50, 50], [180, 255, 255])],
        'blue': [([110, 50,  50], [130, 255, 255])],
        'green': [([40, 40, 40], [70, 255, 255])],
        'yellow': [([25, 50, 50], [30, 255, 255])],
        'purple': [([130, 40, 40], [135, 255, 255])]
    }

    results = {}
    for color, bounds_list in ranges.items():
        mask = None
        for lower, upper in bounds_list:
            # TO DO: Convert lower/upper to NumPy arrays
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            # TO DO: Create mask using cv2.inRange()
            new_mask = cv2.inRange(hsv, lower_bound, upper_bound)

            if mask is None: # AI to see how to combine the two masks for detecting red
                mask = new_mask
            else:
                # TO DO: Combine masks using cv2.bitwise_or()
                mask = cv2.bitwise_or(new_mask, mask, mask=None)

        
        # TO DO: Extract the color region using cv2.bitwise_and()
        result = cv2.bitwise_and(image, image, mask=mask)
        filtered = result
        results[color] = {'mask': mask, 'filtered': filtered}
    
    return results

def display_results(original, rgb_results, hsv_results):
    """Display results in separate windows"""
    cv2.imshow('Original', original)
    
    # ['red', 'green', 'blue', 'yellow', 'purple']
    for color in ['yellow', 'purple']:
        # TO DO: Show the filtered images for both RGB and HSV
        cv2.imshow(f'Color {color} - RGB Filtered', rgb_results[color]['filtered'])
        cv2.imshow(f'Color {color} - HSV Filtered', hsv_results[color]['filtered'])

        pass

def main():
    """Main function"""
    # Load or create image
    image_path = "images/wheel.png"
    image = cv2.imread(image_path)
    print(f"Image loaded: {image.shape}")

    # TO DO: Apply both RGB and HSV filters
    rgb_results = apply_rgb_filters(image)
    hsv_results = apply_hsv_filters(image)

    # TO DO: Display results
    display_results(image, rgb_results, hsv_results)
       
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
