import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time


class ImageBagProcessor(Node):
    def __init__(self):
        super().__init__('image_bag_processor')

        self.bridge = CvBridge()
        
        # Counters
        self.processed_count = 0
        self.compressed_count = 0
        self.start_time = time.time()

        # Define the QoS profile for best effort
        qos = QoSProfile(
            depth=1,  # Keep a shallow history
            reliability=QoSReliabilityPolicy.BEST_EFFORT  # Set to best effort
        )
        
        # TO DO: Create subscribers for EITHER compressed and uncompressed images
        # Uncompressed images
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, qos)

        # Compressed images
        self.subscription = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.compressed_image_callback, qos)

        self.subscription 
        
        self.processed_count = 0

        self.get_logger().info("Image Bag Processor initialized")


    # USE ONLY ONE OF THE CALL BACKS DEPENDING ON THE TOPIC TYPE
    
    # -----------------------------
    # Callbacks 
    # -----------------------------
    def compressed_image_callback(self, msg):
        """Process compressed image messages"""
        try:
            # TO DO: Convert ROS compressed image to OpenCV (BGR8)
            # Hint: self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            self.compressed_count += 1
            
            # TO DO: Process the image
            processed = self.process_cv_image(cv_image)
            
            # Display every 5th frame
            if self.compressed_count % 5 == 0:
                cv2.imshow('Filtering Grid', processed)
                cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error (compressed): {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing compressed image: {e}')
    
    def image_callback(self, msg):
        """Process raw image messages"""
        try:
            # TO DO: Convert ROS raw image to OpenCV (BGR8)
            # Hint: self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            self.processed_count += 1
            
            # TO DO: Process the image
            processed = self.process_cv_image(cv_image)
            
            # Display every 5th frame
            if self.processed_count % 5 == 0:
                cv2.imshow('Filtering Grid', processed)
                cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error (raw): {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing raw image: {e}')
    
    # -----------------------------
    # Image Processing
    # -----------------------------
    def process_cv_image(self, cv_image):
        """Resize, apply RGB/HSV filters, and stack in 2x2 grid"""

        new_width = 500
        height, width, _ = cv_image.shape
        aspect_ratio = height / width
        new_height = int(new_width * aspect_ratio)
        
        # Resize the image
        frame = cv2.resize(cv_image, (new_width, new_height))

        # CHOSE WHICH ONE YOU ARE USING RGB OR HSV

        # TO DO: Apply RGB filters
        # rgb_results = self.apply_rgb_filters(frame)

        # TO DO: Apply HSV filters
        hsv_results = self.apply_hsv_filters(frame)

        # TO DO: Annotate the original image with text
        original = frame.copy()

        # TO DO: Choose which filtered images to display in the grid
        # TO DO: 2 x 2 grid display of original and 3 filtered images
        # Hint: np.hstack(image1, image2)
        image_green = hsv_results['green']['filtered']
        image_blue = hsv_results['blue']['filtered']
        image_red = hsv_results['red']['filtered']
        top = np.hstack([original, image_red])
        bottom = np.hstack([image_green, image_blue])
        grid = np.vstack([top, bottom])
        
        return grid
    
    # -----------------------------
    # Color Filter Functions
    # -----------------------------
    def apply_rgb_filters(self, image):
        """Apply RGB color filtering"""
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


    def apply_hsv_filters(self, image):
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

    
def main(args=None):
    rclpy.init(args=args)
    processor = ImageBagProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
