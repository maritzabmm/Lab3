import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor_sobel')
        self.bridge = CvBridge()

        # Counters
        self.processed_count = 0
        self.compressed_count = 0

        # Define the QoS profile for best effort
        qos = QoSProfile(
            depth=1,  # Keep a shallow history
            reliability=QoSReliabilityPolicy.BEST_EFFORT  # Set to best effort
        )

        # TO DO: Create subscribers for raw and compressed images
        # Uncompressed images
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, qos)

        # Compressed images
        self.subscription = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.compressed_image_callback, qos)

        self.subscription

    # -----------------------------
    # Callbacks
    # -----------------------------
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            self.processed_count += 1
            
            # TO DO: Process the image
            processed = self.process_cv_image(cv_image)

            # TO DO: Display 2x2 grid
            if self.processed_count % 5 == 0:
                cv2.imshow("Sobel Grid", processed)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing raw image: {e}")

    def compressed_image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            self.compressed_count += 1
            
            # TO DO: Process the image
            processed = self.process_cv_image(cv_image)
            
            # Display every 5th frame
            if self.compressed_count % 5 == 0:
                # TO DO: Display 2x2 grid
                cv2.imshow("Sobel Grid", processed)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Error processing compressed image: {e}")

    # -----------------------------
    # Image processing
    # -----------------------------
    def process_cv_image(self, image):
        """Apply Sobel filters and stack 2x2 grid"""
        # TO DO: Resize image
        frame = cv2.resize(image, dsize = (0,0), fx = 0.4, fy = 0.4)

        # TO DO: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TO DO: Apply cv2.Sobel in X and Y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # TO DO: Compute Sobel magnitude
        sobel_mag = cv2.magnitude(sobel_x, sobel_y)

        sobel_x = cv2.convertScaleAbs(sobel_x) # AI was used in this part to verify in which part of the process is cv2.convertScaleAbs used.
        sobel_y = cv2.convertScaleAbs(sobel_y)
        sobel_mag = cv2.convertScaleAbs(sobel_mag)

        # TO DO: Convert grayscale images to BGR for stacking
        sobel_x_bgr = cv2.cvtColor(sobel_x, cv2.COLOR_GRAY2RGB)
        sobel_y_bgr = cv2.cvtColor(sobel_y, cv2.COLOR_GRAY2RGB)
        sobel_mag_bgr = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2RGB)

        # TO DO: Annotate each image with cv2.putText
        # frame -> "Original", sobel_mag_bgr -> "Sobel Mag", sobel_x_bgr -> "Sobel X", sobel_y_bgr -> "Sobel Y"
        cv2.putText(frame, "Original", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(sobel_mag_bgr, "Sobel Mag", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(sobel_x_bgr, "Sobel X", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(sobel_y_bgr, "Sobel Y", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        top = np.hstack([frame, sobel_mag_bgr])
        bottom = np.hstack([sobel_x_bgr, sobel_y_bgr])
        grid = np.vstack([top, bottom])
        cv2.imshow('Sobel Grid', grid)

        return grid

def main(args=None):
    rclpy.init(args=args)
    processor = ImageProcessor()
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
