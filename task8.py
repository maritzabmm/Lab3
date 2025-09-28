import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class CannyProcessor(Node):
    def __init__(self):
        super().__init__('canny_processor')
        self.bridge = CvBridge()
        self.processed_count = 0
        self.compressed_count = 0

        # Define the QoS profile for best effort
        qos = QoSProfile(
            depth=1,  # Keep a shallow history
            reliability=QoSReliabilityPolicy.BEST_EFFORT  # Set to best effort
        )

        # TO DO: Create subscriptions for raw and/or compressed images
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
            # TO DO: Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            self.processed_count += 1
            processed = self.process_cv_image(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error processing raw image: {e}")

    def compressed_image_callback(self, msg):
        try:
            # TO DO: Convert ROS CompressedImage message to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            self.compressed_count += 1
            processed = self.process_cv_image(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error processing compressed image: {e}")

    # -----------------------------
    # Image processing
    # -----------------------------
    
    def process_cv_image(self, image):
        """Resize, blur, apply Canny edge detection, stack grid with colored labels"""
        # TO DO: Resize if necessary
        frame = cv2.resize(image, dsize = (0,0), fx = 0.4, fy = 0.4)

        # TO DO: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TO DO: Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3,3), 0)

        # TO DO: Compute Sobel gradients (X and Y)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        magnitude = cv2.convertScaleAbs(magnitude)
        magnitude_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

        # TO DO: Apply Canny edge detection
        edges = cv2.Canny(blurred, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # TO DO: Convert gray and blurred to BGR for stacking
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)

        # -----------------------------
        # TO DO: Add colored labels on each image
        # -----------------------------
        cv2.putText(frame, "Original", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(gray_bgr, "Grayscale", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(blurred_bgr, "Blurred", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(magnitude_bgr, "Grad Mag", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(edges_bgr, "Canny", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(edges_bgr, "Canny", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        # -----------------------------
        # TO DO: Stack images into 2x3 grid
        # -----------------------------
        top = np.hstack([frame, gray_bgr, blurred_bgr])
        bottom = np.hstack([magnitude_bgr, edges_bgr, edges_bgr])
        grid = np.vstack([top, bottom])
        cv2.imshow('Canny Grid', grid)
        cv2.waitKey(1)

        return grid


def main(args=None):
    rclpy.init(args=args)
    processor = CannyProcessor()
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
