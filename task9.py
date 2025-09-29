import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class HarrisCornerStudent(Node):
    def __init__(self):
        super().__init__('harris_corner_student')
        self.bridge = CvBridge()

        # Define the QoS profile for best effort
        qos = QoSProfile(
            depth=1,  # Keep a shallow history
            reliability=QoSReliabilityPolicy.BEST_EFFORT  # Set to best effort
        )

        # TO DO: Subscribe to raw and/or compressed image topics
        # Uncompressed images
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos)
        # Compressed images
        self.compressed_sub = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self.compressed_image_callback, qos)

        self.image_sub
        self.compressed_sub

    # -----------------------------
    # Callbacks
    # -----------------------------
    def image_callback(self, msg):
        try:
            # TO DO: Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            processed = self.process_cv_image(cv_image)

        except Exception as e:
            self.get_logger().error(f"Error processing raw image: {e}")

    def compressed_image_callback(self, msg):
        try:
            # TO DO: Convert ROS CompressedImage to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

            processed = self.process_cv_image(cv_image)
  
        except Exception as e:
            self.get_logger().error(f"Error processing compressed image: {e}")

    # -----------------------------
    # Image Processing
    # -----------------------------
    def process_cv_image(self, image):
        # TO DO: Resize image if needed
        frame = cv2.resize(image, dsize = (0,0), fx = 0.4, fy = 0.4)

        # TO DO: Convert to grayscale
        # Hint: cv.cvtColor(src, code[, dst[, dstCn]])
        gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayf = np.float32(gray8)

        # TO DO: Harris corner detection
        # Hint: cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]])
        dst = cv2.cornerHarris(grayf, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)

        # TO DO: Create a heatmap visualization of the Harris response
        # Hint 1. cv2.normalize(src, dst, alpha, beta, norm_type[, dtype])
        #      2. cv2.applyColorMap(src, colormap[, dst])
        normalized_image = cv2.normalize(dst, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_image = np.uint8(normalized_image)
        heatmap = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)

        # TO DO: Mark corners on original image
        corners_img = frame.copy()
        corners_img[dst > 0.04 * dst.max()] = [0, 255, 0]

        # TO DO: Convert gray to BGR for stacking
        gray = cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB) # AI used to identify where the code was incorrect.
        # Specifically regarding the grayscale visualization (needs type uint8) and the use of Harris corner detection (uses float32). 

        # TO DO: Add labels for each image (Original, Grayscale, Heatmap, Corners Overlay)
        cv2.putText(frame, "Original", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (237, 244, 34), 2, cv2.LINE_AA)
        cv2.putText(gray, "Grayscale", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, "Heatmap", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (21, 242, 242), 2, cv2.LINE_AA)
        cv2.putText(corners_img, "Overlay", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        # TO DO: Stack images in a 2x2 grid for step-by-step visualization
        top = np.hstack((frame, gray)) if gray is not None else frame
        bottom = np.hstack((heatmap, corners_img)) if heatmap is not None else corners_img
        grid = np.vstack((top, bottom)) if gray is not None and heatmap is not None else frame

        cv2.imshow('Harris Corners', grid)
        cv2.waitKey(1)

        return grid


def main(args=None):
    rclpy.init(args=args)
    node = HarrisCornerStudent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
