import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class RosBagImageSubscriber(Node):

    def __init__(self):
        super().__init__('ros_bag_image_subscriber')
        topic_name = '/camera/image_raw/compressed'

        self.get_logger().info(f'Subscribing to topic: {topic_name}')
        
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,      # More compatible with rosbag playback
            durability=DurabilityPolicy.VOLATILE,           # Standard for streaming data
            history=HistoryPolicy.KEEP_LAST,                # Only store the last N messages
            depth=10                                        # The number of messages to store (N=10)
        )
        # The QoS profile was set based on 
        # ros2 topic info /camera/image_raw/compressed --verbose
        # Type: sensor_msgs/msg/CompressedImage

        self.subscription = self.create_subscription(
            CompressedImage,
            topic_name,
            self.listener_callback,
            self.qos_profile)
        
        self.subscription

        self.frame_counter = 0 # Frame counter for received images

        # Obtain the start time from the first message received
        self.start_time = None


    def listener_callback(self, msg):
        self.get_logger().info('Received image message')
        
        # Convert ROS CompressedImage -> numpy array -> cv2 image
        np_arr = np.frombuffer(msg.data, np.uint8) # Obtain a numpy array from the compressed image data buffer
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # We convert the compressed image data into a format that OpenCV can work with (a numpy array)
        # This allows us to process and display the image using OpenCV functions.
        # Explanation of convertion process guided with AI Copilot Sonnet model
        
        # Set start time
        if self.start_time is None:
            self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        
        self.frame_counter += 1
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Add info
        cv2.putText(frame, f"Frame: {self.frame_counter} (Compressed)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"ROS Time: {timestamp:.2f}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, f"Elapsed: {timestamp - self.start_time:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Display video
        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RosBagImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
