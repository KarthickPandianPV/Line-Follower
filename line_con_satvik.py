#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

class LineContourNode:
    def __init__(self): 
        rospy.init_node('line_follower', anonymous=True)

        # Subscribers
        rospy.Subscriber('/usb_cam_node/image_raw', Image, self.camera_callback)

        # Publishers
        self.angle_pub = rospy.Publisher('/line/angle_difference', Float32, queue_size=10)
        self.image_pub = rospy.Publisher('/processed_image', Image, queue_size=10)

        # Variables
        self.bridge = CvBridge()
        self.frame_counter = 0
        self.output_dir = "~/camera_ws/src/line_follower_node/frames"

    def camera_callback(self, msg):
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Define color range for masking
        low_b = np.array([0, 0, 0], dtype=np.uint8)
        high_b = np.array([5, 5, 5], dtype=np.uint8)

        # Create a mask
        mask = cv2.inRange(cv_image, low_b, high_b)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            # Find the largest contour
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            if M["m00"] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                rospy.loginfo(f"CX: {cx}, CY: {cy}")

                # Determine direction based on cx
                if cx >= 120:
                    rospy.loginfo("Turn Left")
                elif 40 < cx < 120:
                    rospy.loginfo("On Track!")
                else:  # cx <= 40
                    rospy.loginfo("Turn Right")

                # Draw a circle at the centroid
                cv2.circle(cv_image, (cx, cy), 5, (255, 255, 255), -1)

            # Draw the largest contour
            cv2.drawContours(cv_image, [c], -1, (0, 255, 0), 1)
        else:
            rospy.loginfo("I don't see the line")
        
        
        self.frame_counter += 1
        filename = os.path.join(self.output_dir, f"frame_{self.frame_counter:04d}.png")
        if(self.frame_counter <= 25):
            cv2.imwrite(filename, cv_image)

        # Publish the processed image
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error during image publishing: {e}")

        # Display the mask and frame
        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", cv_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("User requested shutdown")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = LineContourNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
