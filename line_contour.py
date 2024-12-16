#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3

cap=cv2.VideoCapture(1)
cap.set(3,160)
cap.set(4,120)

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

    def camera_callback(self,msg):
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                rospy.loginfo("Failed to capture frame. Exiting.")
                break
            
            low_b = np.array([5, 5, 5], dtype=np.uint8)
            high_b = np.array([0, 0, 0], dtype=np.uint8)
            
            mask = cv2.inRange(frame, high_b, low_b)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                
                if M["m00"] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    rospy.loginfo(f"CX: {cx}, CY: {cy}")
                    
                    if cx >= 120:
                        rospy.loginfo("Turn Left")
                    elif 40 < cx < 120:
                        rospy.loginfo("On Track!")

                    else:  # cx <= 40
                        rospy.loginfo("Turn Right")
                    
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
            else:
                rospy.loginfo("I don't see the line")

            # Publish the processed image
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

            cv2.imshow("Mask", mask)
            cv2.imshow("Frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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

