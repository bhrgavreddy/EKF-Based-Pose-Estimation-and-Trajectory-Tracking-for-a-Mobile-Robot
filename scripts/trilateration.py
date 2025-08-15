#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
from math import atan, atan2, pi, sin, cos, sqrt

from kalman.msg import Trilateration, Landmark
from tf.transformations import euler_from_quaternion


gazebo_flag=False
pose = [0.0, 0.0, 0.0]

landmarkA = [ 7,  -7]
varA = 0.1
landmarkB = [-7, -7]
varB = 0.1
landmarkC = [ 7, 7] 
varC = 0.1

def heading_from_quaternion(x, y, z, w):
    ang_1 = 2*(w*z + x*y)
    ang_2 = 1-2*(y**2 + z**2)
    return atan2(ang_1,ang_2) % (2*pi)

def callback(data):
    global pose
    x  = data.pose.pose.orientation.x
    y  = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    pose = [data.pose.pose.position.x, data.pose.pose.position.y, euler_from_quaternion([x,y,z,w])[2]]
    # print(pose,'PoseFromTrilateration')
    # print(f"actual x={x:.2f}, actual y={y:.2f}, actual theta={pose[2]:.2f}")#, variance={lA.variance:.2f}")

def dist(p1, p2):
    return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )**(0.5)

def callback_vicon(data):
    global pose
    pos_x = data.transform.translation.x
    pos_y = data.transform.translation.y
    orientation_q = data.transform.rotation
    heading = heading_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
    pose = np.array([pos_x,pos_y, heading]).reshape(3,1)
    # print(pose,'PoseFromTrilateration')

def trilateration_pub():
    global landmarkA, landmarkB, landmarkC
    rospy.init_node('Trilateration_node', anonymous=True)
    if gazebo_flag==True:
      rospy.Subscriber('/odom', Odometry, callback)
    elif gazebo_flag!=True:
        rospy.Subscriber('/vicon/tb3_7/tb3_7', TransformStamped, callback_vicon)
    
    pub = rospy.Publisher('trilateration_data', Trilateration, queue_size=10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        lA = Landmark(landmarkA[0], landmarkA[1], dist(pose, landmarkA)+np.random.normal(0,varA), varA)
        lB = Landmark(landmarkB[0], landmarkB[1], dist(pose, landmarkB)+np.random.normal(0,varB), varB)
        lC = Landmark(landmarkC[0], landmarkC[1], dist(pose, landmarkC)+np.random.normal(0,varC), varC)
        t = Trilateration(lA, lB, lC)
        rospy.loginfo("Sent a message!")
        if gazebo_flag==True:
            print(f"\nC******************************in gazebo*********************************")
        else:
            print(f"\nC******************************in vicon*********************************")
        print(lA,lB,lC)
        # print(f"Landmark A: x={lA.x}, y={lA.y}, distance={lA.distance:.2f}, variance={lA.variance:.2f}")
        # print(f"Landmark B: x={lB.x}, y={lB.y}, distance={lB.distance:.2f}, variance={lB.variance:.2f}")
        # print(f"Landmark C: x={lC.x}, y={lC.y}, distance={lC.distance:.2f}, variance={lC.variance:.2f}")
        # print(f"actual x={pose[0]:.2f}, actual y={pose[1]:.2f}, actual theta={pose[2]:.2f}")#, variance={lA.variance:.2f}")
        # # print(lA)
        # print(lB)
        # print(lC)
        pub.publish(t)
        rate.sleep()

if __name__ == '__main__':
    try:
        trilateration_pub()
    except rospy.ROSInterruptException:
        pass
