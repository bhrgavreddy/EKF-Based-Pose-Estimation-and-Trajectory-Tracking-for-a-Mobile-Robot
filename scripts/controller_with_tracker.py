#!/usr/bin/env python3

from __future__ import division
import rospy
from geometry_msgs.msg import Twist, TransformStamped

from std_msgs.msg import String
from nav_msgs.msg import Odometry
import numpy as np
from math import atan, atan2, pi, sin, cos, sqrt
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from kalman.msg import Trilateration
from trilateration import varA, varB, varC, landmarkA, landmarkB, landmarkC
import time  # Import the time module

import matplotlib.pyplot as plt
from datetime import datetime
import os
import math
import threading
from numpy import lcm
import signal
import sys
import logging
import csv

# Trajectory parameters
A = 1.25#8.0
B = 1.25#
a = 2
b = 4
delta_values = [0]  # Different delta values for analysis
time_limit = 20  # Duration to generate the trajectory waypoints
tolerance = 0.1  # Distance tolerance to consider a waypoint reached

# Control parameters
kp_linear = 0.55#0.5#0.5 # 1.0 working with 25 waypoints
kp_angular = 0.5# 1.0 working with 25 waypoints
max_linear_velocity = 0.7  # Maximum linear velocity
max_angular_velocity = 0.7# Maximum angular velocity

# Initialize storage for MSE calculations
mse_results = {}
final_plot_data = {}  # To store actual and desired positions for plotting before termination

# Global variables for odometry data
current_x = 0.0
current_y = 0.0
current_waypoint_index = 0
waypoints = []
actual_positions = []
desired_positions = []
# Global variable to store the time of the last call
last_call_time = None
last_call_time_1=None

# Suppress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3, suppress=True)

# Define global variables Here (pose, prev_pose, sys_input, sampling gain
# predicted_pose, estimated_pose)

# Define system noise related variables here
# Q = Process Noise
# R = Measurement noise imported from trilateration
# P = Some reasonable initial values - State Covariance
# F = System matrix for discretized unicycle is Identity
# get_current_H(pose, landmarkA, landmarkB, landmarkC)  ## H has to be calculated on the fly

# G = np.zeros((3, 2))
# I = identity matrix

kalman_flag=1

FILTER_ORDER = 5  # Change filter settings
i = 0
filter_a = [0 for i in range(FILTER_ORDER)]
filter_b = [0 for i in range(FILTER_ORDER)]
filter_c = [0 for i in range(FILTER_ORDER)]

idxA = 0
idxB = 0
idxC = 0
theta = 0

# added by jtain Define noise parameters
varX = 0.1  # Define variance for x
varY = 0.1  # Define variance for y
varTHETA = 0.1  # Define variance for theta
input_sys = np.zeros((2, 1))  # Initialize input_sys.
input_sys_previous = np.zeros((2, 1))  # Initialize input_sys.
control_command=np.zeros(2)
control_command_pre=np.zeros(2)

P_previos = np.eye(3)*0.1
P = np.eye(3)*0.1
# Q = np.eye(3)
# Q = np.eye(2)*0.1
# Q = np.zeros(2)
Q = np.diag([0.01, 0.01, 0.005])


# estimated_pose = np.zeros((3, 1)) 
# estimated_pose_previous = np.zeros((3, 1)) 
estimated_pose = np.zeros(3) 
estimated_pose_previous = np.zeros(3) 
# predicted_pose= np.zeros((3, 1)) 
# updated_pose= np.zeros((3, 1)) 
predicted_pose= np.zeros(3) 
updated_pose= np.zeros(3)
noisy_pose = None
# residual=np.zeros((6,1))
residual=np.zeros((4,1))

# K = np.zeros((3,6))
K = np.zeros((3,4))

# R = np.eye(6) * 0.1 
# R = np.eye(3) * 0.1 
# R = np.zeros(3)
# R = np.diag([0.5, 0.5, 0.5,0.5])  # assuming similar noise for each landmark measurement
R = np.diag([0.1, 0.1, 0.1,0.05])  # assuming similar noise for each landmark measurement

gazebo_flag=False

odoo = Odometry()
vdoo=Odometry()
depub = ''

estimated_angle = 0.0 
predicted_angle = 0.0



desktop_path = os.path.expanduser("/home/jatin/Desktop/log/logdata/")
os.makedirs(desktop_path, exist_ok=True)  # Create the log folder if it doesn't exist
def init_csv_log():
    """Initialize the CSV log with headers."""
    global csv_file
    if gazebo_flag!=True:
        csv_file = os.path.join(desktop_path, f"vicon_{A:.2f}_{a:.2f}_controller_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    else:
        csv_file = os.path.join(desktop_path, f"gazebo_{A:.2f}_{a:.2f}_controller_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (s)', 'Actual X', 'Actual Y', 'Actual Theta', 'Desired X', 'Desired Y', 'desired angle(rad)', 'timer','position_error (mt)','angle error(deg)','linear_Velocity_Cmd(mt/sec)', 'angular velocitycommand cmd(rad/sec)','Proportional_gain_velocity','proportional_gain_Angle','A','B','a','b','delta','x_gazebo','y_gazebo','theta_gazebo','kalman_error_x','kalman_error_y','kalman_error_theta'])

# def log_csv_data(current_time, current_x, current_y, current_angle, x_r, y_r, target_angle,timer,distance_to_waypoint,angle_err_norm,linear_velocity,angular_velocity,kp_linear,kp_angular,A,B,a,b,delta,pose_jj[0],pose_jj[1],pose_jj[2],current_x - pose_jj[0] ,current_y - pose_jj[1],current_x - pose_jj[0],updated_pose[2] - pose_jj[2]):
def log_csv_data(current_time, current_x, current_y, current_angle, x_r, y_r, target_angle,
                 timer, distance_to_waypoint, angle_err_norm, linear_velocity, angular_velocity,
                 kp_linear, kp_angular, A, B, a, b, delta, pose_jj, updated_pose):   
    """Logs the data to a CSV file."""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Log data with two decimal digits
        writer.writerow([f"{current_time:.2f}", f"{current_x:.2f}", f"{current_y:.2f}", f"{current_angle:.2f}", f"{x_r:.2f}", f"{y_r:.2f}", f"{target_angle:.2f}", f"{timer:.3f}",f"{distance_to_waypoint:.2f}",f"{angle_err_norm:.2f}",f"{linear_velocity:.2f}",f"{angular_velocity:.2f}",f"{kp_linear:.2f}",f"{kp_angular:.2f}",f"{A:.2f}",f"{B:.2f}",f"{a:.2f}",f"{b}",f"{delta}",f"{float(pose_jj[0]):.3f}",f"{float(pose_jj[1]):.3f}",f"{float(pose_jj[2]):.3f}",f"{float(current_x - pose_jj[0]):.3f}" ,f"{float(current_y - pose_jj[1]):.3f}",f"{float(updated_pose[2] - pose_jj[2]):.3f}"])
        # log_csv_data(current_time, current_x, current_y, current_angle, x_r, y_r, target_angle,timer,distance_to_waypoint,angle_err_norm,linear_velocity,angular_velocity,kp_linear,kp_angular,A,B,a,b,delta)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def desired_trajectory(t, delta):
    """Calculate desired x and y at time t based on the parametric equations."""
    x_r = A * np.sin(a * t + delta)
    y_r = B * np.sin(b * t)
    return x_r, y_r


# def generate_waypoints(delta, num_points=25):
#     """Generate waypoints along the trajectory to cover a single traversal."""
#     # Determine the period for one traversal based on the least common multiple of a and b
#     period = 2 * np.pi * (lcm(a, b) / a)

#     # Generate time points from a small offset instead of starting at 0 to avoid the origin
#     time_points = np.linspace(0.1, period/2, num_points)  # Start slightly after 0
    
#     # Generate waypoints by evaluating the trajectory at each time point
#     waypoints = [desired_trajectory(t, delta) for t in time_points]
    
#     for t in time_points:
#        print(f"t : {t}") 
#     # Print all waypoints
#     print(f"Waypoints generated for a single traversal with delta={delta:.2f}:")
#     for i, waypoint in enumerate(waypoints, start=1):
#         print(f"Waypoint {i}: {waypoint}")
    
#     return waypoints

def generate_waypoints(delta, num_points=10):
    """Generate waypoints along the trajectory to cover a single traversal and save them to a file."""
    # Determine the period for one traversal based on the least common multiple of a and b
    period = 2 * np.pi * (lcm(a, b) / a)

    # Generate time points from a small offset instead of starting at 0 to avoid the origin
    time_points = np.linspace(0.1, period / 2, num_points)  # Start slightly after 0
    
    # Generate waypoints by evaluating the trajectory at each time point
    waypoints = [desired_trajectory(t, delta) for t in time_points]
    
    # Display waypoints in the console
    print(f"Waypoints generated for a single traversal with delta={delta:.2f}:")
    for i, waypoint in enumerate(waypoints, start=1):
        print(f"Waypoint {i}: {waypoint}")

    # Prepare directory and filename for saving waypoints
    save_dir = '/home/jatin/Desktop/log'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f'waypoints_delta_{delta}_time_{timestamp}.csv')

    # Save waypoints to a CSV file
    with open(file_path, 'w') as file:
        file.write("Time, X, Y\n")
        for t, waypoint in zip(time_points, waypoints):
            file.write(f"{t:.2f}, {waypoint[0]:.2f}, {waypoint[1]:.2f}\n")
    
    print(f"Waypoints saved to {file_path}")
    return waypoints

def plot_desired_trajectory_and_waypoints(waypoints, delta):
    """Plot the desired trajectory and waypoints."""
    # Calculate trajectory points for plotting
    t_values = np.linspace(0, time_limit, 100)
    trajectory_points = [desired_trajectory(t, delta) for t in t_values]

    # Unpack the trajectory and waypoints for plotting
    trajectory_x, trajectory_y = zip(*trajectory_points)
    waypoints_x, waypoints_y = zip(*waypoints)

    plt.figure()
    plt.plot(trajectory_x, trajectory_y, label="Desired Trajectory", color='blue')
    plt.scatter(waypoints_x, waypoints_y, color='red', label='Waypoints')

    # Mark waypoints with numbers
    for i, (wx, wy) in enumerate(zip(waypoints_x, waypoints_y)):
        plt.text(wx, wy, str(i + 1), fontsize=12, ha='right', color='black')

    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title(f"Desired Trajectory with Waypoints for delta={delta:.2f}")
    plt.legend()
    plt.grid(True)

    # Save the figure with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = '/home/jatin/Desktop/log'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'desired_trajectory_waypoints_delta_{delta}_{timestamp}.png')
    plt.savefig(file_path)
    print(f"Desired trajectory plot saved at {file_path}")
    plt.show()

# Initialize storage for final plotting data
final_plot_data = {'desired': [], 'actual': []}
 
def print_matrix(name, matrix):
    """Prints the name, shape, and contents of a matrix."""
    # print(f"{name} (shape {matrix.shape}):")
    # print(matrix)
    # print("\n")

def trilaterate():

    A = (-2*state_x1 + 2*state_x2)
    B = (-2*state_y1 + 2*state_y2)
    C = (state_r1)**2 - (state_r2)**2 - (state_x1)**2 + (state_x2)**2 - (state_y1)**2 + (state_y2)**2 
    D = (-2*state_x2 + 2*state_x3)
    E = (-2*state_y2 + 2*state_y3)
    F = (state_r2)**2 - (state_r3)**2 - (state_x2)**2 + (state_x3)**2 - (state_y2)**2 + (state_y3)**2
    
    Robot_Position_x = (C*E - F*B)/(E*A - B*D + 0.000001)
    Robot_Position_y = (C*D - A*F)/(B*D - A*E + 0.000001) 

    return Robot_Position_x, Robot_Position_y

def heading_from_quaternion(x, y, z, w):
    ang_1 = 2*(w*z + x*y)
    ang_2 = 1-2*(y**2 + z**2)
    return atan2(ang_1,ang_2) % (2*pi)

def get_current_H(pose, lA, lB, lC):
    # Calculate the linearized measurement matrix H(k+1|k) at the current robot pose

    # x and y co-ordinates of landmarks    
    # current robot pose
    # Write the code for calculating Linearized H here   
    x, y, theta = pose
    # x, y = pose
    # H = np.zeros((3, 3))  # 6 measurements, 3 state variables
    # H=[]
    H = np.zeros((4, 3))  # 6 measurements, 3 state variables
    landmarks = np.array([lA, lB, lC])  # Combine landmarks into an array
    for i, landmark in enumerate(landmarks):
        lx, ly = landmark.x, landmark.y  
        dx = lx - x
        dy = ly - y
        # distance = np.sqrt(dx[0]**2 + dy[0]**2)
        distance = np.sqrt(dx**2 + dy**2)
        
        # Predicted angle to landmark i
        # angle = np.arctan2(dy, dx) - theta

        # Calculate the Jacobian for distances
        if distance != 0:
            H[i, 0] = -dx / distance  # ∂d_i/∂x
            H[i, 1] = -dy / distance  # ∂d_i/∂y
            H[i, 2] = 0              # ∂d_i/∂θ is 0 since distance is independent of θ
        # H.append([
        #     -(lx - x) / distance,
        #     -(ly - y) / distance,
        #     0
        # ])
        # # Calculate the Jacobian for angles
        # H[i + 3, 0] = -dy / (distance ** 2)  # ∂a_i/∂x
        # H[i + 3, 1] = dx / (distance ** 2)   # ∂a_i/∂y
        # H[i + 3, 2] = -1                     # ∂a_i/∂θ (change in angle due to θ)
    H[3, 0] = 0#-dx / distance  # ∂d_i/∂x
    H[3, 1] = 0#-dy / distance  # ∂d_i/∂y
    H[3, 2] = 1              # ∂d_i/∂θ is 0 since distance is independent of θ
    return H
    # return np.array(H)
    

    #return 2 * np.array(H)


def sq_dist(p1, p2):
    # Given a pair of points the function returns euclidean distance
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)#**(0.5)


def predict_state(estimated_pose_previous,P_previos):
    # System evolution
    global noisy_pose, pose, F, P, our_predicted_pose, estimated_angle, predicted_angle 
    #input_sys.shape = (2, 1)
    # Write the code for predicted pose from estimated pose here   
    # 
    global last_call_time  # Access the global variable

    # Record the current time
    current_time = time.time()  # Get the current time in seconds

    # Calculate the time difference from the last call
    if last_call_time is not None:
        time_difference = current_time - last_call_time
        # print(f"Time difference between successive calls: {time_difference:.6f} seconds")
    else:
        print("This is the first call to predict_state.")

    # Update the last call time
    last_call_time = current_time
    # print(f"Shape of input_sys_previous: {np.shape(input_sys_previous)}")
    # print(f"Type of input_sys_previous: {type(input_sys_previous)}")

    # v1=input_sys_previous[0]# = velocity_msg.linear.x
    v=control_command_pre[0]
    omega=control_command_pre[1]
    # v1 = input_sys_previous[0, 0][0]  # 
    # omega=input_sys_previous[1, 0][0] #= velocity_msg.angular.z
    
    # omega=input_sys_previous[1] #= velocity_msg.angular.z
    # def predict_state(estimated_pose, control_input):
    #v, omega = control_input  # Control inputs
    dt = time_difference  # Time step, adjust as necessary
    # dt=1
    # print("Shape of estimated_pose_previous:", estimated_pose_previous.shape)
    # print("\n")

    x, y, theta = estimated_pose_previous.flatten()  # Current estimated pose
    # x, y = estimated_pose_previous.flatten()  # Current estimated pose
    # x, y = estimated_pose_previous.flatten()  # Current estimated pose

    # Predict next state
    # x_pred = x + v1 * dt
    # y_pred = 0
    x_pred = x + v * cos(theta) * dt
    y_pred = y + v * sin(theta) * dt
    theta_pred = theta + omega * dt

    # predicted_pose[0]=x_pred
    # predicted_pose[1]=y_pred
    # predicted_pose[2]=theta_pred

    # Jacobian of the motion model (F)
    # F = np.eye(3)
    # F = np.eye(2)
    # F[0, 2] = -v * sin(theta) * dt
    # F[1, 2] = v * cos(theta) * dt
    F = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1, v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
    # Predict the covariance matrix
    # P = np.dot(np.dot(F, P_previos), F.T) + Q
    P_pred = F @ P @ F.T + Q

    #return 
    predicted_pose=np.array([[x_pred], [y_pred], [theta_pred]])#.reshape(3, 1) 
    # predicted_pose=np.array([[x_pred], [y_pred]])#.reshape(2, 1) 
   
    return predicted_pose 


def predict_measurement(predicted_pose, landmark_A, landmark_B, landmark_C):
    # Predicts the measurement (d1, d2, d3) given the current position of the robot
    
    """
    Predicts the measurement distances (d1, d2, d3) given the predicted position of the robot 
    with respect to landmarks A, B, and C.

    Parameters:
        predicted_pose (np.ndarray): Predicted position of the robot, shape (3, 1).
        landmark_A, landmark_B, landmark_C (tuples): Positions of landmarks (x, y).

    Returns:
        np.ndarray: Predicted measurements (distances to each landmark), shape (3, 1).
    """
    # Extract the predicted x, y position of the robot
    # x_pred, y_pred, theta_pred = predicted_pose[0], predicted_pose[1],predicted_pose[2]
    x_pred, y_pred= predicted_pose[0], predicted_pose[1]
    # x_pred, y_pred = predicted_pose[0], predicted_pose[1]

    theta_pred = float(predicted_pose[2][0])  
    # Calculate predicted distances to each landmark
    # d1 = sqrt((x_pred - landmark_A[0])**2 + (y_pred - landmark_A[1])**2)
    # d2 = sqrt((x_pred - landmark_B[0])**2 + (y_pred - landmark_B[1])**2)
    # d3 = sqrt((x_pred - landmark_C[0])**2 + (y_pred - landmark_C[1])**2)

    d1 = sqrt((x_pred - landmark_A.x)**2 + (y_pred - landmark_A.y)**2)
    d2 = sqrt((x_pred - landmark_B.x)**2 + (y_pred - landmark_B.y)**2)
    d3 = sqrt((x_pred - landmark_C.x)**2 + (y_pred - landmark_C.y)**2)

    # dtheta1=np.arctan2(landmark_A.y-y_pred,landmark_A.x-x_pred)-theta_pred
    # dtheta2=np.arctan2(landmark_B.y-y_pred,landmark_B.x-x_pred)-theta_pred
    # dtheta3=np.arctan2(landmark_C.y-y_pred,landmark_C.x-x_pred)-theta_pred
    # dtheta1 = (dtheta1 + np.pi) % (2 * np.pi) - np.pi
    # dtheta2 = (dtheta2 + np.pi) % (2 * np.pi) - np.pi
    # dtheta3 = (dtheta3 + np.pi) % (2 * np.pi) - np.pi
    
    
    
    # Return as a column vector of predicted measurements
    #return np.array([[d1], [d2], [d3]])
    # measurement=np.array([[d1], [d2], [d3],[dtheta1],[dtheta2],[dtheta3]])
    # measurement=np.array([[d1], [d2], [d3],[theta_pred]]).reshape(4,1)
    measurement=np.array([[d1], [d2], [d3],[theta_pred]]).reshape(4,1)
    # measurement.append(theta_pred)
    return measurement


def callback2(data):
    global noisy_pose, varX, varY, varTHETA
    global  pose_jj
    x = data.pose.pose.orientation.x
    y = data.pose.pose.orientation.y
    z = data.pose.pose.orientation.z
    w = data.pose.pose.orientation.w
    noise = [np.random.normal(0, varX), np.random.normal(
        0, varY), np.random.normal(0, varTHETA)]
    noisy_pose = np.array([data.pose.pose.position.x + noise[0], data.pose.pose.position.y +
                          noise[1], euler_from_quaternion([x, y, z, w])[2] + noise[2]]).reshape(3, 1)
    pose_jj = [data.pose.pose.position.x, data.pose.pose.position.y, euler_from_quaternion([x,y,z,w])[2]]
    # print(noisy_pose.shape,' in callback NoisyPose')
    global last_call_time_1  # Access the global variable

    # Record the current time
    current_time = time.time()  # Get the current time in seconds

    # Calculate the time difference from the last call
    if last_call_time_1 is not None:
        time_difference = current_time - last_call_time_1
        # print(f"Time difference between successive calls:of callback2= {time_difference:.6f} seconds")
    else:
        print("This is the first call to predict_state.")

    # Update the last call time
    last_call_time_1 = current_time

def callback_vicon(data):
    global noisy_pose,pose_jj
    
    # x = data.pose.pose.orientation.x
    # y = data.pose.pose.orientation.y
    # z = data.pose.pose.orientation.z
    # w = data.pose.pose.orientation.w
    pos_x = data.transform.translation.x
    pos_y = data.transform.translation.y
    orientation_q = data.transform.rotation
    heading = heading_from_quaternion(orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
    noisy_pose = np.array([pos_x,pos_y, heading]).reshape(3,1)
    #print(noisy_pose,'NoisyPose')
    pose_jj = [pos_x,pos_y, heading]

    
def get_waypoint(t):
    global K_samp  
    # write the code to generate waypoints for the desired trajectory
    
    return [m, n]

iteration_count = 0
def callback(data):
    
    global distanceLandmarkA, distanceLandmarkB, distanceLandmarkC,iteration_count 
    global idxA, idxB, idxC
    global filter_a, filter_b, filter_c
    global prev_pose, theta, pose
    global P, Q, R, F, K, residual, predicted_pose, updated_pose
    global estimated_pose, noisy_pose, our_predicted_pose, estimated_angle, predicted_angle, pose_list  
    
    global state_x1, state_y1, state_r1
    global state_x2, state_y2, state_r2
    global state_x3, state_y3, state_r3
    global current_x, current_y, current_angle,actual_positions, desired_positions, waypoints, current_waypoint_index

    lA = data.landmarkA
    lB = data.landmarkB
    lC = data.landmarkC
    # Increment the counter
    iteration_count += 1
    # print(f"Iteration: {iteration_count}")

    # Check if the callback has run 50 times
    # if iteration_count >= 50:
    #     rospy.loginfo("Reached 50 iterations, shutting down node.")
    #     rospy.signal_shutdown("50 iterations completed")
    #     exit()
    
    #######################################################
    # FILTERING VALUES
    #######################################################
    # Add value into r buffer at indices idxA, idxB, idxC
    filter_a[idxA] = lA.distance
    filter_b[idxB] = lB.distance
    filter_c[idxC] = lC.distance
    # Increment indexes
    idxA += 1
    idxB += 1
    idxC += 1

    # Wrap around the indices if buffer full
    if idxA >= FILTER_ORDER:
        idxA = 0
    if idxB >= FILTER_ORDER:
        idxB = 0
    if idxC >= FILTER_ORDER:
        idxC = 0

    # Calculate moving averages of the distances
    avg_distance_a = np.mean(filter_a)
    avg_distance_b = np.mean(filter_b)
    avg_distance_c = np.mean(filter_c)

    state_r1 = avg_distance_a
    state_r2 = avg_distance_b
    state_r3 = avg_distance_c

    state_x1 = lA.x
    state_y1 = lA.y
    state_x2 = lB.x
    state_y2 = lB.y
    state_x3 = lC.x
    state_y3 = lC.y

    # Prediction:
    global estimated_pose_previous
    global P_previos, input_sys_previous,control_command_pre

    predicted_pose = predict_state(estimated_pose_previous, P_previos)
    predict_measurements = predict_measurement(predicted_pose, lA, lB, lC)
   
    print_matrix("Predicted Pose", predicted_pose)
    print_matrix("Predicted Measurements", predict_measurements)


    measured_Position_x, measured_Position_y = trilaterate()

    # print(f"state_r1: {state_r1},state_r2:{state_r2},state_r3:{state_r3}, measured_Position_x:{measured_Position_x},measured_Position_y:{measured_Position_y} ")
   
    # state_r1 = 2.8
    # state_r2 = 2.8
    # state_r3 = 2.8
    try:
        residual[0, 0] = state_r1 - predict_measurements[0]
        residual[1, 0] = state_r2 - predict_measurements[1]
        residual[2, 0] = state_r3 - predict_measurements[2]
        residual[3, 0]=noisy_pose[2]-predict_measurements[3]
        print_matrix("Residual", residual)

    except Exception as e:
        print(f"Error calculating residual: {e}")
        return  # or handle it as appropriate

    # Covariance update:
    H = get_current_H(predicted_pose, lA, lB, lC)
    print_matrix("H (Measurement Matrix)", H)

    S = H @ P @ H.T + R
    print_matrix("S (Innovation Covariance)", S)

    # Calculate the Kalman gain K
    K = P @ H.T @ np.linalg.inv(S) 
    print_matrix("Kalman Gain K", K)

    I = np.eye(P.shape[0])

    # Step 1: Update the state estimate
    updated_pose = predicted_pose + K @ residual  # (2x1)
# #     #  print(f"updated_pose: {updated_pose.shape}\n")
#     updated_distance = predict_measurements + K @ residual
#     state_r1 = updated_distance[0]
#     state_r2 = updated_distance[1]
#     state_r3 = updated_distance[2]

#     Robot_Position_x, Robot_Position_y = trilaterate()
#     updated_pose[0] = Robot_Position_x
#     updated_pose[1] = Robot_Position_y
#     updated_pose[2] = pose_jj[2]
    
    print_matrix("Updated Pose", updated_pose)

    # Step 2: Update the covariance matrix
    updated_P = (I - K @ H) @ P  
    print_matrix("Updated Covariance P", updated_P)

    estimated_pose_previous = updated_pose
    print_matrix("estimated_pose_previous", estimated_pose_previous)
    P_previos = updated_P
    print_matrix("P_previos", P_previos)
    control_command_pre=control_command
    print_matrix("control_command_pre", control_command_pre)

   
    
  

    if kalman_flag==1:   
        # print(f"filtered x: {float(updated_pose[0]):.2f}, without_filter: {float(measured_Position_x):.2f}, actual_from_gazebo: {float(pose_jj[0]):.2f}, kalman-original: {float(updated_pose[0] - pose_jj[0]):.2f}, raw-original: {float(measured_Position_x - pose_jj[0]):.2f}")
        # print(f"filtered y: {float(updated_pose[1]):.2f}, without_filter: {float(measured_Position_y):.2f}, actual_from_gazebo: {float(pose_jj[1]):.2f}, kalman-original: {float(updated_pose[1] - pose_jj[1]):.2f}, raw-original: {float(measured_Position_y - pose_jj[1]):.2f}")
        # print(f"filtered theta: {float(updated_pose[2]):.2f}, without_filter: {float(noisy_pose[2]):.2f}, actual_from_gazebo: {float(pose_jj[2]):.2f}, kalman-original: {float(updated_pose[2] - pose_jj[2]):.2f}, raw-original: {float(noisy_pose[2] - pose_jj[2]):.2f}")
        print(f"filtered x: {(current_x):.2f}, actual_from_gazebo/vicon: {float(pose_jj[0]):.2f}, kalman-original: {float(current_x - pose_jj[0]):.2f}")#, raw-original: {float(measured_Position_x - pose_jj[0]):.2f}")
        print(f"filtered y: {current_y:.2f},  actual_from_gazebo/vicon: {float(pose_jj[1]):.2f}, kalman-original: {float(updated_pose[1] - pose_jj[1]):.2f}")#, raw-original: {float(measured_Position_y - pose_jj[1]):.2f}")
        print(f"filtered theta: {float(updated_pose[2]):.2f},actual_from_gazebo/vicon: {float(pose_jj[2]):.2f}, kalman-original: {float(updated_pose[2] - pose_jj[2]):.2f}")#, raw-original: {float(noisy_pose[2] - pose_jj[2]):.2f}")
        print("\n")#, raw-original: {float(noisy_pose[2] - pose_jj[2]):.2f}")

    x = updated_pose[0]
    y = updated_pose[1]
    theta = updated_pose[2]


    estimated_pose=updated_pose
    # pubep.publish(estimated_pose)
    # Write the code to Update covariance matrix
    # P(k+1|k+1)
    
    # Do not modify the code below
    # Send an Odometry message for visualization (refer vis.py)
    # if gazebo_flag!=True:
    #     odoo.pose.pose.position.x = x
    #     odoo.pose.pose.position.y = y
    #     quaternion_val = quaternion_from_euler(0, 0, theta)
    #     odoo.pose.pose.orientation.x = quaternion_val[0]
    #     odoo.pose.pose.orientation.y = quaternion_val[1]
    #     odoo.pose.pose.orientation.z = quaternion_val[2]
    #     odoo.pose.pose.orientation.w = quaternion_val[3]
    #     depub.publish(odoo)

   
    # odoo.pose.pose.position.x = x
    # odoo.pose.pose.position.y = y
    # quaternion_val = quaternion_from_euler(0, 0, theta)
    # odoo.pose.pose.orientation.x = quaternion_val[0]
    # odoo.pose.pose.orientation.y = quaternion_val[1]
    # odoo.pose.pose.orientation.z = quaternion_val[2]
    # odoo.pose.pose.orientation.w = quaternion_val[3]
    # depub.publish(odoo)
   
     # Update current position
    current_x = float(updated_pose[0])
    current_y = float(updated_pose[1])
    # float(predicted_pose[2][0])  
    current_angle =float(updated_pose[2] )

    # Store positions for MSE calculation
    # actual_positions.append((current_x, current_y))
    # desired_positions.append(waypoints[current_waypoint_index])
    actual_positions.append((current_x, current_y))
    desired_positions.append((x_r,y_r))

    # Store data for final plotting
    # final_plot_data['actual'].append((current_x, current_y))
    # final_plot_data['desired'].append(waypoints[current_waypoint_index])
    final_plot_data['actual'].append((current_x, current_y))
    final_plot_data['desired'].append((x_r,y_r))




    
# This is where we will write code for trajectory following
def control_loop():
    global pose, depub,pubep,pubw, input_sys, estimated_pose, control_command,noisy_pose, our_predicted_pose, estimated_angle,x_r,y_r,kalman_flag
    global current_x, current_y, current_angle,actual_positions, desired_positions, waypoints, current_waypoint_index
    x_r=0
    y_r=0
    rospy.init_node('controller_node')
   
    if gazebo_flag!=True:
        pub = rospy.Publisher('/tb3_7/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/vicon/tb3_7/tb3_7', TransformStamped, callback_vicon)
    elif gazebo_flag==True:
          pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
          rospy.Subscriber('/odom', Odometry, callback2)   
    
    rospy.Subscriber('/trilateration_data', Trilateration, callback)
    rospy.Subscriber('/odom', Odometry, callback2)
    #rospy.Subscriber('/vicon/tb3_7/tb3_7', TransformStamped, callback_vicon)
    depub = rospy.Publisher('/odom2', Odometry, queue_size=10)

    pubw = rospy.Publisher('/bot_0/waypoint', String, queue_size=10)
    pubep = rospy.Publisher('/bot_0/estimatedpose', String, queue_size=10)

    # Setting the rate for loop execution
    rate = rospy.Rate(5)

    # Twist values to move the robot
    timer = 0
    #updated_pose = estimated_pose

    # while True:
    #     user_input = input("Press '7' to exit: ")
    #     if user_input == '7':
    #         print("You pressed 7! Exiting the loop.")
    #         break  # Exit the loop
    #     else:
    #         print("You pressed:", user_input)
    
    current_waypoint_index=2
    rospy.sleep(5)
    kalman_flag=0

    while not rospy.is_shutdown():

        # Write your code here for waypoint tracking

        x_r = A * np.sin(a * timer + delta)
        y_r = B * np.sin(b * timer)


        current_angle =float(updated_pose[2] )
        # final_plot_data['actual'].append((current_x, current_y))
        # final_plot_data['desired'].append((x_r,y_r))

        # error_x = waypoints[current_waypoint_index][0] - current_x
        # error_y = waypoints[current_waypoint_index][1] - current_y
        # distance_to_waypoint = np.sqrt(error_x**2 + error_y**2)
        # print(f"***************Waypoint {current_waypoint_index + 1} **********************")
        # print(f"\nCurrent Position: (x: {current_x:.2f}, y: {current_y:.2f})")
        # print(f"Target Waypoint: (x: {waypoints[current_waypoint_index][0]:.2f}, y: {waypoints[current_waypoint_index][1]:.2f})")
        # print(f"Distance to Waypoint: {distance_to_waypoint:.2f}")
        # logging.info(f"***************Waypoint {current_waypoint_index + 1} **********************")
        # logging.info(f"\nCurrent Position: (x: {current_x:.2f}, y: {current_y:.2f})")
        # logging.info(f"Target Waypoint: (x: {waypoints[current_waypoint_index][0]:.2f}, y: {waypoints[current_waypoint_index][1]:.2f})")
        # logging.info(f"Distance to Waypoint: {distance_to_waypoint:.2f}")

        # if distance_to_waypoint < tolerance:
        #     print(f"***************Waypoint {current_waypoint_index + 1} reached.**********************")
        #     logging.info(f"***************Waypoint {current_waypoint_index + 1} reached.**********************")
        #     current_waypoint_index += 1
        #     while current_waypoint_index<len(waypoints):
                
        #         error_x = waypoints[current_waypoint_index][0] - current_x
        #         error_y = waypoints[current_waypoint_index][1] - current_y
        #         distance_to_waypoint = np.sqrt(error_x**2 + error_y**2)
        #         if distance_to_waypoint > 0.5: # skip if new waypoint is too near
        #             break
        #         current_waypoint_index += 1
        #     if current_waypoint_index >= len(waypoints):
        #         print("All waypoints have been reached. Shutting down...")
        #         rospy.signal_shutdown("Trajectory completed")
        #         return
        
        if current_x>=2.0 or current_y>=2.0:
            print("Error: robo beyond trajectory abort...")
            rospy.signal_shutdown("Trajectory completed")
            return     
           
        if current_waypoint_index < len(waypoints):
            # Current target waypoint
            # x_r, y_r = waypoints[current_waypoint_index]

            # Calculate error and distance to waypoint
            error_x = x_r - current_x
            error_y = y_r - current_y
            distance_to_waypoint = np.sqrt(error_x**2 + error_y**2)

            # Compute control inputs
            
            target_angle = np.arctan2(error_y, error_x)
            # current_angle = 2 * np.arcsin(data.pose.pose.orientation.z)  # Convert quaternion to angle
            angle_error= target_angle - current_angle
            angle_err_norm=math.atan2(math.sin(angle_error), math.cos(angle_error))  # Normalize ew
            # angular_velocity = kp_angular * wrap_angle(target_angle - current_angle)
            angular_velocity= kp_angular *angle_err_norm
            linear_velocity = kp_linear * distance_to_waypoint
            # Limit linear and angular velocities
            linear_velocity = np.clip(linear_velocity, -max_linear_velocity, max_linear_velocity)
            angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)
            current_time = rospy.get_time()
            # Print command details
            #  print(f"***************Waypoint {current_waypoint_index + 1} **********************")
            if gazebo_flag==True:
                print(f"\nC******************************in gazebo*********************************")
            else:
                print(f"\nC******************************in vicon*********************************")
            print(f"filtered x: {(current_x):.2f}, actual_from_gazebo/vicon: {float(pose_jj[0]):.2f}, kalman-original: {float(current_x - pose_jj[0]):.2f}")#, raw-original: {float(measured_Position_x - pose_jj[0]):.2f}")
            print(f"filtered y: {current_y:.2f},  actual_from_gazebo/vicon: {float(pose_jj[1]):.2f}, kalman-original: {float(updated_pose[1] - pose_jj[1]):.2f}")#, raw-original: {float(measured_Position_y - pose_jj[1]):.2f}")
            print(f"filtered theta: {float(updated_pose[2]):.2f},actual_from_gazebo/vicon: {float(pose_jj[2]):.2f}, kalman-original: {float(updated_pose[2] - pose_jj[2]):.2f}")#, raw-original: {float(noisy_pose[2] - pose_jj[2]):.2f}")

            print(f"Current Position: (x: {current_x:.2f}, y: {current_y:.2f})")
            print(f"Target Waypoint: (xr: {x_r:.2f}, yr: {y_r:.2f})")
            print(f"Distance to Waypoint: {distance_to_waypoint:.2f}")
            # logging.info(f"***************Waypoint {current_waypoint_index + 1} **********************")
            # logging.info(f"\nCurrent Position: (x: {current_x:.2f}, y: {current_y:.2f})")
            # logging.info(f"Target Waypoint: (x: {waypoints[current_waypoint_index][0]:.2f}, y: {waypoints[current_waypoint_index][1]:.2f})")
            # logging.info(f"Distance to Waypoint: {distance_to_waypoint:.2f}")
            print(f"target_angle: {target_angle:.2f}")
            print(f"current_angle: {current_angle:.2f}")
            print(f"angle_err_norm: {angle_err_norm:.2f}")
            print(f"Linear Velocity Command: {linear_velocity:.2f}")
            print(f"Angular Velocity Command: {angular_velocity:.2f}")
            # logging.info(f"target_angle: {target_angle:.2f}")
            # logging.info(f"current_angle: {current_angle:.2f}")
            # logging.info(f"angle_err_norm: {angle_err_norm):.2f}")
            # logging.info(f"Linear Velocity Command: {linear_velocity:.2f}")
            # logging.info(f"Angular Velocity Command: {angular_velocity:.2f}")
            # logging.debug("This is a debug message.")
            # logging.info("This is an info message.")
            # logging.warning("This is a warning message.")
            # logging.error("This is an error message.")
            # logging.critical("This is a critical message.")
            # log_csv_data(current_time, current_x, current_y, current_angle, x_r, y_r, target_angle,timer,distance_to_waypoint,angle_err_norm,linear_velocity,angular_velocity,kp_linear,kp_angular,A,B,a,b,delta,pose_jj[0],pose_jj[1],pose_jj[2],current_x - pose_jj[0] ,current_y - pose_jj[1],current_x - pose_jj[0],updated_pose[2] - pose_jj[2])
            # log_csv_data(current_time, current_x, current_y, current_angle, x_r, y_r, target_angle,timer,distance_to_waypoint,angle_err_norm,linear_velocity,angular_velocity,kp_linear,kp_angular,A,B,a,b,delta,pose_jj[0],pose_jj[1],pose_jj[2])
            log_csv_data(current_time, current_x, current_y, current_angle, x_r, y_r, target_angle,timer,distance_to_waypoint,angle_err_norm,linear_velocity,angular_velocity,kp_linear,kp_angular,A,B,a,b,delta,pose_jj,updated_pose)

            # Command the TurtleBot

            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_velocity
            control_command[0]=linear_velocity#0.0000000
            control_command[1]=angular_velocity#0.000000
            input_sys[0] = linear_velocity
            input_sys[1] = angular_velocity
            pub.publish(twist)


        # velocity_msg = Twist() 

        # vel_cmd=0.1
        # ang_cmd=0.1
        # velocity_msg.linear.x = vel_cmd #0.00000#0.1
        # velocity_msg.angular.z = ang_cmd#0.000000000000#0.2

        # control_command[0]=vel_cmd#0.0000000
        # control_command[1]=ang_cmd#0.000000
        # input_sys[0] = velocity_msg.linear.x
        # input_sys[1] = velocity_msg.angular.z
        # timer = timer + 0.004 * pi
        timer = timer + 0.0015* pi

        # If robot has reached the current waypoint
        # Sample a new waypoint
        # Apply proportional control to reach the waypoint
        ####
        # pub.publish(velocity_msg)       
        # pubep.publish(str([estimated_pose.tolist()[i][0] for i in range(2)])) 
       

          # Prepare the string to publish
           
        
        odoo.pose.pose.position.x = current_x
        odoo.pose.pose.position.y = current_y
        quaternion_val = quaternion_from_euler(0, 0, current_angle)
        odoo.pose.pose.orientation.x = quaternion_val[0]
        odoo.pose.pose.orientation.y = quaternion_val[1]
        odoo.pose.pose.orientation.z = quaternion_val[2]
        odoo.pose.pose.orientation.w = quaternion_val[3]
        depub.publish(odoo)
        
        
        # estimated_pose_str = f"{x_r},{y_r},{target_angle}"

        estimated_pose_str = f"{x_r},{y_r}"
        pubep.publish(estimated_pose_str)
        # pubep.publish(str(noisy_pose.tolist()))   #commented bu jj
        # # print("Controller message pushed at {}".format(rospy.get_time()))
        rate.sleep()

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_final_trajectory(delta):
    """Plot the final desired vs actual trajectory."""
    global mse
    if not final_plot_data['desired'] or not final_plot_data['actual']:
        print(f"Error: No trajectory data found for delta={delta:.2f}.")
        return

    desired_positions_np = np.array(final_plot_data['desired'])
    actual_positions_np = np.array(final_plot_data['actual'])

    plt.figure(figsize=(10, 6))
    plt.plot(desired_positions_np[:, 0], desired_positions_np[:, 1], label="Desired Trajectory", color='blue')
    plt.plot(actual_positions_np[:, 0], actual_positions_np[:, 1], label="Actual Trajectory", color='orange')
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title(f"Final Trajectory Tracking for delta={delta}")

    # Adding additional info in the legend
    additional_info = (f"MSE: {mse:.2f}\n"
                       f"Waypoints: {len(waypoints)}\n"
                       f"Delta: {delta:.2f}\n"
                       f"Kp (Linear): {kp_linear:.2f}\n"
                       f"Kp (Angular): {kp_angular:.2f}\n"
                       f"A:{A} B:{B} a:{a} b:{b}\n")

    # Determine the maximum x-value in the trajectory data
    max_x_value = max(np.max(desired_positions_np[:, 0]), np.max(actual_positions_np[:, 0]))

    # Place legend just outside of the max x-value
    plt.legend(title=additional_info, loc="center left", bbox_to_anchor=(1.00, 0.5), borderaxespad=0.)

    # Set the x-axis limit to ensure there’s space for the legend outside of max x-value
    plt.xlim(right=max_x_value + 1)  # Add some padding to the right of max x-value

    plt.grid(True)
    plt.tight_layout()

    # Save the figure with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = '/home/jatin/Desktop/log'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'waypoints_{len(waypoints)}_Kv_{kp_linear}_kw_{kp_angular}_delta_{delta}_time_{timestamp}.png')
    plt.savefig(file_path, bbox_inches="tight", pad_inches=0.2)
    print(f"Final trajectory plot saved at {file_path}")
    plt.show()



def track_trajectory(delta):
    """Track the trajectory through waypoints with a given delta and calculate MSE."""
    global waypoints, current_waypoint_index
    # Subscriber for odometry data
    # rospy.Subscriber('/odom', Odometry, odom_callback)
    # Reset data storage for each run
    actual_positions.clear()
    desired_positions.clear()

    # Generate waypoints for the given delta
    waypoints = generate_waypoints(delta)
    current_waypoint_index = 0

    # Plot desired trajectory and waypoints before tracking
    plot_desired_trajectory_and_waypoints(waypoints, delta)

    # Start the control loop in a separate thread
    # control_thread = threading.Thread(target=control_loop)
    # control_thread.start()

    control_loop()

    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("\nTerminating the program gracefully...")
    #     # Calculate Mean Squared Error (MSE) if data is available
    #     if actual_positions and desired_positions:
    #         mse = np.mean([(dx - ax) ** 2 + (dy - ay) ** 2 for (dx, dy), (ax, ay) in zip(desired_positions, actual_positions)])
    #         mse_results[delta] = mse
    #         print(f"MSE for delta={delta:.2f}: {mse:.4f}")
    #     else:
    #         print(f"Warning: No trajectory data available for delta={delta:.2f}.")

    #     # Store final positions for plotting
    #     final_plot_data['actual'] = actual_positions
    #     final_plot_data['desired'] = desired_positions



def mse_read():
    global mse
    if actual_positions and desired_positions:
            mse = np.mean([(dx - ax) ** 2 + (dy - ay) ** 2 for (dx, dy), (ax, ay) in zip(desired_positions, actual_positions)])
            mse_results[delta] = mse
            print(f"MSE for delta={delta:.2f}: {mse:.4f}")
    else:
            print(f"Warning: No trajectory data available for delta={delta:.2f}.")

def signal_handler(sig, frame):
    print("Ctrl+C pressed. Shutting down...")
    # rospy.signal_shutdown("Shutdown signal received")
    mse_read()
    plot_final_trajectory(delta)
    rospy.loginfo("Ctrl+C pressed. Shutting down...")
    rospy.signal_shutdown("Shutdown signal received")

if __name__ == '__main__':
    global x_r,y_r
    init_csv_log()
    try:
        signal.signal(signal.SIGINT, signal_handler)
        
        for delta in delta_values:
            track_trajectory(delta)
            mse_read()
            plot_final_trajectory(delta)
    
    
    except rospy.ROSInterruptException:
        pass
