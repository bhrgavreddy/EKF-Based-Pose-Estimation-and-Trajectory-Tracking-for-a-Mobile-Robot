#!/usr/bin/env python3
from __future__ import division
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import signal
import os
from datetime import datetime

# Initialize the plot
plt.ion() 
LIM = 2.5  # Limit for the plot
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_xlim([-LIM, LIM])
ax.set_ylim([-LIM, LIM])
ax.grid()

line_actual, = ax.plot([], [], 'ro', label="Actual Pose", ms=3.0, alpha=0.5)
line_desired, = ax.plot([], [], 'b-', label="Desired Trajectory", lw=1, alpha=0.7)

# Data lists
X_actual, Y_actual = [], []
X_desired, Y_desired = [], []

def pose_listener(data):
    global X_actual, Y_actual
    # Extract position
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    # Save the actual pose
    X_actual.append(x)
    Y_actual.append(y)
    print(f"Actual pose updated: ({x}, {y})")  # Debug output

def ep_listener(data):
    global X_desired, Y_desired
    try:
        # Assuming data.data is a comma-separated string "x,y"
        x, y = map(float, data.data.split(','))
        # Save the desired trajectory points
        X_desired.append(x)
        Y_desired.append(y)
        print(f"Desired pose updated: ({x}, {y})")  # Debug output
    except Exception as e:
        print(f"Error processing desired trajectory data: {e}")

def save_plot():
    """Save the current plot to the desktop with a timestamp."""
    # Update the plot with current data
    line_actual.set_data(X_actual, Y_actual)
    line_desired.set_data(X_desired, Y_desired)

    # Force a redraw to capture the current state of the plot
    fig.canvas.draw()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop = os.path.join(os.path.expanduser("~"), "/home/jatin/Desktop/log/live_plot/")
    filename = f'Robot_Trajectory_{timestamp}.png'
    filepath = os.path.join(desktop, filename)

    # Save the figure
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to save plot."""
    save_plot()
    rospy.signal_shutdown("Shutdown signal received.")

def process():
    rospy.init_node('plotting_node', anonymous=True)
    rospy.Subscriber('/odom2', Odometry, pose_listener)
    rospy.Subscriber('/bot_0/estimatedpose', String, ep_listener)

    rate = rospy.Rate(10)  # 10Hz
    
    while not rospy.is_shutdown():
        # Update the plot in the main loop
        line_actual.set_data(X_actual, Y_actual)
        line_desired.set_data(X_desired, Y_desired)

        # Redraw the figure to ensure it updates
        fig.canvas.draw()
        fig.canvas.flush_events()
        rate.sleep()

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    try:
        print("Process started ")
        plt.legend()
        plt.title('Robot Trajectory')
        # Connect the event for window close
        fig.canvas.mpl_connect('close_event', lambda event: save_plot())
        process()
    except rospy.ROSInterruptException:
        pass
