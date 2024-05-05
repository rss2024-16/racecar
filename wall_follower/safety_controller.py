#!/usr/bin/env python
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

# from wall_follower.visualization_tools import VisualizationTools
import math

#how to use
#Step 1: run the wall follower sim               ros2 launch racecar_simulator simulate.launch.xml
#Step 2: run the safety controller sim           cd launch, then ros2 launch safety_controller.launch.xml

class SafetyController(Node):
    def __init__(self):
        '''
        The below section on muxes will help you decide which topic your safety controller
        should publish to once deployed on the racecar.

        Make this topic a ROS parameter so that you can easily change it between the simulation and the racecar.
        '''
        super().__init__("safety_controller")

        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("safety_topic", "default")
        self.declare_parameter("navigation_topic", "default")
        self.declare_parameter("stop_range", "default")

        # Fetch constants from the ROS parameter server
        # self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.SCAN_TOPIC = '/scan'
        # self.SAFETY_TOPIC = self.get_parameter('safety_topic').get_parameter_value().string_value
        self.SAFETY_TOPIC = '/vesc/low_level/input/safety'
        # self.NAVIGATION_TOPIC = self.get_parameter('navigation_topic').get_parameter_value().string_value
        self.NAVIGATION_TOPIC = '/vesc/low_level/ackermann_cmd'
        # self.STOP_RANGE = self.get_parameter("stop_range").get_parameter_value().double_value

        self.sub_navigation = self.create_subscription(AckermannDriveStamped, self.NAVIGATION_TOPIC, self.navigation_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)
        self.pub_safety = self.create_publisher(AckermannDriveStamped, self.SAFETY_TOPIC, 10)

        self.get_logger().info('HERE "%s"' % self.SAFETY_TOPIC)

        # self.VELOCITY = 1.6
        # self.STOP_RANGE = 1.0
        self.VELOCITY = 0.0


    def navigation_callback(self, msg: AckermannDriveStamped):
        '''
        Process navigation commands here
        For now, let's just pass them through
        '''
        # self.pub_safety.publish(msg)
        self.VELOCITY = msg.drive.speed


    def slice_ranges(self, laser_scan: LaserScan):
        '''
        Given an array of ranges (from the laser scan), returns the appropriate slice of data of (distances, thetas)
        Just the front

        TODO: make the slices a function of our stopping distance parameter

        angle_min=-2.3550000190734863, angle_max=2.3550000190734863, angle_increment=0.047575756907463074
        length = 100
        '''
        ranges = laser_scan.ranges
        length = len(ranges) #100
        # self.get_logger().info(str(ranges))

        #laser scans counterclockwise
        #part 1: filter the ranges data to just the front
        ranges = ranges[2*length//5 : 3*length//5]
        distances : np.ndarray = np.array(ranges)

        angles : np.ndarray = np.linspace(laser_scan.angle_min, laser_scan.angle_max, num=length)
        thetas = angles[2*length//5 : 3*length//5]

        return distances, thetas

    def scan_callback(self, laser_scan: LaserScan):
        '''
        Process laser scan data to detect obstacles
        If an obstacle is detected, issue a stop command
        '''
        distances, thetas = self.slice_ranges(laser_scan)
        stop_cmd = AckermannDriveStamped()
        # self.STOP_RANGE = 0.025*self.VELOCITY + .175 + 0.5
        offset = 0.5
        self.STOP_RANGE = self.VELOCITY**2/45 + offset
        if min(distances) < self.STOP_RANGE:
            # Example threshold, adjust as needed
            # self.get_logger().info('stopping')
            stop_cmd.drive.speed = 0.0
            stop_cmd.drive.steering_angle = 0.0
            self.pub_safety.publish(stop_cmd)
        # else:
        #     stop_cmd.drive.speed = self.VELOCITY
        #     stop_cmd.drive.steering_angle = 0.0
        # self.get_logger().info('stop_range: "%s"' % self.STOP_RANGE)
        # stop_cmd.drive.steering_angle_velocity = 0.0
        # stop_cmd.drive.acceleration = 0.0 # a= -v^2/2(d-d')
        # stop_cmd.drive.jerk = 0.0
        # self.get_logger().info('HELP "%s"' % stop_cmd.drive.speed)

def main():

    rclpy.init()
    safety_controller = SafetyController()
    rclpy.spin(safety_controller)
    safety_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
