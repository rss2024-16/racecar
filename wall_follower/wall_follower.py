#!/usr/bin/env python3
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value # 1 is left wall, -1 is right wall
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # Local constants
        self.L_CULL_ANGLE = math.radians(0)
        self.R_CULL_ANGLE = math.radians(90)
        self.CULL_DISTANCE = 5
        self.LOOK_AHEAD = 2
        self.BASE_LENGTH = 0.3

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.line_pub = self.create_publisher(Marker, '/wall', 10)

    def polar_to_cartesian(self, radius, theta):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

    def weighted_linear_fit(self, x, y, weights):
        w_sum = np.sum(weights)
        w_mean_x = np.sum(weights * x) / w_sum
        w_mean_y = np.sum(weights * y) / w_sum

        numerator = np.sum(weights * (x - w_mean_x) * (y - w_mean_y))
        denominator = np.sum(weights * (x - w_mean_x)**2)

        slope = numerator / denominator
        intercept = w_mean_y - slope * w_mean_x

        return slope, intercept
    
    def fit_line_ransac(self, points, num_iterations=100, threshold=0.1):
        best_line = None
        best_inliers = []

        for _ in range(num_iterations):
            # Randomly sample two points
            sample_indices = np.random.choice(len(points), size=2, replace=False)
            sample_points = points[sample_indices]

            # Fit a line using the sampled points
            x1, y1 = sample_points[0]
            x2, y2 = sample_points[1]
            line_params = np.polyfit([x1, x2], [y1, y2], deg=1)

            # Calculate distances from each point to the line
            distances = np.abs(np.polyval(line_params, points[:, 0]) - points[:, 1])

            # Identify inliers (points within the threshold)
            inliers = points[distances < threshold]

            # Update best fit if current fit has more inliers
            if len(inliers) > len(best_inliers):
                best_line = line_params
                best_inliers = inliers

        return best_line[0], best_line[1]
    
    def circle_intersection(self, line_slope, line_intercept, circle_radius):
        # Coefficients of the quadratic equation
        a = line_slope**2 + 1
        b = 2 * line_slope * line_intercept
        c = line_intercept**2 - circle_radius**2

        # Solve the quadratic equation for x
        x_solutions = np.roots([a, b, c])

        # Find corresponding y values using the line equation
        intersection_points = [(float(x_val), float(line_slope * x_val + line_intercept)) for x_val in x_solutions]
        
        # +X forward, +Y left, -Y right

        if intersection_points[0][0] > 0 and intersection_points[1][0] < 0:
            return intersection_points[0]
        elif intersection_points[0][0] < 0 and intersection_points[1][0] > 0:
            return intersection_points[1]
        elif intersection_points[0][1] > 0 and self.SIDE == -1 or intersection_points[0][1] < 0 and self.SIDE == 1:
            return intersection_points[0]
        else:
            return intersection_points[1]
        
    def scan_callback(self, msg):
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # Filter scan by angle range and distances
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        if self.SIDE == -1: # Right wall
            valid_indices = (angles <= -self.L_CULL_ANGLE) & (angles >= -self.R_CULL_ANGLE) & (ranges <= self.CULL_DISTANCE)
        else: # Left wall
            valid_indices = (angles >= self.L_CULL_ANGLE) & (angles <= self.R_CULL_ANGLE) & (ranges <= self.CULL_DISTANCE)  
        
        side_angles = angles[valid_indices]
        side_ranges = ranges[valid_indices]

        # Convert the polar coordinates to cartesian
        x_values, y_values = self.polar_to_cartesian(np.array(side_ranges), side_angles)

        if len(x_values) != 0:
            # Find our wall estimate lines
            slope, y_intercept = np.polyfit(x_values, y_values, 1)
            shifted_y_intercept = y_intercept - self.DESIRED_DISTANCE if self.SIDE == 1 else y_intercept + self.DESIRED_DISTANCE

            # Plot the wall
            y_plot_wall = slope * x_values + y_intercept
            VisualizationTools.plot_line(x_values, y_plot_wall, self.line_pub, frame="/base_link")

            # Plot the path
            # y_plot_path = slope * x_values + shifted_y_intercept
            # VisualizationTools.plot_line(x_values, y_plot_path, self.line_pub, frame="/base_link")

            # Find where our look ahead intersects the path
            intersect = self.circle_intersection(slope, shifted_y_intercept, self.LOOK_AHEAD)

            # Plot the destination point
            angles = np.linspace(0, 2*np.pi, 20)
            x_dest = intersect[0] + 0.1 * np.cos(angles)
            y_dest = intersect[1] + 0.1 * np.sin(angles)
            VisualizationTools.plot_line(x_dest, y_dest, self.line_pub, frame="/base_link", color=(0., 1., 0.))

            # Calculate our turn angle
            turn_angle = math.atan2(2 * self.BASE_LENGTH * intersect[1], self.LOOK_AHEAD**2)
            speed = self.VELOCITY
        else:
            # This is bad, debug
            speed = 0.0
            turn_angle = 0.0
            side = "Left Wall" if self.SIDE == 1 else "Right Wall"
            self.get_logger().info(side)

        # Publish our drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = speed
        drive_cmd.drive.steering_angle = turn_angle
        self.cmd_pub.publish(drive_cmd)

def main():
    
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
