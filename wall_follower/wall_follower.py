#!/usr/bin/env python3
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

from wall_follower.visualization_tools import VisualizationTools

import time


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")
        self.declare_parameter("plot_frame", "/base_link")

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.PLOT_FRAME = self.get_parameter('plot_frame').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value # 1 is left wall, -1 is right wall
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

        # Local constants
        self.L_CULL_ANGLE = math.radians(30)
        self.R_CULL_ANGLE = math.radians(120)

        self.F_CULL_ANGLE = math.radians(20)

        self.CULL_DISTANCE = 2.0
        self.LOOK_AHEAD = 3.0 # Should prob be a function of speed
        self.BASE_LENGTH = 0.3

        self.get_logger().info(str(self.VELOCITY))
        
        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.line_pub_left = self.create_publisher(Marker, '/left_wall', 10)
        self.line_pub_right = self.create_publisher(Marker, '/right_wall', 10)
        self.line_pub_front = self.create_publisher(Marker, '/front_wall', 10)
        self.path_pub = self.create_publisher(Marker, '/path', 10)
        self.metric_pub = self.create_publisher(String, '/wall_follower_metrics', 10)

        #stat tracking
        try:
            self.metrics_dict['test'] = False
        except:
            self.metrics_dict = \
                {
                'look_ahead':[],
                'front_wall': [],
                'following_wall':[],
                'average_dist':[],
                'test':False
                }

    def polar_to_cartesian(self, radius, theta):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

    def fit_line_weighted(self, x, y, weights):
        w_sum = np.sum(weights)
        w_mean_x = np.sum(weights * x) / w_sum
        w_mean_y = np.sum(weights * y) / w_sum

        numerator = np.sum(weights * (x - w_mean_x) * (y - w_mean_y))
        denominator = np.sum(weights * (x - w_mean_x)**2)

        slope = numerator / denominator
        intercept = w_mean_y - slope * w_mean_x

        return slope, intercept
    
    def fit_line_ransac(self, x, y, num_iterations=100, threshold=0.1):
        points = np.column_stack((x, y))

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
        
    def line_intersection(self, eq1, eq2):
        '''
        eq1/2 - 2d array [m,c] where m is slope and c is intercept

        returns [x,y] where x and y are intercept location relative to robot
        '''
        slope_diff = eq1[0]-eq2[0]
        y_int_diff = eq2[1]-eq1[1]
        x_int = y_int_diff/slope_diff
        y_int = eq1[0]*x_int + eq1[1]

        xy1 = [x_int-1, eq1[0]*(x_int-1)+eq1[1]]
        xy2 = [x_int+1,eq2[0]*(x_int+1)+eq2[1]]

        a=( (xy1[0]-x_int)**2 + (xy1[1]-y_int)**2 )**(1/2)
        b=( (xy2[0]-x_int)**2 + (xy2[1]-y_int)**2 )**(1/2)
        c=( (xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2 )**(1/2)
        angle = np.arccos( (c**2-a**2-b**2) / (-2*a*b) )
        return angle
        
    def scan_callback(self, msg):
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        # Filter scan by angle range and distances
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)

        distance = self.CULL_DISTANCE

        # Get our data slices for left, front, and right walls
        valid_indices_rw = (angles <= -self.L_CULL_ANGLE) & (angles >= -self.R_CULL_ANGLE) & (ranges <= distance)
        while not valid_indices_rw.any(): # Increase cull distance until we get data
            distance += 1
            valid_indices_rw = (angles <= -self.L_CULL_ANGLE) & (angles >= -self.R_CULL_ANGLE) & (ranges <= distance)
        side_angles_rw = angles[valid_indices_rw]
        side_ranges_rw = ranges[valid_indices_rw]

        distance = self.CULL_DISTANCE
        valid_indices_lw = (angles >= self.L_CULL_ANGLE) & (angles <= self.R_CULL_ANGLE) & (ranges <= distance)  
        while not valid_indices_lw.any(): # Increase cull distance until we get data
            distance+=1
            valid_indices_lw = (angles >= self.L_CULL_ANGLE) & (angles <= self.R_CULL_ANGLE) & (ranges <= distance)  
        side_angles_lw = angles[valid_indices_lw]
        side_ranges_lw = ranges[valid_indices_lw]

        valid_indices_fw = (angles >= -self.F_CULL_ANGLE) & (angles <= self.F_CULL_ANGLE)
        front_angles = angles[valid_indices_fw]
        front_ranges = ranges[valid_indices_fw]

        # Convert the polar coordinates to cartesian
        x_values_rw, y_values_rw = self.polar_to_cartesian(np.array(side_ranges_rw), side_angles_rw)
        x_values_lw, y_values_lw = self.polar_to_cartesian(np.array(side_ranges_lw), side_angles_lw)
        x_values_fw, y_values_fw = self.polar_to_cartesian(np.array(front_ranges), front_angles)

        # Find our wall estimate lines
        slope_rw, y_intercept_rw = np.polyfit(x_values_rw, y_values_rw, 1)
        slope_lw, y_intercept_lw = np.polyfit(x_values_lw, y_values_lw, 1)
        slope_fw,y_intercept_fw = self.fit_line_ransac(x_values_fw, y_values_fw)

        slope_threshold = .5
        
        slope = slope_rw if self.SIDE == -1 else slope_lw

        self.get_logger().info(str(slope))

        # if abs(slope) > slope_threshold:
        #     look_ahead = 1
        # else:
        #     look_ahead = self.LOOK_AHEAD

        look_ahead = self.LOOK_AHEAD


        
        # Plot the walls
        y_plot_wall_rw = slope_rw * x_values_rw + y_intercept_rw
        VisualizationTools.plot_line(x_values_rw, y_plot_wall_rw, self.line_pub_right, frame=self.PLOT_FRAME)
    
        # y_plot_wall_lw = slope_lw * x_values_lw + y_intercept_lw
        # VisualizationTools.plot_line(x_values_lw, y_plot_wall_lw, self.line_pub_left, frame=self.PLOT_FRAME)

        # y_plot_fw = slope_fw * x_values_fw + y_intercept_fw
        # VisualizationTools.plot_line(x_values_fw, y_plot_fw, self.line_pub_front, frame=self.PLOT_FRAME)
        
        # Plot the path
        shifted_y_intercept = y_intercept_lw - self.DESIRED_DISTANCE if self.SIDE == 1 else y_intercept_rw + self.DESIRED_DISTANCE
        x_path = x_values_lw if self.SIDE == 1 else x_values_rw
        slope_path = slope_lw if self.SIDE == 1 else slope_rw
        y_plot_path = x_path * slope_path + shifted_y_intercept
        VisualizationTools.plot_line(x_path, y_plot_path, self.path_pub, frame=self.PLOT_FRAME, color=(0., 1., 0.))

        # Find where our look ahead intersects the path
        max_wall_distance = 3.0
        if self.SIDE == -1:
            intersect = self.circle_intersection(slope_rw, shifted_y_intercept, look_ahead)
        else:
            intersect = self.circle_intersection(slope_lw, shifted_y_intercept, look_ahead)
        
        turn_angle = math.atan2(2 * self.BASE_LENGTH * intersect[1], look_ahead**2)
            
        # If close to a front wall
        if max(front_ranges) < max_wall_distance:

            # If there is a corner, find which side is open and drive toward it
            if max(abs(side_ranges_lw)) > 2 * max(abs(side_ranges_rw)): # Right wall, y positive after reflection
                angle = self.line_intersection([slope_fw, y_intercept_fw], [slope_rw, y_intercept_rw])
                intersect = look_ahead*np.array([np.cos(angle),np.sin(angle)])
                turn_angle = math.atan2(2 * self.BASE_LENGTH * math.sin(math.atan2(intersect[1], intersect[0])), math.sqrt(intersect[0]**2 + intersect[1]**2))

            elif max(abs(side_ranges_rw)) > 2 * max(abs(side_ranges_lw)): # Left wall, y negative after reflection
                angle = self.line_intersection([slope_fw, y_intercept_fw], [slope_lw, y_intercept_lw])
                intersect = -look_ahead*np.array([np.cos(angle),np.sin(angle)])
                turn_angle = math.atan2(2 * self.BASE_LENGTH * math.sin(math.atan2(intersect[1], intersect[0])), math.sqrt(intersect[0]**2 + intersect[1]**2))
            
        # Plot the destination point
        ang_dest = np.linspace(0, 2*np.pi, 20)
        x_dest = intersect[0] + 0.1 * np.cos(ang_dest)
        y_dest = intersect[1] + 0.1 * np.sin(ang_dest)
        VisualizationTools.plot_line(x_dest, y_dest, self.path_pub, frame=self.PLOT_FRAME, color=(0., 1., 0.))
        
        # Publish our drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = self.VELOCITY
        drive_cmd.drive.steering_angle = turn_angle
        #self.get_logger().info("publishing drive cmd %s" % str(drive_cmd))
        self.cmd_pub.publish(drive_cmd)

        # Publish performance metrics
        metric_str = String()        
        valid_side = (angles > math.radians(85)) & (angles < math.radians(95)) if self.SIDE == -1 else (angles < math.radians(-85)) & (angles > math.radians(-95))
        side_dist = np.mean(ranges[valid_side])
        min_dist = min(ranges[valid_side])
        metric_str.data = f"Average side distance: {side_dist}, Min distance: {min_dist}"
        self.metric_pub.publish(metric_str)
        self.metrics_dict['look_ahead'].append(look_ahead)
        self.metrics_dict['front_wall'].append(max(front_ranges))
        self.metrics_dict['following_wall'].append(min_dist)
        self.metrics_dict['average_dist'].append(side_dist)

def main():
    
    rclpy.init()
    wall_follower = WallFollower()
    try:
        rclpy.spin(wall_follower)
    except:
        pass
    np.save(str(time.time()),wall_follower.metrics_dict)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    


