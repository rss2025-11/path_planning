import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from nav_msgs.msg import Odometry
from .utils import LineTrajectory
from tf_transformations import euler_from_quaternion
import numpy as np

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.pose_sub = self.create_subscription(Odometry,
                                                 "/pf/pose/odom",
                                                 self.pose_callback,
                                                 1)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_pos = np.array([0.0, 0.0])
        self.trajectory_array = np.array([0.0])

    def minimum_distance_vectorized(self):
        starts = self.trajectory_array[:-1]
        ends = self.trajectory_array[1:]
        segment_lengths = np.diff(self.trajectory.distances)  # Gives length of each segment
        segment_lengths2 = segment_lengths**2
        # Handle zero-length segments to avoid divide-by-zero
        segment_lengths2 = np.where(segment_lengths2 == 0, 1e-10, segment_lengths2)

        start_to_current_pos = self.current_pos - starts

        # Projection scalar t for each segment
        t = np.sum(start_to_current_pos * segment_lengths, axis=1) / segment_lengths2
        # Clip the value of t to ensure it falls on the finite line segment
        t = np.clip(t, 0.0, 1.0)  # Clamp to [0, 1]

        # Projection point on each segment
        projections = starts + (t[:, np.newaxis] * segment_lengths)  # Shape: (N-1, 2)

        # Distance from current_pos to each projection
        dists = np.linalg.norm(projections - self.current_pos, axis=1)

        # Find minimum distance and segment index
        min_index = np.argmin(dists)
        min_point = projections[min_index]
        min_segment = [self.trajectory_array[min_index], self.trajectory_array[min_index+1]]
        return min_point, min_segment

    def find_lookahead_point(self, segment):
        circle_radius = self.lookahead
        circle_center = self.current_pos
        segment_start = segment[0]
        segment_end = segment[1]
        segment_vector = segment[1] - segment[0]

        a = np.dot(segment_vector, segment_vector)
        b = np.dot(segment_vector, segment_start - circle_center)
        c = np.dot(segment_start, segment_start) + np.dot(circle_center, circle_center) - 2* np.dot(segment_start, circle_center) - circle_radius**2
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            self.get_logger().info("THE CIRCLE DOES NOT INTERSECT WITH THE LINE SEGMENT")
            return
        sqrt_discriminant = np.sqrt(discriminant)
        sol1 = (-b + sqrt_discriminant)/(2 * a)
        sol2 = (-b - sqrt_discriminant)/(2 * a)
        if not (0 <= sol1 <= 1 or 0<= sol2 <= 1):
            self.get_logger().info("THE CIRCLE DOES NOT INTERSECT WITH THE LINE SEGMENT")
            return
        lookahead_point = segment_start+ np.max(0, np.min(1, -b / (2 * a))) * segment_vector
        return lookahead_point

    def pose_callback(self, odometry_msg):
        
        # Access the orientation quaternion
        orientation = odometry_msg.pose.pose.orientation
        # Convert quaternion to Euler angles 
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_x = odometry_msg.pose.pose.position.x
        self.current_y = odometry_msg.pose.pose.position.y
        self.current_theta = yaw
        self.current_pos[0] = self.current_x
        self.current_pos[1] = self.current_y

        
        # self.get_logger().info(f'Receiving a pose from localization: {self.current_x, self.current_y, self.current_theta}')

    # find the closest point from the robot to a trajectory segment (assuming piecewise linear segments)



    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True
        self.trajectory_array = np.array(self.trajectory.points)
        self.get_logger().info(f'type of traj {type(self.trajectory_array), type(self.trajectory_array[0]), self.trajectory_array[0]}')        



def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
