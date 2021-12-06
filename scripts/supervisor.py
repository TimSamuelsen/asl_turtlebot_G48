#!/usr/bin/env python3

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String
import tf
import numpy as np
# home = [3.152, 1.600, -0.001]
waypoints = np.array([[3.388, 2.732,  0.798],
                      [0.901, 2.717, -0.998],
                      [0.641, 2.570, -0.963],
                      [0.324, 2.375, -0.961],
                      [0.159, 0.580, -0.641],
                      [0.612, 0.301, -0.138],
                      [2.426, 0.400,  0.603],
                      [3.368, 0.301, -0.056],
                      [3.152, 1.600,  0.005]])
counter = 0

class Mode(Enum):
    """State machine modes. Feel free to change."""
    # IDLE = 1
    # POSE = 2
    # STOP = 3
    # CROSS = 4
    # NAV = 5
    # MANUAL = 6
    IDLE = 1
    NAVIGATE = 2




class SupervisorParams:

    def __init__(self, verbose=False):
        # If sim is True (i.e. using gazebo), we want to subscribe to
        # /gazebo/model_states. Otherwise, we will use a TF lookup.
        self.use_gazebo = rospy.get_param("sim")

        # How is nav_cmd being decided -- human manually setting it, or rviz
        self.rviz = rospy.get_param("rviz")

        # If using gmapping, we will have a map frame. Otherwise, it will be odom frame.
        self.mapping = rospy.get_param("map")

        # Threshold at which we consider the robot at a location
        self.pos_eps = rospy.get_param("~pos_eps", 0.3) #0.1
        self.theta_eps = rospy.get_param("~theta_eps", 0.5) #0.3

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 3.)

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5)
        

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            print("    rviz = {}".format(self.rviz))
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))
            print("    stop_time, stop_min_dist, crossing_time = {}, {}, {}".format(self.stop_time, self.stop_min_dist, self.crossing_time))


class Supervisor:

    def __init__(self):
        # Initialize ROS node
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Current state
        self.x = 0
        self.y = 0
        self.theta = 0

        # Goal state
        self.x_g = 0
        self.y_g = 0
        self.theta_g = 0
        self.counter = 0
        # Current mode
        self.mode = Mode.IDLE
        self.prev_mode = None  # For printing purposes

        # obj positions
        keys = ["fire_hydrant", "car", "potted_plant"]
        self.obj_pos = dict.fromkeys(keys, None)

        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        ########## SUBSCRIBERS ##########

        # Stop sign detector
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        ### Added subscribers for our objects (TIM)
        rospy.Subscriber('/detector/car', DetectedObject, self.car_detected_callback)
        rospy.Subscriber('/detector/fire_hydrant', DetectedObject, self.fire_hydrant_detected_callback)
        rospy.Subscriber('/detector/potted_plant', DetectedObject, self.potted_plant_detected_callback)
        ###

        # High-level navigation pose
        rospy.Subscriber('/nav_pose', Pose2D, self.nav_pose_callback)

        # If using gazebo, we have access to perfect state
        if self.params.use_gazebo:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.trans_listener = tf.TransformListener()

        # If using rviz, we can subscribe to nav goal click
        if self.params.rviz:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)
        else:
            self.x_g, self.y_g, self.theta_g = 1.5, -4., 0.
            self.mode = Mode.NAV
        

    ########## SUBSCRIBER CALLBACKS ##########

    def gazebo_callback(self, msg):
        if "turtlebot3_burger" not in msg.name:
            return

        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if self.params.mapping else "/odom"
        print("Rviz command received!")

        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (nav_pose_origin.pose.orientation.x,
                          nav_pose_origin.pose.orientation.y,
                          nav_pose_origin.pose.orientation.z,
                          nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        self.mode = Mode.NAV

    def nav_pose_callback(self, msg):
        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        self.mode = Mode.NAV

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.params.stop_min_dist and self.mode == Mode.NAV:
            self.init_stop_sign()

    ### Added callback functions for our object (TIM)
    def fire_hydrant_detected_callback(self, msg):
        # distance to object
        print("Fire hydrant")
        dist = msg.distance
        if dist > 0 and dist < self.params.stop_min_dist:
            #self.save_obj_pos("fire_hydrant")
            self.stop_at_obj("fire_hydrant")
            #if self.mode == Mode.Explore:
            #    self.save_obj_pos("fire_hydrant")
            #elif self.mode == Mode.Rescue:
            #    self.stop_at_obj("fire_hydrant")
    
    def potted_plant_detected_callback(self, msg):
        # distance to object
        print("Potted plant found")
        dist = msg.distance
        if dist > 0 and dist < self.params.stop_min_dist:
            self.save_obj_pos("potted_plant")
            #if self.mode == Mode.Explore:
            #    self.save_obj_pos("potted_plant")
            #elif self.mode == Mode.Rescue:
            #    self.stop_at_obj("potted_plant")

    def car_detected_callback(self, msg):
        # distance to object
        print("Car found")
        dist = msg.distance
        if dist > 0 and dist < self.params.stop_min_dist:
            self.save_obj_pos("car")
            #if self.mode == Mode.Explore:
            #    self.save_obj_pos("car")
            #elif self.mode == Mode.Rescue:
            #    self.stop_at_obj("car")

    ########## STATE MACHINE ACTIONS ##########

    ########## Code starts here ##########
    # Feel free to change the code here. You may or may not find these functions
    # useful. There is no single "correct implementation".

    def go_to_pose(self):
        """ sends the current desired pose to the pose controller """

        pose_g_msg = Pose2D()
        pose_g_msg.x = self.x_g
        pose_g_msg.y = self.y_g
        pose_g_msg.theta = self.theta_g

        self.pose_goal_publisher.publish(pose_g_msg)

    def nav_to_pose(self):
        """ sends the current desired pose to the navigator """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)

    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        return abs(x - self.x) < self.params.pos_eps and \
               abs(y - self.y) < self.params.pos_eps and \
               abs(theta - self.theta) < self.params.theta_eps

    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP

    def save_obj_pos(self, obj):
        self.obj_pos[obj] = [self.x, self.y, self.theta]
        print(obj, self.x, self.y, self.theta)
        
    
    def stop_at_obj(self, obj):
        self.stop_start = rospy.get_rostime()
        self.mode = Mode.STOP
        # TODO: Add something to save which object we stopped at


    

    def has_stopped(self):
        """ checks if stop sign maneuver is over """

        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.params.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """

        self.cross_start = rospy.get_rostime()
        self.mode = Mode.CROSS

    def has_crossed(self):
        """ checks if crossing maneuver is over """

        return self.mode == Mode.CROSS and \
               rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.params.crossing_time)

    ########## Code ends here ##########


    ########## STATE MACHINE LOOP ##########
    def loop(self):
        
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        if not self.params.use_gazebo:
            try:
                origin_frame = "/map" if self.params.mapping else "/odom"
                translation, rotation = self.trans_listener.lookupTransform(origin_frame, '/base_footprint', rospy.Time(0))
                self.x, self.y = translation[0], translation[1]
                self.theta = tf.transformations.euler_from_quaternion(rotation)[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            self.prev_mode = self.mode

        ########## Code starts here ##########
        # TODO: Currently the state machine will just go to the pose without stopping
        #       at the stop sign.
        

        if self.mode == Mode.IDLE:
            # Set the next goal point
            print("Waypoint no = ",self.counter)
            print("No of waypoints = ",np.shape(waypoints)[0])
            if self.counter < np.shape(waypoints)[0]:
                self.x_g,self.y_g,self.theta_g = waypoints[self.counter,:]
                #Call nav_to_pose function
                #Switch sate to Navigate
                self.mode = Mode.NAVIGATE

        elif self.mode == Mode.NAVIGATE:
            # Moving towards a desired pose
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                print("Goal is ",self.x_g, self.y_g, self.theta_g)
                print("Current position is ",self.x,self.y,self.theta)
                self.mode = Mode.IDLE
                self.counter = self.counter + 1
            else:
                self.nav_to_pose()
                # self.go_to_pose()

        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))

        ############ Code ends here ############

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()


if __name__ == '__main__':
    sup = Supervisor()
    sup.run()
