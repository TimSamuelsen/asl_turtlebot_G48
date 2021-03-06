#!/usr/bin/env python3

from enum import Enum

import rospy
from asl_turtlebot.msg import DetectedObject
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PoseArray, Pose2D, PoseStamped
from std_msgs.msg import Float32MultiArray, String, Int16MultiArray, Int16
import tf
import numpy as np
home = np.array([3.152, 1.600, -0.001, 0, 0, 0.005, 1.000])
waypoints_old = np.array([[3.40,  2.736, -0.001,      0,      0,  0.844,   0.537],
                      [2.371, 2.660, -0.001,      0,      0,  0.672,  -0.740],
                      [0.849, 2.731, -0.001,      0,      0,  1.000,  -0.033],
                      [0.623, 2.554, -0.001,  0.003,      0,  0.930,  -0.367],
                      [0.214, 2.200, -0.001,      0,      0,  0.937,  -0.349],#0.341, 2.310
                      [0.212, 0.439, -0.001,      0,      0,  0.752,  -0.659],
                      [0.523, 0.203, -0.001,      0,      0,  0.358,  -0.934],
                      [2.369, 0.448, -0.001,      0,      0, -0.619,  -0.785],
                      [3.412, 0.219, -0.001,      0,      0, -0.123,  -0.992],
                      [3.152, 1.600, -0.001,      0,      0,  0.005,   1.000]])

waypoints = np.array([[3.329, 2.808, -0.001, 0, 0, 0.751,  0.660],  # first corner
                      [2.333, 2.710, -0.001, 0, 0, -0.440, 0.897],
                      # old[2.329, 2.633, -0.001, 0, 0, 0.54, -0.841],  # look at banana
                      [0.345, 2.615, -0.001, 0, 0, -0.855,  -0.518],  # look at car
                      #[0.100, 1.339, -0.001, 0, 0, 0.672, -0.740],  # midpoint between car and tree
                      [0.285, 0.152, -0.001, 0, 0, -0.938, -0.346],  # look at tree
                      [2.388, 0.321, -0.001, 0, 0, 0.582,   0.813], # look at fire hydrant
                      [3.152, 1.600, -0.001, 0, 0, 0.005,  1.000]]) # go home
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
    RESCUE = 3
    RESCUE_NAVIGATE = 4
    STOP = 5




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
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 1.5)
        

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
        self.rescue_counter = 0
        # Current mode
        self.mode = Mode.IDLE
        self.prev_mode = None  # For printing purposes

        # Added dicts for storing object position and whether they have been rescued
        objects = ["fire_hydrant", "car", "potted_plant", "kite"]
        self.obj_pos = dict.fromkeys(objects, None)
        self.obj_rescue = dict.fromkeys(objects, False)
        self.obj_order = []
        self.rescue_set = False
        self.stop_sign_start = rospy.get_rostime()

        ########## PUBLISHERS ##########

        # Command pose for controller
        self.pose_goal_publisher = rospy.Publisher('/cmd_pose', Pose2D, queue_size=10)
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Adding publishers for settings robot state and obj rescue order
        self.robot_state_publisher = rospy.Publisher('/robot_state', Int16, queue_size=10)
        self.obj_rescue_publisher = rospy.Publisher('/obj_rescue', String, queue_size=10)

        # Add publisher for publishing "vroom" when car is seen and possibly more things when other objects detected
        self.obj_detect_publisher = rospy.Publisher('/obj_detect', String, queue_size=10)
        ########## SUBSCRIBERS ##########

        # Stop sign detector
        #rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)
        ### Added subscribers for our objects (TIM)
        rospy.Subscriber('/detector/car', DetectedObject, self.car_detected_callback)
        rospy.Subscriber('/detector/fire_hydrant', DetectedObject, self.fire_hydrant_detected_callback)
        rospy.Subscriber('/detector/potted_plant', DetectedObject, self.potted_plant_detected_callback)
        rospy.Subscriber('detector/kite', DetectedObject, self.kite_detected_callback)
        ### Adding subsribers for setting state and rescue order
        rospy.Subscriber('/robot_state', Int16, self.set_robot_state_callback)
        rospy.Subscriber('/rescue_order', String, self.set_rescue_order_callback)

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
        self.obj_detected("fire_hydrant", msg.distance)
        self.obj_detect_publisher.publish("splash")
    
    def potted_plant_detected_callback(self, msg):
        self.obj_detected("potted_plant", msg.distance)
        self.obj_detect_publisher.publish("i'm a tree")

    def car_detected_callback(self, msg):
        self.obj_detected("car", msg.distance)
        self.obj_detect_publisher.publish("vroom")

    def kite_detected_callback(self, msg):
        self.obj_detected("kite", msg.distance)
        self.obj_detect_publisher.publish("I like flying")

    def set_robot_state_callback(self, state):
        print("robot state change")
        if state.data == 1:
            print("entering idle mode")
            self.mode = Mode.IDLE
        elif state.data == 2:
            print("entering nav mode")
            self.mode = Mode.NAV
        elif state.data == 3:
            print("entering rescue mode")
            self.mode = Mode.RESCUE
        print(self.mode)

    def set_rescue_order_callback(self, order_string):
        
        
        #self.obj_pos = {'fire_hydrant': [2.45182918414535, 0.2805413800584965, 0.769803638004356], 'car': [0.33802552076124925, 2.5225276399422185, -2.9188169500932153], 'potted_plant': [0.37100254429535223, 0.08157594453454071, -1.9378321557604188], 'kite': [2.271641266995752, 2.6407838119898845, -3.0858863561287904]}
        if self.rescue_set is False:
            self.rescue_set = True
            charList = list(order_string.data)
            p = np.zeros((3,2))
            j=0
            for i in charList:
                if i == "0":
                    p[j] = (self.obj_pos["fire_hydrant"])[0:2]
                elif i == "1":
                    p[j] = (self.obj_pos["potted_plant"])[0:2]
                elif i == "2":
                    p[j] = (self.obj_pos["car"])[0:2]
                elif i == "3":
                    p[j] = (self.obj_pos["kite"])[0:2]
                j = j + 1    
            Order = self.tsp_path(p[0],p[1],p[2],order_string)
            for i in Order:
                if i == "0":
                    self.obj_order.append("fire_hydrant")
                elif i == "1":
                    self.obj_order.append("potted_plant")
                elif i == "2":
                    self.obj_order.append("car")
                elif i == "3":
                    self.obj_order.append("kite")
            print(self.obj_order)

    ########## STATE MACHINE ACTIONS ##########

    ########## Code starts here ##########
    # Feel free to change the code here. You may or may not find these functions
    # useful. There is no single "correct implementation".

    def tsp_path(self,p1,p2,p3,order_string):
        #given positions of objects  to rescue and start point
        dist = np.zeros(6)
        paths = []
        s = home[0:2]
        #Calculate all distances between nodes - use manhattan distance
        dist[0] = np.abs(s[0]-p1[0]) + np.abs(s[1]-p1[1])
        dist[1] = np.abs(s[0]-p2[0]) + np.abs(s[1]-p2[1])
        dist[2] = np.abs(s[0]-p3[0]) + np.abs(s[1]-p3[1])
        dist[3] = np.abs(p1[0]-p2[0]) + np.abs(p1[1]-p2[1])
        dist[4] = np.abs(p1[0]-p3[0]) + np.abs(p1[1]-p3[1])
        dist[5] = np.abs(p3[0]-p2[0]) + np.abs(p3[1]-p2[1])
        
        #Possible paths
        if order_string.data == "012":
            paths = ["012","021","102"]
        elif order_string.data == "123":
            paths = ["123","132","213"]    
        elif order_string.data == "013":
            paths = ["013","031","103"]
        elif order_string.data == "023":
            paths = ["023","032","203"]
        length = np.zeros(3)
        #Calculate all possible path-lengths
        length[0] = dist[0] + dist[3] + dist[5] + dist[2]
        length[1] = dist[0] + dist[4] + dist[5] + dist[1]
        length[2] = dist[1] + dist[3] + dist[4] + dist[2]
        
        return paths[np.argmin(length)]

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

    # Added functions for object detection, storing object position and stopping at object (TIM)

    def obj_detected(self, obj, dist):
        # if we are within an acceptable range of the object
        if dist > 0 and dist < self.params.stop_min_dist:
            # if we are exploring and have not saved this objects position -> save object position
            if self.mode == Mode.NAVIGATE and self.obj_pos[obj] is None:
                self.save_obj_pos(obj)
            # if we are rescuing and have not rescued this object -> stop at the object for rescue
            #elif self.mode == Mode.RESCUE_NAVIGATE and self.obj_rescue[obj] is False:
            #    x=0
                #self.stop_at_obj(obj)
                #self.obj_rescue[obj] = True

    def save_obj_pos(self, obj):
        self.obj_pos[obj] = [self.x, self.y, self.theta]
        print("Updated object positions: ", self.obj_pos)
    
    def stop_at_obj(self, obj):
        self.stop_sign_start = rospy.get_rostime()
        self.mode = Mode.STOP
        print("rescuing: ", obj)

    def get_next_obj(self):
        obj = self.obj_order[self.rescue_counter]
        next_obj = self.obj_pos[obj]
        self.x_g = next_obj[0]
        self.y_g = next_obj[1]
        self.theta_g = next_obj[2]
        print("Res Goal: ", self.x_g, self.y_g, self.theta_g)

    def transformquat2euler(self,waypoint):
        try:
            self.x_g = waypoint[0]
            self.y_g = waypoint[1]
            quaternion = (waypoint[3],
                          waypoint[4],
                          waypoint[5],
                          waypoint[6])
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
            print("Exp Goal: ", self.x_g, self.y_g, self.theta_g)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass        
    

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
            #print("Waypoint no = ",self.counter)
            #print("No of waypoints = ",np.shape(waypoints)[0])
            if self.counter < np.shape(waypoints)[0]:
                self.transformquat2euler(waypoints[self.counter,:])
                #Call nav_to_pose function
                #Switch sate to Navigate
                self.mode = Mode.NAVIGATE

        elif self.mode == Mode.NAVIGATE:
            # Moving towards a desired pose
            #print("Error is position is", np.sqrt((self.x_g-self.x)**2+(self.y_g-self.y)**2))
            #print("Error is orientation is", self.theta_g-self.theta)
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                #print("Goal is ",self.x_g, self.y_g, self.theta_g)
                #print("Current position is ",self.x,self.y,self.theta)
                self.mode = Mode.IDLE
                self.counter = self.counter + 1
            else:
                self.nav_to_pose()
                self.go_to_pose()

        elif self.mode == Mode.RESCUE:
            print("Rescues completed: ", self.rescue_counter)
            if not self.obj_order:
                print("Set rescue order first!")
            else:
                if self.rescue_counter < len(self.obj_order):
                    self.get_next_obj()
                    self.mode = Mode.RESCUE_NAVIGATE
                else:
                    self.transformquat2euler(home)
                    self.mode = Mode.NAVIGATE

        elif self.mode == Mode.RESCUE_NAVIGATE:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                print("Goal is ",self.x_g, self.y_g, self.theta_g)
                print("Current position is ",self.x,self.y,self.theta)
                self.stop_at_obj(self.obj_order[self.rescue_counter])
                #self.rescue_counter += 1
                #self.mode = Mode.RESCUE
            else:
                self.nav_to_pose()
                self.go_to_pose()

        elif self.mode == Mode.STOP:
            if self.has_stopped():
                self.rescue_counter += 1
                if self.rescue_counter < 3:
                    self.obj_rescue[self.obj_order[self.rescue_counter]] = True
                self.mode = Mode.RESCUE
            else:
                print("Rescuing...")

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
