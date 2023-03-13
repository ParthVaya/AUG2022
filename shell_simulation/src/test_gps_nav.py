#!/usr/bin/env python3
from pandas import read_csv
import rospy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import numpy as np
from msg import Score


global xy_list, spd_list, target, prev_angle
xy_list = []
spd_list = []

"""
def get_target_points():

    #full_list = read_csv("/home/jose/catkin_ws/src/juno_ros_scripts/src/Test_virtual_autonomous/gps_points_saved.csv") 
    full_list = read_csv("/opt/carla-simulator/PythonAPI/carla/test/gps_points_7_4.csv") 
    mycolumns = ['x','y'] 
    xy_points = full_list[mycolumns]
    spd_limits = full_list['speed']
    xy_points = xy_points.values.tolist()
    spd_limits = spd_limits.values.tolist()
    return xy_points , spd_limits

"""


def get_target_points(full_list):#need to take them with ROS
    xy_points = []
    spd_limits = []

    for idx in full_list:
        xy_points.append([idx[0],idx[1]])
        spd_limits.append(idx[2])

    return xy_points, spd_limits

def quaternion_to_euler_angle_vectorized1(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2a
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z 



def odometry_direction(start,end, heading): #define an angle that represents the direction to the next gpx point
    global target, prev_angle
    dist_threshold=2.7#max distance to check the target  [m]
    reached=False
    x_c=start[0] #current latitude in radians
    x_t=end[0] #target latitude in radians
    y_c=start[1] #current longitude in radians
    y_t=end[1] #target longitude in radians

    d_x=x_t-x_c
    d_y=y_t-y_c


    current_dist=np.sqrt(d_x**2+d_y**2) #dist in m

    if(current_dist< dist_threshold) :
        reached=True

    
    
    direction_angle = math.atan2(d_y,d_x)  #angle in radians
    
    #print(f"direction_angle is : {direction_angle}" )
    #heading_angle = math.degrees(heading)
    heading_angle=math.radians(heading) #for the non-transform3d function
    print(heading_angle)

    steer_angle = heading_angle-direction_angle
    
    while steer_angle > math.pi:
        steer_angle -= 2*math.pi
      
    while steer_angle < -math.pi:
        steer_angle += 2*math.pi
    
    #if steer_angle < -180:
    #    steer_angle = 360-(steer_angle) 
    #elif steer_angle > 180:
    #    steer_angle = steer_angle-360
#
    #if steer_angle > 180:
    #    steer_angle = 180
    #elif steer_angle < -180:
    #    steer_angle = -180

    if abs(steer_angle)-abs(prev_angle) > math.radians(150):
        #print("filter")
        if prev_angle*steer_angle > 0: # same sign 
            steer_angle = prev_angle
        else:
            steer_angle = prev_angle*-1
    
    prev_angle = steer_angle

    #print(steer_angle)

    return steer_angle, reached 


def callback_odo(odo):
    global target, prev_angle
    
    """
    quaternion = (
    odo.pose.pose.orientation.w,
    odo.pose.pose.orientation.x,
    odo.pose.pose.orientation.y,
    odo.pose.pose.orientation.z
    )
    """
    w = odo.pose.pose.orientation.w
    x = odo.pose.pose.orientation.x
    y = odo.pose.pose.orientation.y
    z = odo.pose.pose.orientation.z
    
    pitch,roll,yaw=quaternion_to_euler_angle_vectorized1(w,x,y,z)
    yaw=-yaw
    #pitch,roll,yaw=quat2euler(quaternion)
   
    steer_angle, reached=odometry_direction([odo.pose.pose.position.x, odo.pose.pose.position.y],xy_list[target],yaw)
    if(reached):
        target+=1
        if(target==len(xy_list)):
            target=0

    pub_dir_gps_comp.publish(Float64(steer_angle)) #this message is sent to kalman filter

    

    
    #
    print(target)
    pub_limit.publish(spd_list[target]) #publish speed limit
    pub_targ.publish(target)

    #pub_limit.publish(target_spd) #publish speed limit
    
def callback_score(score):
    pass


def main_loop(): 

    odo_sub = rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, callback_odo)
    score_sub=rospy.Subscriber('/score',Score, callback_score)
    rospy.spin()
    

if __name__ == '__main__': 
  
    xy_list, spd_list = get_target_points()
    pub_dir_gps_comp=rospy.Publisher('/steering_command',Float64, queue_size=1) #computed steering based on odometry, queue_size 1 as we need the current steering stuation
    gear_pub=rospy.Publisher('/gear_command',str,queue_size=1)#publisher to put the car in gear
    gear_pub.publish(str("forward"))
    #pub_dir_gps_comp=rospy.Publisher('/steering_command',Float64, queue_size=1)
    pub_targ=rospy.Publisher('target_number',Float64, queue_size=1)
    target = 0
    prev_angle = 0
    pub_limit=rospy.Publisher('speed_limit',Float64, queue_size=1) #speed limit publisher
    rospy.init_node('odo_nav', anonymous=True)
    
    try:   
     main_loop()  
    except rospy.ROSInterruptException: 
     pass


