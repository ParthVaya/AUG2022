#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64
from shell_simulation.msg import CarlaEgoVehicleControl
import message_filters


def callback_cmd(angle, throttle):
    control_msg = CarlaEgoVehicleControl()
    control_msg.steer = angle.data
    control_msg.throttle = throttle.data
    #print(throttle.data)
    #print(command) 
    pub_cmd.publish(control_msg)

def main_loop():

    throttle_sub = message_filters.Subscriber('/throttle_command',Float64)
    steer_sub = message_filters.Subscriber('/steering_command', Float64)
    ts = message_filters.ApproximateTimeSynchronizer([steer_sub, throttle_sub], 10, slop=0.1, allow_headerless='True')
    ts.registerCallback(callback_cmd)
    rospy.spin()

if __name__ == '__main__':
    #pub_cmd = rospy.Publisher('steering_command',Float64, queue_size=1)
    pub_cmd = rospy.Publisher('/carla/ego_vehicle/vehicle_control_cmd',CarlaEgoVehicleControl, queue_size=1)
    rospy.init_node('steering_cmd', anonymous=True)

    try:   
     main_loop()  
    except rospy.ROSInterruptException: 
     pass

