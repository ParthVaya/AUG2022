import rospy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64,Int32
import numpy as np
import pandas as pd
import ast
import random
from scipy.spatial.distance import euclidean
from shell_simulation.msg import Score
from rospkg import RosPack


def callback_odo(odo):
    if not computed:
        return
    x = odo.pose.pose.orientation.x
    y = odo.pose.pose.orientation.y
    z = odo.pose.pose.orientation.z
    
    #check with josè
    closest_appr = [euclidean([x,y,z],fake_nodes.iloc[i]) for i in range(len(fake_nodes))]
    msg.closest_approach = closest_appr
    score_pub.publish(msg)
    

def main_loop(): 
    odo_sub = rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, callback_odo)
    
    rospy.spin()
    

if __name__ == '__main__': 
    computed=False
    rospy.init_node('score_simulation', anonymous=True)
    rp=RosPack()
    score_pub=rospy.Publisher('/Score',Score,queue_size=1)
    path=rp.get_ros_paths()
    points=pd.read_csv(path[1]+"/src/points.csv", index_col=False)#all the 40 possible waypoints
    msg=Score()
    msg.mean_cpu_usage=Float64(0)
    msg.mean_speed=Float64(0)
    msg.energy_spent=Float64(0)
    msg.distance_traveled=Float64(0)
    msg.score=Int32(0)
    msg.penalties=Int32(0)
    np.random.seed(0)
    fake_nodes=np.random.permutation(40)[:15]
    print(fake_nodes)
    fake_nodes=points.iloc[fake_nodes]
    computed=True
    try:   
     main_loop()  
    except rospy.ROSInterruptException: 
     pass


