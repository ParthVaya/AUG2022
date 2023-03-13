import pandas as pd
import rospy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String, Int32,Float64MultiArray
import numpy as np
import ast
from rospkg import RosPack

#from shell_av_ros_msgs import Score
from shell_simulation.msg import Score

def add_dummy_node(graph):
    dummy_row = np.zeros((1, len(graph)))
    dummy_col = np.zeros((len(graph) + 1, 1))
    new_graph_arr = np.hstack((np.vstack((graph, dummy_row)), dummy_col))
    return new_graph_arr.tolist()

def held_karp(distances):           # O(n^2 * 2^n)  
    n = len(distances)
    memo = {}

    def dp(mask, last):
        if mask == (1 << n) - 1:
            return distances[last][0]

        if (mask, last) in memo:
            return memo[(mask, last)]

        ans = float('inf')
        for city in range(n):
            if not (mask & (1 << city)):
                new_mask = mask | (1 << city)
                new_ans = distances[last][city] + dp(new_mask, city)
                ans = min(ans, new_ans)

        memo[(mask, last)] = ans
        return ans

    # Find the minimum weight Hamiltonian Cycle
    min_distance = dp(1, 0)

    # Build the path by backtracking
    path = [0]
    mask = 1
    last = 0
    for i in range(n - 1):
        best = None
        for city in range(n):
            if not (mask & (1 << city)):
                new_mask = mask | (1 << city)
                new_distance = distances[last][city] + dp(new_mask, city)
                if best is None or new_distance < best[0]:
                    best = (new_distance, city)
        path.append(best[1])
        mask |= 1 << best[1]
        last = best[1]

    path.append(0)

    return min_distance, path

def compute_path(distances_arr):
    graph = add_dummy_node(distances_arr)
    _, path = held_karp(graph)
    path.pop(len(path)-1)                         # rimozione del nodo fittizio
    path = path[1:]

    while path[0] != len(path):
        path.append(path.pop(0))
    return path


def get_next_xyz_list():
    global waypoint_counter,computed_path, computed_points
    print(computed_path[waypoint_counter])
    start=int(computed_points[int(computed_path[waypoint_counter]-1)])
    print(start)
    end=int(computed_points[int(computed_path[waypoint_counter+1]-1)])
    #print(f"{points.loc[ast.literal_eval(start)]}, {points.loc[ast.literal_eval(end)]}")
    print(f"next target: {points.iloc[start]}")
    print("list of waypoints:")
    print(np.array(ast.literal_eval(waypoints.iloc[start, end])))
    waypoint_counter+=1
    return np.array(ast.literal_eval(waypoints.iloc[start, end]))

#taken from internet, should be correct
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
    dist_threshold=5#max distance to check the target  [m]
    reached=False
    x_c=start[0] #current latitude in radians
    x_t=end[0] #target latitude in radians
    y_c=start[1] #current lonwdagitude in radians
    y_t=end[1] #target longitude in radians
    d_x=x_t-x_c
    d_y=y_t-y_c

    current_dist=np.sqrt(d_x**2+d_y**2) #dist in m

    if(current_dist< dist_threshold) :
        reached=True

    direction_angle = math.degrees(math.atan2(d_y,d_x))  #angle in radians
    heading_angle=heading #for the non-transform3d function

    steer_angle = heading_angle-direction_angle
    
    if steer_angle < -180:
        steer_angle = 360-(steer_angle) 
    elif steer_angle > 180:
        steer_angle = steer_angle-360

    if steer_angle > 180:
        steer_angle = 180
    elif steer_angle < -180:
        steer_angle = -180
    print(heading_angle)
    print(direction_angle)
    print(steer_angle)
    
    prev_angle = steer_angle
    return steer_angle, reached 


def callback_odo(odo):
    global target, prev_angle, xy_list
    if not computed :
        return
    
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
    yaw=yaw
   
    steer_angle, reached=odometry_direction([odo.pose.pose.position.x, odo.pose.pose.position.y],xy_list[target],yaw)
    if(reached):
        target+=1
        if(target==len(xy_list)):
            xy_list=get_next_xyz_list()
            target=0
            print(f"REACHED {waypoint_counter+1}")
    steer_angle=steer_angle/180
    
    steer_angle=np.clip(steer_angle,-1,1)
    print(waypoint_counter)
    print(xy_list[target])
    if(np.abs(steer_angle)<0.25):
        thr_pub.publish(Float64(0.35))#primitive throttle control
    else:
        thr_pub.publish(Float64(0.20))#
    steer_pub.publish(Float64(steer_angle)) #this message is sent to kalman filter

    
    pass

def callback_score(score):
    global computed_points, computed, computed_path, xy_list, carla_distances, waypoints
    if carla_distances.size > 0 and not computed:
        print("callback score")
        distances_rcv=score.closest_approach
        computed_points=np.zeros(len(distances_rcv)+1)
        for i in range(len(distances_rcv)):
            computed_points[i]=np.argmin(np.abs(distances-distances_rcv[i]))
        computed_points[len(distances_rcv)]=40
        carla_distances=carla_distances.iloc[computed_points,computed_points]
        computed_path=compute_path(carla_distances.values)
        print(computed_path)
        xy_list=get_next_xyz_list()
        #set the gear forward
        gear_pub.publish("forward")
        computed=True

def main_loop():
    #setup subscribers
    global thr_pub
    odo_sub=rospy.Subscriber('/carla/ego_vehicle/odometry', Odometry, callback_odo)
    score_sub=rospy.Subscriber('/Score',Score, callback_score)#subscribe to the score topic in order to get the distance to the 15 waypoints
    rospy.spin()

if __name__ == '__main__':
    #start node
    computed=False #compute the distances only one time
    done=False
    rospy.init_node('shell_simulation_node', anonymous=True)
    r=rospy.Rate(10)
    rp=RosPack()
    #setup variables
    target = 0#keeps track of the carla waypoints 
    prev_angle = 0#previuos angle 
    #computed_points=np.array([1,20,36,35,26,14,25,6,5,11,13,28,31,16,29,40],dtype=int)#the 15 waypoints we have to go through in no specific order
    computed_points=np.array([])
    computed_path=[]#15 points we want to go through 
    xy_list=[]#coordinates of the points we have to go through
    waypoint_counter=0#keeps track of what waypoint_counter we reached
    path=rp.get_path('shell_simulation')
    #read from CSV files
    points=pd.read_csv(path+"/src/points.csv", index_col=False)#all the 40 possible waypoints
    waypoints=pd.read_csv(path+"/src/waypoints.csv",index_col=0)#all the waypoints for pathplanning
    distances=pd.read_csv(path+"/src/distances.csv",index_col=0)#euclidean distances for estimating the position of the 15 waypoints
    carla_distances=pd.read_csv(path+"/src/carla_distances.csv",index_col=0)#distances in carla to compuet the TSP 
    #setup publishers
    gear_pub=rospy.Publisher("/gear_command",String, queue_size=1)#publisher for gear NB don't touch it again after putting the gear as forward
    steer_pub=rospy.Publisher("/steering_command",Float64, queue_size=1)#publisher for steering command
    thr_pub=rospy.Publisher("/throttle_command",Float64, queue_size=1)#publisher for throttle command
    brake_pub=rospy.Publisher("/brake_command",Float64, queue_size=1)#publisher for brake command
    done=True
    try:   
     main_loop()  
    except rospy.ROSInterruptException: 
     pass
