import rosbag
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import numpy
import sensor_msgs.point_cloud2 as pc2
import math
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import DBSCAN
from simple_pid import PID
from controller import PIDController

bag_file = "lidardata1.bag"

def transform_cluster(cluster_points):
    # Definisci la matrice di trasformazione per convertire le coordinate dei punti
    # dal sistema di riferimento del lidar a quello del mondo.
    # Supponiamo che questa matrice sia chiamata "lidar_to_world".
    lidar_to_world = numpy.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    # Applica la trasformazione ai punti del cluster.
    return numpy.dot(lidar_to_world, cluster_points.T).T

def lidar_callback(data):
    filtered_points = []
    
    gen = pc2.read_points(data)
    cloud_points = [next(gen) for _ in range(data.width * data.height)]

    cloud_points = numpy.array(cloud_points)
    filtered_points = [p for p in cloud_points if p[2] > -1.3 and p[2] < -0.15 and p[0] > 2.5 and p[1] > -1 and p[1] < 1]
    filtered_points = numpy.array(filtered_points)

    if not len(filtered_points):
        return numpy.inf
    
    dbscan = DBSCAN(eps=0.5, min_samples=8).fit(filtered_points)

    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_list = []
    for i in range(n_clusters):
        cluster_points = filtered_points[labels == i]
        cluster_list.append(cluster_points)

    centroids = []
    for i, cluster_points in enumerate(cluster_list):
            centroid = numpy.mean(cluster_points, axis=0)
            centroids.append(centroid)

    if centroids:
        car_centroids = sorted(centroids, reverse=False, key=lambda x: x[2])[0]
        print(f"Distance to car ahead: {car_centroids[0]}")
        return car_centroids[0]
    else:
        return numpy.inf

def camera_callback(data):
    image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB )
    # plt.imshow(image)
    cv2.imshow('data', data)
    cv2.waitKey(0) 

if __name__ == '__main__':
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file, 'r')

    controller = PIDController()

    j = 0
    for topic, msg, t in bag.read_messages():
        if j % 1 == 0:
            if topic == '/carla/ego_vehicle/vlp16_1':
                distance_to_lead = lidar_callback(msg)
                throttle, brake = controller.get_brake_and_throttle_commands(distance_to_lead)
                print(f"Throttle command: {throttle}")
                print(f"Brake command: {brake}")
            elif topic == '/carla/ego_vehicle/rgb_front/image':
                camera_callback(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
        j += 1

    bag.close()
