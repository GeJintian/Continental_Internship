import numpy as np
import open3d
import struct
import os
import math
from PIL import Image
from matplotlib import pyplot as plt
from bbx import Calibration, Object3d, compute_box_3d

# def voxel_construction(focal,img_height,img_w,file_name):
#     depth_map = np.load(file_name)
#     pc_list = []
#     for i in range(1,img_height-1):
#         for j in range(1,img_w-1):
#             if depth_map[i][j]<200:
#                 #if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
#                 x = depth_map[i][j]*(j-(img_w-1)/2)/focal
#                 y = -depth_map[i][j]*(i-(img_height-1)/2)/focal
#                 z = depth_map[i][j]
#                 pc_list.append([x,y,z])
#     pc_list = np.array(pc_list)
#     #print(pc_list.shape)
#     return pc_list

def read_kitti_velodyne(path):
    pc_list=[]
    points = np.fromfile(path,dtype="float32").reshape((-1,4))
    for i in range(len(points)):
            pc_list.append([points[i][0],points[i][1],points[i][2]])
    # print("z_max is",z_max)
    # print("z_min is",z_min)       
    return np.array(pc_list,dtype=np.float32)

def add_bbx(point_set):
    # point_set should be 8 points, and is_pred determines the color: True for RED, False for GREEN
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]
    #lines = [[0, 1], [1, 2], [2, 3], [0, 3]]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    bbx = open3d.open3d.geometry.LineSet()
    bbx.points = open3d.open3d.utility.Vector3dVector(point_set)
    bbx.lines = open3d.open3d.utility.Vector2iVector(lines)
    bbx.colors = open3d.open3d.utility.Vector3dVector(colors)
    return bbx

if __name__=="__main__":
    file_name = "001875"
    calib_file = "..\\kitti_object_detection\\data_object_calib\\training\\calib\\"+file_name+".txt"
    calib = Calibration(calib_filepath=calib_file)
    results_file = "disp_rcnn\\val\\"+file_name+".txt"
    #results_file = "yolo3D\\val\\"+file_name+".txt"
    object_list = []
    f = open(results_file,"r")
    line = f.readline()[:-2]
    while(line):
        obj = Object3d(line)
        obj.print_object()
        object_list.append(obj)
        line = f.readline()[:-2]
    #pc_list = voxel_construction(focal,img_height,img_w,file_name+".npy")
    pc_list = read_kitti_velodyne("..\\kitti_object_detection\\velo\\training\\velodyne\\"+file_name+".bin")
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.open3d.utility.Vector3dVector(pc_list)
    draw_list = [pcd]
    for obj in object_list:
        _, rect_cam_point_set = compute_box_3d(obj,calib.P)
        velo_point_set = calib.project_rect_to_velo(rect_cam_point_set)
        bbx = add_bbx(velo_point_set)
        draw_list.append(bbx)
    COR = open3d.open3d.geometry.TriangleMesh.create_coordinate_frame(size = 15, origin = [0,0,0])
    draw_list.append(COR)
    open3d.open3d.visualization.draw_geometries(draw_list)