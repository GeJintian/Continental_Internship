import os
import sys
from tkinter import *

import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2
import open3d
from PIL import Image, ImageDraw
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from nets import Model
from sklearn.cluster import DBSCAN
import torch
import shutil
import math


def load_model(model_path):
    """
    Input:
        model_path: str; path to the model of Crestereo
    Output:
        model: nn.model; loaded model in evaluation mode
    """
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = mge.load(model_path)
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict["state_dict"], strict=True)

    model.eval()
    return model


def inference(left, right, model, n_iter=20):
    """
    Source code from author
    """
    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = mge.tensor(imgL).astype("float32")
    imgR = mge.tensor(imgR).astype("float32")

    imgL_dw2 = F.nn.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.nn.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

    pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = F.squeeze(pred_flow[:, 0, :, :]).numpy()

    return pred_disp

def build_bev_points(x,z,result):
    """
    Input:
        x: float; x coordinate in 3D
        z: float; z coordinate in 3D
        result: (n,2) array; contains all bev points
    Effect:
        Transfer a voxel in 3D coordinate system into a bev point
    """
    y_axis = math.floor(z/0.1)
    #print(id1)
    x_axis = math.floor(x/0.1)
    result[159+x_axis][y_axis] += 1

# This function is a test, and shouldn't be used any more.
def sigma(xmin, ymin, xmax, ymax, focal, depth_map, mode):
    # mode: 0 for sigma-rejection, 1 for repeated-sigma rejection, 2 for DBSCAN
    thres = 0.5
    line_set = []
    pc_list = []
    depth_statistic = []

    img_w = 1280
    img_height = 720

    # for i in range(1,img_height-1):
    #     for j in range(1,img_w-1):
    #         if depth_map[i][j]<200:
    #             if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
    #                 x = depth_map[i][j]*(j-(img_w-1)/2)/focal
    #                 y = -depth_map[i][j]*(i-(img_height-1)/2)/focal
    #                 z = depth_map[i][j]
    #                 pc_list.append([x,y,z])

    for i in range(ymin,ymax):
        for j in range(xmin,xmax):
            if depth_map[i][j]<200:
                if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
                    x = depth_map[i][j]*(j-(img_w-1)/2)/focal
                    y = -depth_map[i][j]*(i-(img_height-1)/2)/focal
                    z = depth_map[i][j]
                    pc_list.append([x,y,z])
                    depth_statistic.append(depth_map[i][j])
    depth_statistic = np.array(depth_statistic)
    sigma = np.std(depth_statistic)
    miu = np.mean(depth_statistic)
    #repeated process
    if mode == 1:
        while sigma > 2:
            depth_stat = []
            for i in range(ymin,ymax):
                for j in range(xmin,xmax):
                    if depth_map[i][j]<miu + sigma and depth_map[i][j] > miu-sigma:
                        if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
                            depth_stat.append(depth_map[i][j])
            depth_stat = np.array(depth_stat)
            sigma = np.std(depth_stat)
            miu = np.mean(depth_stat)
    pc_list = np.array(pc_list)    
    if mode == 2:
        n_pc_list = []
        for i in pc_list:
            if i[2]>miu-sigma and i[2]<miu+sigma and i[1]>-1.3:
                n_pc_list.append(i)
        n_pc_list = np.array(n_pc_list)
        neps = 0.05
        minsp = 30
        clustering = DBSCAN(eps=neps, min_samples = minsp).fit(n_pc_list)
        counting = 0
        num = 0
        for i in range(np.max(clustering.labels_)):
            if counting < clustering.labels_[ clustering.labels_ == i ].size:
                counting = clustering.labels_[ clustering.labels_ == i ].size
                num = i
        # print("num",num)
        # print("len",i)
        # print("counting",counting)
        # print(n_pc_list.shape)
        # print(n_pc_list[clustering.labels_ == num].shape)
        # print(n_pc_list[clustering.labels_ == num][:][2].shape)
        # sys.exit()
        zmin = np.min(n_pc_list[clustering.labels_ == num].T[2])
        zmax = np.max(n_pc_list[clustering.labels_ == num].T[2])

    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.open3d.utility.Vector3dVector(pc_list)

    #For Corner
    if depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal > depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal:
        cxmin = depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal
    else:
        cxmin = depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal
    if depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal < depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal:
        cymax = depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal
    else:
        cymax = depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal
    if depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal <depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal:
        cxmax = depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal
    else:
        cxmax = depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal
    if depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal > depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal:
        cymin = depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal
    else:
        cymin = depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal
    line_set = [[cxmin,-cymin,miu-sigma],
                [cxmin,-cymax,miu-sigma],
                [cxmax,-cymax,miu-sigma],
                [cxmax,-cymin,miu-sigma],
                [cxmin,-cymin,miu+sigma],
                [cxmin,-cymax,miu+sigma],
                [cxmax,-cymax,miu+sigma],
                [cxmax,-cymin,miu+sigma],]
    if mode == 2:
        line_set = [[cxmin,-cymin,zmin],
                    [cxmin,-cymax,zmin],
                    [cxmax,-cymax,zmin],
                    [cxmax,-cymin,zmin],
                    [cxmin,-cymin,zmax],
                    [cxmin,-cymax,zmax],
                    [cxmax,-cymax,zmax],
                    [cxmax,-cymin,zmax],]
    # line_set = [[depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal,-depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal,miu-sigma],
    #             [depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal,-depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal,miu-sigma],
    #             [depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal,-depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal,miu-sigma],
    #             [depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal,-depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal,miu-sigma],
    #             [depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal,-depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal,miu+sigma],
    #             [depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal,-depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal,miu+sigma],
    #             [depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal,-depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal,miu+sigma],
    #             [depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal,-depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal,miu+sigma],
    # ]
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]
    #For bounding box
    # line_set = [[depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal,-depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal,depth_map[ymin][xmin]],
    #             [depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal,-depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal,depth_map[ymax][xmin]],
    #             [depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal,-depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal,depth_map[ymax][xmax]],
    #             [depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal,-depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal,depth_map[ymin][xmax]],
    # ]
    # lines = [[0, 1], [1, 2], [2, 3], [0, 3],]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    bbx = open3d.open3d.geometry.LineSet()
    bbx.points = open3d.open3d.utility.Vector3dVector(line_set)
    bbx.lines = open3d.open3d.utility.Vector2iVector(lines)
    bbx.colors = open3d.open3d.utility.Vector3dVector(colors)

    COR = open3d.open3d.geometry.TriangleMesh.create_coordinate_frame(size = 10, origin = [0,0,0])
    open3d.open3d.visualization.draw_geometries([pcd,bbx])

def add_bbx(point_set, is_pred = True):
    """
    Input:
        point_set: (8,3) array; with the order of 
                  5 -------- 6
                 /|         /|
                1 -------- 2 .
                | |        | |
                . 4 -------- 7
                |/         |/
                0 -------- 3
        is_pred: bool; if this bouding box is predicted or ground truth (maybe helpful if you want to visualize both prediction and ground truth)
    Output:
        bbx: LineSet format in open3d
    Effect:
        This function will convert 8 corner points into lineset which open3d could draw
    """
    # point_set should be 8 points
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]
    if is_pred:
        colors = [[1, 0, 0] for _ in range(len(lines))]
    else:
        colors = [[0, 1, 0] for _ in range(len(lines))]
    bbx = open3d.open3d.geometry.LineSet()
    bbx.points = open3d.open3d.utility.Vector3dVector(point_set)
    bbx.lines = open3d.open3d.utility.Vector2iVector(lines)
    bbx.colors = open3d.open3d.utility.Vector3dVector(colors)
    return bbx

def repeated_sigma(bbx2d, depth_map,use_repeated,use_DBSCAN,save_hist,frame,number, focal = 1000):
    """
    Input:
        bbx2d: (4,2) array; 2d bbx detection result, containing (xmin,ymin,xmax,ymax)
        depth_map: (h,w) array; depth map
        use_repeated: bool; whether to use repeated sigma rejection
        use_DBSCAN: bool; whether to use DBSCAN
        save_hist: bool; whether to save depth histogram
        frame: int; name of hist saving (frame+_+number+.png), not important if don't save hist
        number: int; same as frame
        focal: float; focal length of camera
    Output:
        bbx: LineSet format in open3d
    Effect:
        This function will convert 8 corner points into lineset which open3d could draw
    """
    xmin = bbx2d[0]
    ymin = bbx2d[1]
    xmax = bbx2d[2]
    ymax = bbx2d[3]
    # thres is the threshold to filter the flying points. Increase it if you want larger tolerance in point gap (will remain more points)
    thres = 0.5
    pc_list = []
    depth_statistic = []

    # You could change image size here
    img_w = 1280
    img_height = 720
    # Cut off image edges
    if ymax >= img_height:
        ymax = ymax - 1
    if xmax >= img_w:
        xmax = xmax - 1
    if ymin <=0:
        ymin = ymin +1
    if xmin <=0:
        xmin = xmin + 1
    # Convert from depth map into 3d voxels. Only consider points with depth < 200
    for i in range(ymin,ymax):
        for j in range(xmin,xmax):
            if depth_map[i][j]<200:
                if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
                    x = depth_map[i][j]*(j-(img_w-1)/2)/focal
                    y = -depth_map[i][j]*(i-(img_height-1)/2)/focal
                    z = depth_map[i][j]
                    # filter out grounds
                    if y>-1.2:
                        pc_list.append([x,y,z])
                        depth_statistic.append(depth_map[i][j])
    depth_statistic = np.array(depth_statistic)
    sigma = np.std(depth_statistic)
    miu = np.mean(depth_statistic)

    #repeated process
    if use_repeated:
        while sigma > 2:
            pc_list = []
            depth_stat = []
            for i in range(ymin,ymax):
                for j in range(xmin,xmax):
                    if depth_map[i][j]<miu + sigma and depth_map[i][j] > miu-sigma:
                        if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
                            x = depth_map[i][j]*(j-(img_w-1)/2)/focal
                            y = -depth_map[i][j]*(i-(img_height-1)/2)/focal
                            z = depth_map[i][j]
                            if y>-1.2:
                                pc_list.append([x,y,z])
                                depth_stat.append(depth_map[i][j])
            depth_stat = np.array(depth_stat)
            sigma = np.std(depth_stat)
            miu = np.mean(depth_stat)
            depth_statistic = depth_stat
    pc_list = np.array(pc_list)
    zmin = miu-sigma
    zmin_T = np.min(pc_list.T[2])
    zmax = miu+sigma
    #zmax = np.max(pc_list.T[2])
    if use_DBSCAN:
        # You can uncomment this part if you want to use sigma rejection to filter the voxels at first (not recommended. Only useful when the voxel quality is really poor). Then you need to replace pc_list with n_pc_list in the following
        # n_pc_list = []
        # for i in pc_list:
        #     if i[2]>miu-sigma and i[2]<miu+sigma and i[1]>-1.3:
        #         n_pc_list.append(i)
        # n_pc_list = np.array(n_pc_list)

        # DBSCAN setup. Increase neps if you want larger gap tolerance (might result in additional noise). minsp is the smallest number of points to be considered as a new cluster
        neps = 0.5
        minsp = 100
        clustering = DBSCAN(eps=neps, min_samples = minsp).fit(pc_list)
        counting = 0
        num = 0
        #print(max(clustering.labels_))
        for i in range(np.max(clustering.labels_)+1):
            if counting < clustering.labels_[ clustering.labels_ == i ].size:
                counting = clustering.labels_[ clustering.labels_ == i ].size
                num = i
        # print("num",num)
        # print("len",i)
        # print("counting",counting)
        # print(n_pc_list.shape)
        # print(n_pc_list[clustering.labels_ == num].shape)
        # print(n_pc_list[clustering.labels_ == num][:][2].shape)
        # sys.exit()
        #print(i)
        # Find the cluster with most points. That will be our target object.
        if max(clustering.labels_) > -1:
            print("DBSCAN works")
            zmin_T = np.min(pc_list[clustering.labels_ == num].T[2])
            zmax = np.max(pc_list[clustering.labels_ == num].T[2])
            depth_statistic = pc_list[clustering.labels_ == num].T[2]
            sigma = np.std(depth_statistic)
            miu = np.mean(depth_statistic)
            pc_list = pc_list[clustering.labels_ == num]

    #Plotting histgram
    if save_hist:
        sns.histplot(depth_statistic)
        plt.axvline(miu,0,30000)
        plt.axvline(miu+sigma,0,30000)
        plt.savefig("img\\dist_sigma_dbscan\\"+str(frame)+"_"+str(number)+".png")
        plt.cla()

    #For Corner. No use anymore
    # if depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal > depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal:
    #     cxmin = depth_map[ymin][xmin]*(xmin-(img_w-1)/2)/focal
    # else:
    #     cxmin = depth_map[ymax][xmin]*(xmin-(img_w-1)/2)/focal
    # if depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal < depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal:
    #     cymax = depth_map[ymax][xmin]*(ymax-(img_height-1)/2)/focal
    # else:
    #     cymax = depth_map[ymax][xmax]*(ymax-(img_height-1)/2)/focal
    # if depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal <depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal:
    #     cxmax = depth_map[ymax][xmax]*(xmax-(img_w-1)/2)/focal
    # else:
    #     cxmax = depth_map[ymin][xmax]*(xmax-(img_w-1)/2)/focal
    # if depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal > depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal:
    #     cymin = depth_map[ymin][xmin]*(ymin-(img_height-1)/2)/focal
    # else:
    #     cymin = depth_map[ymin][xmax]*(ymin-(img_height-1)/2)/focal


    cymin = np.min(pc_list.T[1])
    cymax = np.max(pc_list.T[1])

    cxmin = np.min(pc_list.T[0])
    cxmax = np.max(pc_list.T[0])

    line_set = [[cxmin,cymin,zmin_T],
                [cxmin,cymax,zmin_T],
                [cxmax,cymax,zmin_T],
                [cxmax,cymin,zmin_T],
                [cxmin,cymin,zmax],
                [cxmin,cymax,zmax],
                [cxmax,cymax,zmax],
                [cxmax,cymin,zmax],]

    # You can uncomment this part if you want to plot voxel for each object.
    # drawlines = add_bbx(line_set)
    # pcd = open3d.open3d.geometry.PointCloud()
    # pcd.points = open3d.open3d.utility.Vector3dVector(pc_list)
    # open3d.open3d.visualization.draw_geometries([pcd,drawlines])
    bev_limit = [[159+math.floor(np.min(pc_list.T[0])/0.1),math.floor(zmax/0.1)],
                 [159+math.floor(np.max(pc_list.T[0])/0.1),math.floor(zmax/0.1)],
                 [159+math.floor(np.min(pc_list.T[0])/0.1),math.floor(zmin_T/0.1)],
                 [159+math.floor(np.max(pc_list.T[0])/0.1),math.floor(zmin_T/0.1)]]
    return miu, zmin, zmin_T,line_set,bev_limit




if __name__=="__main__":
    mat_backend = matplotlib.get_backend()
    parser = argparse.ArgumentParser(description="A demo to run CREStereo.")
    parser.add_argument(
        "--model_path",
        default="crestereo_eth3d.mge",
        help="The path of pre-trained MegEngine model.",
    )
    parser.add_argument("--number",default = "1", help="The frame of input images")
    parser.add_argument(
        "--input_dir", default="img/test/", help="The path to the directory of the images."
    )
    parser.add_argument(
        "--size",
        default="1024x1536",
        help="The image size for inference. Te default setting is 1024x1536. \
                        To evaluate on ETH3D Benchmark, use 768x1024 instead.",
    )
    parser.add_argument(
        "--output", default="disparity.png", help="The path of output disparity."
    )
    args = parser.parse_args()

    assert os.path.exists(args.model_path), "The model path do not exist."

    yolo_model = torch.hub.load('ultralytics/yolov5','yolov5m6')
    yolo_result = yolo_model(args.input_dir+"/im0/"+args.number+".png")
    #yolo_result.show()
    yolo_result.save()
    yolo_result = yolo_result.pandas().xyxy[0]
    
    del yolo_model

    matplotlib.use(mat_backend)
    model_func = load_model(args.model_path)
    left = cv2.imread(args.input_dir+"/im0/"+args.number+".png")
    right = cv2.imread(args.input_dir+"/im1/"+args.number+".png")

    assert left.shape == right.shape, "The input images have inconsistent shapes."

    in_h, in_w = left.shape[:2]

    print("Images resized:", args.size)
    eval_h, eval_w = [int(e) for e in args.size.split("x")]
    left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    pred = inference(left_img, right_img, model_func, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    disp = np.array(disp)
    print(disp.shape)
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    parent_path = os.path.abspath(os.path.join(args.output, os.pardir))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    cv2.imwrite(args.output, disp_vis)


    # Change setup here
    img_w = 1280
    img_height = 720
    fov = 90
    focal = 1000
    b = 1
    depth_map = np.ones(disp.shape)
    depth_map[:] = focal*b/(np.abs(disp[:]))

    #For gt. If you want to use ground truth to replace predicted depth map, uncomment this part. gt should be in CARLA depth map format
    # gt = cv2.imread("img\\Carla\\depth007584.png",flags = cv2.IMREAD_COLOR)
    # gt = np.array(gt)
    # gt = gt[:, :, :3]
    # gt = gt[:,:,::-1]
    # gray_depth = ((gt[:,:,0] + gt[:,:,1] * 256.0 + gt[:,:,2] * 256.0 * 256.0)/((256.0 * 256.0 * 256.0) - 1))
    # gt = gray_depth * 1000
    # depth_map = gt

    yolo_img_path = 'runs\\detect\\exp\\'+args.number+'.jpg'
    draw_list = []

    # For whole voxels
    # thres is the threshold to filter the flying points. Increase it if you want larger tolerance in point gap (will remain more points)
    thres = 0.5
    pc_list = []
    for i in range(1,img_height-1):
            for j in range(1,img_w-1):
                if depth_map[i][j]<200:
                    if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
                        x = depth_map[i][j]*(j-(img_w-1)/2)/focal
                        y = -depth_map[i][j]*(i-(img_height-1)/2)/focal
                        z = depth_map[i][j]
                        if y>-1.2:
                            pc_list.append([x,y,z])


    pc_list = np.array(pc_list)
    print(pc_list.shape)
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.open3d.utility.Vector3dVector(pc_list)
    draw_list.append(pcd)

    # For coordinate axis. Uncomment this part if you want to label the center and direction of coordinate system in point clouds
    # COR = open3d.open3d.geometry.TriangleMesh.create_coordinate_frame(size = 10, origin = [0,0,0])
    # draw_list.append(COR)   
    
    # For bbx
    img = cv2.imread(yolo_img_path)
    bevs = np.zeros((320,500))
    for i in pc_list:
        if i[0]<=16 and i[0]>-16 and i[2]<50 and i[1]>-1.2:
            build_bev_points(i[0],i[2],bevs)
    bev_img = Image.fromarray(bevs.astype('uint8'),'L')
    draw = ImageDraw.Draw(bev_img)   
    for i in range(len(yolo_result)):
        is_interest = False

        # You can add more classes here as interesting class ()
        if yolo_result["name"][i] == "car" or yolo_result["name"][i]=="person" or yolo_result["name"][i] == "truck" or yolo_result["name"][i] =="motorcycle":
            is_interest = True
        bbx2d = [int(np.round_(yolo_result["xmin"][i])),int(np.round_(yolo_result["ymin"][i])),int(np.round_(yolo_result["xmax"][i])),int(np.round_(yolo_result["ymax"][i]))]
        # This line generate the 3D bounding box
        center, zmin, zmin_T,point_set,bev_corner = repeated_sigma(bbx2d,depth_map,False,True,False,args.number,i)
        # The followings are adding texts into yolo detection results. Not important if you only want to see 3D results
        cv2.putText(img,"AVG: %.2f" % center,(int(np.round_(yolo_result["xmin"][i])),int(np.round_(yolo_result["ymin"][i]))+10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2,cv2.LINE_AA)
        cv2.putText(img,"zmin: %.2f" % zmin,(int(np.round_(yolo_result["xmin"][i])),int(np.round_(yolo_result["ymin"][i]))+25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2,cv2.LINE_AA)
        cv2.putText(img,"true_zmin: %.2f" %zmin_T,(int(np.round_(yolo_result["xmin"][i])),int(np.round_(yolo_result["ymin"][i]))+40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2,cv2.LINE_AA)
        # Add to 3d draw_list
        bbx3d1 = add_bbx(point_set,True)
        draw_list.append(bbx3d1)
        # This is for bev
        if is_interest is True:
            #(depth,x)
            draw.line((bev_corner[2][1],bev_corner[2][0],bev_corner[3][1],bev_corner[3][0]),fill = 128, width = 3)
            draw.line((bev_corner[0][1],bev_corner[0][0],bev_corner[1][1],bev_corner[1][0]),fill = 128, width = 3)
            draw.line((bev_corner[1][1],bev_corner[1][0],bev_corner[3][1],bev_corner[3][0]),fill = 128, width = 3)
            draw.line((bev_corner[2][1],bev_corner[2][0],bev_corner[0][1],bev_corner[0][0]),fill = 128, width = 3)

    # Saving BEV
    # resolution: 0.1; x: +-16; z: +100
    plt.xlabel("depth")
    plt.ylabel("baseline")
    plt.imshow(bev_img)
    #plt.show()
    plt.savefig("img\\BEV_sigma_dbscan\\"+args.number+".jpg")

    # Saving 2D pictures
    #cv2.imwrite('img\\results\\'+args.number+'.jpg',img)

    # Draw 3d voxels
    print("Create voxel")
    open3d.open3d.visualization.draw_geometries(draw_list)

    # Clear yolo results
    shutil.move(yolo_img_path,"img\\yolo_result\\"+args.number+".jpg")
    os.rmdir("runs/detect/exp")


