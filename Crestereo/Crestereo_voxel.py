import os
import math
import megengine as mge
import megengine.functional as F
import argparse
import numpy as np
import cv2
import open3d
from PIL import Image
from matplotlib import pyplot as plt
from torch import arctan
from nets import Model


def load_model(model_path):
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = mge.load(model_path)
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict["state_dict"], strict=True)

    model.eval()
    return model


def inference(left, right, model, n_iter=20):
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

def get_transform(cx,cz,rotate):
    theta = math.arctan(cx,cz)
    alpha = rotate - theta
    x = math.sqrt(cx*cx+cz*cz)*math.cos(alpha)
    z = math.sqrt(cx*cx+cz*cz)*math.sin(alpha)
    return x,z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A demo to run CREStereo.")
    parser.add_argument(
        "--model_path",
        default="crestereo_eth3d.mge",
        help="The path of pre-trained MegEngine model.",
    )
    parser.add_argument(
        "--left", default="img/test/left.png", help="The path of left image."
    )
    parser.add_argument(
        "--right", default="img/test/right.png", help="The path of right image."
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

    model_func = load_model(args.model_path)
    left = cv2.imread("D:\\Weeks\\CREStereo-master\\CREStereo-master\\img\\Carla\\im0.png")
    right = cv2.imread("D:\\Weeks\\CREStereo-master\\CREStereo-master\\img\\Carla\\im1.png")

    assert left.shape == right.shape, "The input images have inconsistent shapes."

    in_h, in_w = left.shape[:2]

    print("Images resized:", args.size)
    eval_h, eval_w = [int(e) for e in args.size.split("x")]
    left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    pred = inference(left_img, right_img, model_func, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    disp_left = np.array(disp)
    print(disp.shape)
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    parent_path = os.path.abspath(os.path.join(args.output, os.pardir))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    cv2.imwrite(args.output, disp_vis)

    # Forward-Backward consistency
    # left_flip = np.zeros(left.shape)
    # for i in range(720):
    #     for j in range(1280):
    #         for k in range(3):
    #             left_flip[i][1279-j][k] = left[i][j][k]
    # right_flip = np.zeros(right.shape)
    # for i in range(720):
    #     for j in range(1280):
    #         for k in range(3):
    #             right_flip[i][1279-j][k] = right[i][j][k]

    # left_img = cv2.resize(left_flip, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    # right_img = cv2.resize(right_flip, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    # img = Image.fromarray(right_flip.astype('uint8'))
    # plt.title("flipped right image")
    # plt.imshow(img)
    # plt.show()
    # # img = Image.fromarray(left_flip.astype('uint8'))
    # # plt.imshow(img)
    # # plt.show()
    # pred = inference(right_img, left_img, model_func, n_iter=20)

    # t = float(in_w) / float(eval_w)
    # disp_flip = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    # disp_flip = np.array(disp_flip)
    # # change back
    # disp_right = np.zeros(disp_flip.shape)
    # for i in range(720):
    #     for j in range(1280):
    #             disp_right[i][1279-j] = -disp_flip[i][j]
    #print(disp.shape)

    # left = cv2.imread("img\\Carlap\\im0\\12.png",cv2.IMREAD_GRAYSCALE)
    # right = cv2.imread("img\\Carlap\\im1\\12.png",cv2.IMREAD_GRAYSCALE)
    # wsize = 31
    # max_disp = 128
    # sigma = 1.5
    # lmbda = 8000
    # left_matcher = cv2.StereoBM_create(max_disp,wsize)
    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # left_disp = left_matcher.compute(left,right)
    # right_disp = right_matcher.compute(right,left)
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)
    # filtered_disp = wls_filter.filter(left_disp,left,disparity_map_right=right_disp)
    # disp = np.array(filtered_disp)
    print(np.max(disp_left),np.min(disp_left))
    #For carla voxel
    img_w = 1280
    img_height = 720
    fov = 90
    focal = 1000
    b = 1.6
    img = Image.fromarray(disp_left.astype('uint8'),'L')
    plt.title("disparity by crestereo")
    plt.imshow(img)
    plt.show()
    depth_map = np.ones(disp_left.shape)
    depth_map[:] = focal*b/(np.abs(disp_left[:]))
    img = Image.fromarray(depth_map.astype('uint8'),'L')
    plt.title("depth_map by crestereo")
    plt.imshow(img)
    plt.show()
    thres = 0.5
    #Validation
    # gt = cv2.imread("img\\Carla\\depth007584.png",flags = cv2.IMREAD_COLOR)
    # gt = np.array(gt)
    # gt = gt[:, :, :3]
    # gt = gt[:,:,::-1]
    # gray_depth = ((gt[:,:,0] + gt[:,:,1] * 256.0 + gt[:,:,2] * 256.0 * 256.0)/((256.0 * 256.0 * 256.0) - 1))
    # gt = gray_depth * 1000
    # img = Image.fromarray(gt.astype('uint8'),'L')
    # plt.title("ground truth from carla")
    # plt.imshow(img)
    # plt.show()
    # difference = np.zeros(depth_map.shape)
    # count = 0
    # counting = 0
    # for i in range(1,img_height-1):
    #     for j in range(1,img_w-1):
    #         if (depth_map[i][j] - depth_map[i-1][j] < thres and depth_map[i][j] - depth_map[i+1][j] < thres and depth_map[i][j] - depth_map[i][j-1] < thres and depth_map[i][j] - depth_map[i][j+1] < thres) and depth_map[i][j]<200 and gt[i][j]<200:
    #             difference[i][j] = np.abs(depth_map[i][j] - gt[i][j])
    #             if difference[i][j] > 1:
    #                 counting = counting+1
    #             count = count + 1
    # print("rmse are ",np.sqrt((difference*difference).sum()/count))
    # print(count)
    # print(counting)
    # img = Image.fromarray(difference.astype('uint8'),'L')
    # plt.imshow(img)
    # plt.show()
    # gt_point=[]
    # pred_point = []
    # for i in range(1,img_height-1):
    #     for j in range(1,img_w-1):
    #         if gt[i][j]<200:
    #             if depth_map[i][j]<200:
    #                 if depth_map[i][j] - depth_map[i-1][j] < thres and depth_map[i][j] - depth_map[i+1][j] < thres and depth_map[i][j] - depth_map[i][j-1] < thres and depth_map[i][j] - depth_map[i][j+1] < thres:
    #                     gt_point.append(gt[i][j])
    #                     pred_point.append(depth_map[i][j])
    # print(len(gt_point))
    # plt.scatter(np.array(gt_point),np.array(pred_point))
    # plt.xlabel("gt")
    # plt.ylabel("prediction")
    # plt.plot()
    # plt.show()


    #Voxel
    pc_list = []
    for i in range(1,img_height-1):
        for j in range(1,img_w-1):
            if depth_map[i][j]<200:
                if np.abs(depth_map[i][j] - depth_map[i-1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i+1][j]) < thres and np.abs(depth_map[i][j] - depth_map[i][j-1]) < thres and np.abs(depth_map[i][j] - depth_map[i][j+1]) < thres:
                    x = depth_map[i][j]*(j-600)/focal
                    y = depth_map[i][j]*(i-182)/focal
                    z = depth_map[i][j]
                    pc_list.append([x,y,z])
                # if round(j+disp_left[i][j])>=0 and round(j+disp_left[i][j])<=1279:
                #     if np.abs(disp_left[i][j]+disp_right[i][round(j+disp_left[i][j])])<=5:
                #         x = depth_map[i][j]*j/focal
                #         y = depth_map[i][j]*i/focal
                #         z = depth_map[i][j]
                #         pc_list.append([x,y,z])
                # else:
                #     if depth_map[i][j] - depth_map[i-1][j] < thres and depth_map[i][j] - depth_map[i+1][j] < thres and depth_map[i][j] - depth_map[i][j-1] < thres and depth_map[i][j] - depth_map[i][j+1] < thres:
                #         x = depth_map[i][j]*j/focal
                #         y = depth_map[i][j]*i/focal
                #         z = depth_map[i][j]
                #         pc_list.append([x,y,z])
            # x = gt[i][j]*j/focal
            # y = gt[i][j]*i/focal
            # z = gt[i][j]
            # pc_list.append([x,y,z])
    pc_list = np.array(pc_list)
    print(pc_list.shape)
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = open3d.open3d.utility.Vector3dVector(pc_list)
    COR = open3d.open3d.geometry.TriangleMesh.create_coordinate_frame(size = 15, origin = [0,0,0])
    open3d.open3d.visualization.draw_geometries([pcd,])

