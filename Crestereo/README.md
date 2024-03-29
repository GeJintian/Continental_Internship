#### Install
To run this file, you should install yolov5 and Crestereo.<br>
For yolov5, please follow: https://github.com/ultralytics/yolov5 <br>
For Crestereo, please follow: https://github.com/megvii-research/CREStereo <br>
Please also install packages: matplotlib, seaborn, sklearn, shutil, open3d, PIL, opencv-python <br>
Please move sigma_reject.py to the root directory of Crestereo. You will also need to download pre-trained models.<br>
#### Dataset
I assume that the dataset will look like: ```.../image/im0/1.png``` (for left img) and ```.../image/im1/1.png``` (for right img). So the program will take ```--input_dir``` (which is ```.../image``` here) and ```--number``` (which is 1 here).<br>
You will also need to create some empty directories for result saving. It should also under the root directory of Crestereo, as:<br>
CREStereo-master:.<br>
└───img<br>
\\\\\ ├───BEV_sigma_dbscan<br>
\\\\\ ├───dist_sigma_dbscan<br>
\\\\\ └───yolo_result<br>
Notice that I run this program on windows. So I use "\\" to separate directories in the code. If you run it on linux, you may need to change it.<br>
#### Run inference
This is an example of running code. You can check the file for details <br>
```python sigma_reject.py --model_path crestereo_eth3d.mge --input_dir .../image --size 720x1280 --output disparity.png --number 7```
