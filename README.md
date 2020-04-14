# Occluded Prohibited Items Detection: An X-ray Security Inspection Benchmark and De-occlusion Attention Module

![framework](https://github.com/OPIXray-author/OPIXray/blob/master/framework.jpg)

# Requirements

Python3.5

Pytorch:1.3.1



# Dataset

Download OPIXray dataset from [here](https://pan.baidu.com/s/1vhaW_dRSim-3Yu_vKGLqjQ). The password is rntm. Here are some sample images in the dataset.

![sample](https://github.com/OPIXray-author/OPIXray/blob/master/sample.png)

# Checkpoint

If you want to test the performance of DOAM, you can download our model from [here](https://pan.baidu.com/s/1OXvFODNcha2b3Jq5F6qkpw). The password is m9zk.

If you want to train your own model, you can download pre-trained weight of SSD on VOC0712 [here](https://pan.baidu.com/s/1KK7GdFeMd1VwimxUjD9Pug). The password is 3fbo.

# Usage

1. Clone the OPIXray repository

   git clone https://github.com/xl4533/OPIXray.git

2. If you want to train your model, execute the following command:

   (1) cd DOAM

   (2) Change the value of OPIXray_ROOT variable in DOAM/data/OPIXray.py file to the path where the training set is located, for example, OPIXray_ROOT = "/mnt/OPIXray_Dataset/train/"

   (3) python train.py --save_folder /mnt/model/DOAM/weights/ --image_sets /mnt/OPIXray_Dataset/train/train_knife.txt --transfer /mnt/ssd300_mAP_77.43_v2.pth

   **save_folder** is used to save the weight file obtained by training the model, 

   **image_sets**  is the path to a TXT file that saves all the picture names used for training, 

   **transfer** indicates the pre-trained weight of SSD on VOC0712

3. If you want to test our model, execute the following command:

   (1) cd DOAM 

   (2) Change the value of OPIXray_ROOT variable in DOAM/data/OPIXray.py file to the path where the testing set is located, for example, OPIXray_ROOT = "/mnt/OPIXray_Dataset/test/"

   (3) python test.py --trained_model /mnt/model/SSD/weights/DOAM.pth --imagesetfile /mnt/OPIXray_Dataset/test/test_knife.txt

   **trained_model** is the weight file you want to test

4. If you want to test our model with different occlusion level, execute the following command:

   (1) cd DOAM 

   (2) Change the value of OPIXray_ROOT variable in DOAM/data/OPIXray.py file to the path where the testing set is located, for example, OPIXray_ROOT = "/mnt/OPIXray_Dataset/test/"

   (3) python test.py --trained_model /mnt/model/SSD/weights/DOAM.pth --imagesetfile /mnt/OPIXray_Dataset/test/test_knife-1.txt **（occlusion  level 1）**

   (4) python test.py --trained_model /mnt/model/SSD/weights/DOAM.pth --imagesetfile /mnt/OPIXray_Dataset/test/test_knife-2.txt **（occlusion  level 2）**

   (5) python test.py --trained_model /mnt/model/SSD/weights/DOAM.pth --imagesetfile /mnt/OPIXray_Dataset/test/test_knife-4.txt **（occlusion  level 3）**

   

# Acknowledgement

In this project, we implemented DOAM on PyTorch based on [amdegroot](https://github.com/amdegroot/ssd.pytorch)

