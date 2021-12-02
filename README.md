# [Occluded Prohibited Items Detection: An X-ray Security Inspection Benchmark and De-occlusion Attention Module](https://arxiv.org/abs/2004.08656)

This repository is the official implementation of **De-occlusion Attention Module(DOAM)** and **OPIXray dataset**. Paper link: https://arxiv.org/abs/2004.08656.

#### DOAM framework:

![DOAM](images/framework.png)

#### OPIXray dataset:

<div align=center>
    <img src="images/samples.png" width="80%" height="80%"/>
</div>

[![license](https://img.shields.io/github/license/:user/:repo.svg)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Install](#install)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgement](#Acknowledgement)  
- [License](#license)
 
## Overview

Security inspection often deals with a piece of baggage or suitcase where objects are heavily overlapped with each other, resulting in an unsatisfactory performance for prohibited items detection in X-ray images. 

In this work, first, we contribute a high-quality dataset named OPIXray, each of which is annotated manually by professional inspectors from an international airport. To the best of our knowledge, this is the first dataset specifically designed for object detection in security inspection. Second, we propose a De-occlusion Attention Module (DOAM) that can be inserted as a plug-and-play module into most detectors, aiming at detecting occluded prohibited items in X-ray images. Finally, we evaluate our method on the OPIXray dataset and compare it to several baselines, including popular methods for detection and attention mechanisms. As is shown from the results, our proposed method significantly outperforms existing models. 

## Dataset

In this project, we build the first dataset specially designed for occluded prohibited items detection in security inspection.

<div align=center>
    <img src="images/occlusion_levels.png" width="50%" height="50%"/>
</div>

- All images of OPIXray dataset are scanned by security inspection machine and annotated manually by professional inspectors from an international airport, and the standard of annotating is based on the standard of training security inspectors.

- OPIXray dataset contains a total of 8885 X-ray images(7019 for training, 1776 for testing), including 5 categories of cutters, namely, Folding Knife, Straight Knife, Scissor, Utility Knife, Multi-tool Knife. 

	<div align=center>
	    <img src="images/statistics.png" width="50%" height="50%"/>
	</div>

- In order to study the impact brought by object occlusion levels, we divide the testing set into three subsets and name them Occlusion Level 1 (OL1), Occlusion Level 2 (OL2) and Occlusion Level 3 (OL3), where the number indicates occlusion level of prohibited items in images.

	<div align=center>
	    <img src="images/occlusion_distribution.png" width="50%" height="50%"/>
	</div>
- The information structure of annotation file is as follows: 
	image name, category, top-left position of prohibited item (x1, y1), bottom-right position of prohibited item (x2, y2).

## Install

1. Install pytorch

	The code is tested on python3.5 and pytorch1.3.1

2. Clone the repository

	```
	git clone https://github.com/OPIXray-author/OPIXray.git
	```

3. Access OPIXray dataset

	OPIXray dataset is available only for ACADEMIC PURPOSE, if you are interested in our dataset, feel free to contact rstao@buaa.edu.cn.
## Usage

#### Training

1. change root to `DOAM`

    ```
    cd DOAM
    ```

2. modify `OPIXray_ROOT` in `data/OPIXray.py` :

    ```
    OPIXray_ROOT = 'OPIXray_Dataset/train/'	# path to trainset of OPIXray Dataset
    ```

3. run `train.py`

    ```
    python train.py --image_sets 'OPIXray_Dataset/train/train_knife.txt' --transfer './weights/ssd300_mAP_77.43_v2.pth' --save_folder './checkpoints'
    ```

    - `--save_folder`: the path to save the checkpoint while training.
    - `--dataset_root`: the path to the training dataset.
    - `--image_sets`: the path to a TXT file that saves all the picture names used for training.
    - `--transfer`: indicates the path to SSD pre-trained weight(available at [here](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth)).
#### Testing

1. change root to `DOAM`

    ```
    cd DOAM
    ```

2. modify `OPIXray_ROOT` in `data/OPIXray.py` :

    ```
    OPIXray_ROOT = "OPIXray_Dataset/test/"	# path to testset of OPIXray Dataset
    ```
    
3. run `test.py`

    ```
    python test.py --imagesetfile 'OPIXray_Dataset/test/test_knife.txt' --trained_model './weights/DOAM.pth' 
    ```
    
    - `--imagesetfile`: the path to a TXT file that saves all the picture names used for testing.
    	- `test_knife.txt`: total testset
    	- `test_knife-1.txt`: occlusion level 1 (no or slight occlusion).
    	- `test_knife-2.txt`: occlusion level 2 (partial occlusion).
    	- `test_knife-3.txt`: occlusion level 3 (severe or full occlusion).
## License

The images and the corresponding annotations in OPIXray Dataset can be used **ONLY for academic purposes**, **NOT for commercial purposes**. 

Copyright © 2020 by State Key Lab of Software Development Environment, Beihang University

All rights reserved.
