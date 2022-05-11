[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/6d-rotation-representation-for-unconstrained/head-pose-estimation-on-biwi)](https://paperswithcode.com/sota/head-pose-estimation-on-biwi?p=6d-rotation-representation-for-unconstrained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/6d-rotation-representation-for-unconstrained/head-pose-estimation-on-aflw2000)](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000?p=6d-rotation-representation-for-unconstrained)

# 6D Rotation Representation for Unconstrained Head Pose Estimation (Pytorch)

A fork of the [official repository](https://github.com/thohemp/6DRepNet). This repository contains scripts for the deployment of the model to process multiple videos.

## Validation
Trained on 300W-LP, Test on AFLW2000 and BIWI

|     Dataset / Release     	|  Yaw 	| Pitch 	| Roll 	|  MSE  	|
|:-------------------------:	|:----:	|:-----:	|:----:	|:-----:	|
| AFLW2000 Official Release 	| 3.63 	|  4.91 	| 3.37 	|  3.97 	|
|       AFLW2000 Fork       	| 3.62 	|  4.90 	| 3.37 	|  3.96 	|
|   BIWI Official Release   	| 3.24 	|  4.48 	| 2.68 	|  3.47 	|
|         BIWI Fork         	| 3.24 	|  4.48 	| 2.68 	|  3.46 	|

## Additions
In the sixdrepnet sub-directory I have added modules to extract headpose angles for all detected faces for all videos in a directory. 


## **Citations**

Please cite the original work:

```
@misc{hempel20226d,
      title={6D Rotation Representation For Unconstrained Head Pose Estimation}, 
      author={Thorsten Hempel and Ahmed A. Abdelrahman and Ayoub Al-Hamadi},
      year={2022},
      eprint={2202.12555},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
