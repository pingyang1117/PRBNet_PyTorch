# PRBNet_Pytorch for Oriented Bounding Boxes

The codes here are mostly based on the [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb) with the following add-ons:

- From [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7):
	- Model common modules
	- Detection layers with implicit knowledge / auxiliary head (IDetect, IAuxDetect, IBin)
	- SimOTA loss
- PoolFormer modules for backbone model (experimental)
- The code implementations of [Parallel Residual Bi-Fusion Feature Pyramid Network](https://doi.org/10.1109/TIP.2021.3118953)

# Installation & Getting started

Please refer to the [yolov5_obb README](./yolov5obb_README.md) to get started.

# Results and Models

The below results are currently based on the DOTAv1.5-subsize1024-gap200 validation dataset. PRB-series have been trained for extra 100 epochs as they have larger sizes. Results on the test dataset will be updated soon.

|Model<br><sup>(download link) |Image Size<br><sup>(pixels) | TTA<br><sup>(multi-scale/<br>rotate testing) | OBB mAP<sup>val<br><sup>0.5<br>DOTAv1.5 | params<br><sup>(M) | GFLOPs<br><sup>@640 (B) |
|---|---|---|---|---|---|
| YOLOv5m ([Google](https://drive.google.com/file/d/1DB32HaSotKj2nyyv9caB8P3Bg1kD8mS5/view?usp=sharing)) | 640 | x | 69.621% | 21.6 | 50.5 |
| YOLOv7-PRB-Reparameterized ([Google](https://drive.google.com/file/d/1Ya8TDitDBDtZbejQwRF0Zo2VD9Ezf4ag/view?usp=sharing)) | 640 | x | 70.228% | 75.0 | 226.4 |

# Updates

- [2022/09/03]: We are still modifying the YOLOv7-PRB structures to attain higher mAP at a lower param/FLOPs size.

# Acknowledgements

Most of the codes are modified based on other amazing open-sourced repos. We would like to give special credits to the below authors:

* [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [hukaixuan19970627/yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [Thinklab-SJTU/CSL_RetinaNet_Tensorflow](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow)
* [jbwang1997/OBBDetection](https://github.com/jbwang1997/OBBDetection)
* [CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
* [sail-sg/poolformer](https://github.com/sail-sg/poolformer)




