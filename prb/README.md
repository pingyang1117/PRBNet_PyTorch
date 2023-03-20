# YOLOv7 with PRBNet

## Performance 
### MS COCO
#### P5 Model

| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | AP<sub>s</sub><sup>test</sup> |Model Description |
| :-- | :-: | :-: | :-: | :-: | :-: |  :-: |
| **YOLOX-x** | 640 | **51.5%** | **-** | **-** | **-** | - |
| **YOLOv7** | 640 | **51.4%** | **69.7%** | **55.9%** | **31.8%** | [yaml](https://github.com/pingyang1117/PRBNet_PyTorch/blob/main/prb/cfg/training/yolov7.yaml)|
|  |  |  |  |  |  |  |  |
| [**PRB-FPN-CSP**](https://drive.google.com/file/d/1vUglmai8lqfiEL2_nJZBZju-tGlrFL0I/view?usp=sharing) | 640 | **51.8%** | **70.0%** | **56.7%** | **32.6%** | [yaml](https://github.com/pingyang1117/PRBNet_PyTorch/blob/main/prb/cfg/training/PRB_Series/PRB-FPN-CSP.yaml)|
| [**PRB-FPN**](https://drive.google.com/file/d/1XQ2hSXq3fAWoH1qBynrMZwYSzPGe78nT/view?usp=sharing) | 640 | **52.5%** | **70.4%** | **57.2%** | **33.4%** | [yaml](https://github.com/pingyang1117/PRBNet_PyTorch/blob/main/prb/cfg/training/PRB_Series/PRB-FPN.yaml) |
|  |  |  |  |  |  |  |  |

#### P6 Model
| Model | Test Size | AP<sup>test</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | Params (M) |Model Description |
| :-- | :-: | :-: | :-: |  :-: | :-: |  :-: |
| **YOLOv7-D6** | 1280 | **56.6%** | **74.0%** | **61.8%** | **154.7M** | 
| **YOLOv7-E6E** | 1280 | **56.8%** | **74.4%** | **62.1%** | **151.7M**| 
|  |  |  |  |  |  |  |  | 
| [**PRB-FPN6-2PY**](https://drive.google.com/file/d/1kxmVqGe-j9rVSUbg-122Q7hwwbeQACGM/view?usp=sharing) | 1280 | **55.9%** | **73.7%** | **61.1%** | **137.5M**| [yaml](https://github.com/pingyang1117/PRBNet_PyTorch/blob/main/prb/cfg/training/PRB_Series/PRB-FPN6-2PY.yaml)|
| [**PRB-FPN6-3PY**](https://drive.google.com/file/d/1vcMgBM6KseSZKHjUuRhpLiVA4TswDzYu/view?usp=sharing) | 1280 | **56.7%** | **74.2%** | **61.9%** | **184.5M**| [yaml](https://github.com/pingyang1117/PRBNet_PyTorch/blob/main/prb/cfg/training/PRB_Series/PRB-FPN6-3PY.yaml)|
|  |  |  |  |  |  |  |   |

## Installation & Getting started

Please refer to the [yolov7 README](./yolov7_README.md) to get started.

## Testing

Tested with: Python 3.8.0, Pytorch 1.12.0+cu117

[`prb-fpn.pt`](https://drive.google.com/file/d/1hhOGyPHogXIe0MrMw9ReJLAuDcvRbdCI/view?usp=sharing) [`prb-fpn-csp.pt`](https://drive.google.com/file/d/1vUglmai8lqfiEL2_nJZBZju-tGlrFL0I/view?usp=sharing) 
[`PRB-FPN6-2PY.pt`](https://drive.google.com/file/d/1kxmVqGe-j9rVSUbg-122Q7hwwbeQACGM/view?usp=sharing) 
[`PRB-FPN6-3PY.pt`](https://drive.google.com/file/d/1vcMgBM6KseSZKHjUuRhpLiVA4TswDzYu/view?usp=sharing) 

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights prb-fpn.pt --name prb-fpn_640_val
```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.70304
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.57118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.36666
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.56971
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.38975
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.65053
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.70243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.54643
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.74958
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.84504
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

## Training

Data preparation

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training for P5 model

``` shell
# train prb-fpn-csp models
python train.py --workers 8 --device 0 --batch-size 36 --data data/coco.yaml --epochs 400 --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-CSP.yaml --weights '' --name PRB-FPN-CSP --hyp data/hyp.scratch.p5.yaml

# train prb-fpn models
python train.py --workers 8 --device 0 --batch-size 25 --data data/coco.yaml --epochs 400 --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN.yaml --weights '' --name PRB-FPN --hyp data/hyp.scratch.p5.yaml
```

Multiple GPU training

``` shell
# train prb-fpn-csp models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 144 --data data/coco.yaml --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-CSP.yaml --weights '' --name PRB-FPN-CSP-4GPU --hyp data/hyp.scratch.p5.yaml

# train prb-fpn models
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 200 --data data/coco.yaml --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN.yaml --weights '' --name PRB-FPN-8GPU --hyp data/hyp.scratch.p5.yaml

```

Single GPU training for P6 model

``` shell
# train PRB-FPN6-2PY models
python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/coco.yaml --epochs 400 --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-2PY.yaml --weights '' --name PRB-FPN6-2PY --hyp data/hyp.scratch.p6.yaml

# train PRB-FPN6-3PY models
python train_aux.py --workers 8 --device 0 --batch-size 28 --data data/coco.yaml --epochs 330 --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-3PY.yaml --weights '' --name PRB-FPN6-3PY --hyp data/hyp.scratch.p6.yaml

```



## Transfer learning

[`prb-fpn-csp.pt`](https://drive.google.com/file/d/1vUglmai8lqfiEL2_nJZBZju-tGlrFL0I/view?usp=sharing) [`prb-fpn_training.pt`](https://drive.google.com/file/d/1XQ2hSXq3fAWoH1qBynrMZwYSzPGe78nT/view?usp=sharing) 
 [`prb-fpn6-3py_training.pt`](https://drive.google.com/file/d/1_xAVNL2Zg2HGbJsh7n4bDehyFNrlXbDV/view?usp=sharing) 


Single GPU finetuning for custom dataset

``` shell
# finetune p5 models (prb-fpn-csp,prb-fpn)
python train.py --workers 8 --device 0 --batch-size 36 --data data/custom.yaml --epochs 300 --img 640 640 --cfg cfg/training/PRB_Series/PRB-FPN-CSP.yaml --weights 'prb-fpn-csp.pt' --name prb-fpn-csp-custom --hyp data/hyp.scratch.custom.yaml

# finetune p6 models (prb-fpn6-3py)
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/custom.yaml --epochs 300 --img 1280 1280 --cfg cfg/training/PRB_Series/PRB-FPN6-3PY.yaml --weights 'prb-fpn6-3py_training.pt' --name prb-fpn6-3py-custom --hyp data/hyp.scratch.custom.yaml
```

## Re-parameterization

See [reparameterization-prb.ipynb](reparameterization-prb.ipynb)

## Inference

On video:
``` shell
python detect.py --weights prb-fpn.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights prb-fpn.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

<div align="center">
    <a href="./">
        <img src="./figure/horses_prediction.jpg" width="65%"/>
    </a>
</div>

# Updates

- ~~[2022/08/31]: We are still modifying the YOLOv7-PRB structures to attain higher mAP at a lower param/FLOPs size with auxiliary head (IAuxDetect).~~
- ~~[2022/11/24]: We are still modifying the *re-parameterization* for PRB-FPN6-L structures to attain higher mAP at a lower inference time with auxiliary head (IAuxDetect).~~

## Citation

```
@ARTICLE{9603994,
  author={Chen, Ping-Yang and Chang, Ming-Ching and Hsieh, Jun-Wei and Chen, Yong-Sheng},
  journal={IEEE Transactions on Image Processing}, 
  title={Parallel Residual Bi-Fusion Feature Pyramid Network for Accurate Single-Shot Object Detection}, 
  year={2021},
  volume={30},
  number={},
  pages={9099-9111},
  doi={10.1109/TIP.2021.3118953}}
```

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```


## Acknowledgements


* https://github.com/WongKinYiu/yolov7



