# ScaleFormer
Code for IJCAI 2022 paper '[*ScaleFormer: Revisiting the Transformer-based Backbones from a Scale-wise Perspective for Medical Image Segmentation*](https://www.ijcai.org/proceedings/2022/0135.pdf)'

## 1.Dataset
- **Synapse:**  
The dataset we used is provided by TransUnet's authors. Please go to [link](https://github.com/Beckschen/TransUNet) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it (following the TransUnet's License).  
 
- **Monuseg:**  
The dataset we used is provided by UCTransNet's authors. Please go to [link](https://github.com/McGregorWwww/UCTransNet) for details.  
- **ACDC:**  
We will upload the dataset later.

## 2.Enviorments
- python 3.7
- pytorch 1.9.0
- torchvision 0.10.0
## 3.Train/Test
- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.  
```
python train.py
```
- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.
```
python test.py
```
## 4.Reference
- [TransUnet](https://github.com/Beckschen/TransUNet)
- [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
- [UcTransNet](https://github.com/McGregorWwww/UCTransNet)
