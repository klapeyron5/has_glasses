# has_glasses
Predicts face image on two classes:  
either person on image has glasses or not  

## How to use as the task:
download and extract export.rar https://github.com/klapeyron5/has_glasses/blob/master/export_002.rar  
or **git clone** and go to the *export* directory (the latest version 002 is here)  
  
**cd export**  
**python task_py.py /path/to/faces/**  
or the same query:  
**python task_py.py /path/to/faces/ 0**  
0 here is number of GPU  

*older versions:*  
https://github.com/klapeyron5/has_glasses/blob/master/export_001.rar  

## Inference time:
If u wish to measure inference time of classificator only:  
**python task_py.py get_time -1**  
-1 here is CPU

## Description:
*v002:*
The Model is ResNet14_v2_mini *https://github.com/klapeyron5/klapeyron_py_utils/blob/master/klapeyron_py_utils/models/ResNet14_v2_mini3.py*  
The Model was trained on:  
* Celeba train set (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
* SoF (https://sites.google.com/view/sof-dataset)  
* MeGlass (https://github.com/cleardusk/MeGlass)  

with validation on Celeba val set  
The Model was tested on Celeba test set and example_data_glasses (40 images)  

*v001:*
The Model is ResNet14_v2_mini *https://github.com/klapeyron5/klapeyron_py_utils/blob/master/klapeyron_py_utils/models/ResNet14_v2_mini2.py*  
The Model was trained on Celeba train set (utilizing info about attribute Eyeglasses) with validation on Celeba val set  
The Model was tested on Celeba test set and example_data_glasses (40 images)  

The final pipeline is utilizing open-source face detector (bboxes) from https://github.com/TropComplique/FaceBoxes-tensorflow  
This detector was exported from tf1 to tf2  


## Metrics:
*v002:*  
Inference time on 1 sample is around 6ms on CPU  
Saved_model (tensorflow) of classificator takes around 2MB space  
EER on example_data_glasses is 0.0  
EER on Celeba test is 0.018

## Retrain the Model:
Run /train/train.py  
Get the data in proper format (better to request from me as archive)  
Don't forget to *pip install git+https://github.com/klapeyron5/klapeyron_py_utils* if u r running my code  

## Eval the Model
Run /train/eval.py  
Follow other points from **Retrain the Model** paragraph  
