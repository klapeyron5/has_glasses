# has_glasses
Predicts face image on two classes:  
either person on image has glasses or not  

## How to use as the task:
cd export  
**python task_py.py /path/to/faces/ /device:GPU:0**  
or the same query:  
**python task_py.py /path/to/faces/**  

## Inference time:
If u wish to measure inference time of classificator only:  
**python task_py.py get_time /device:CPU:0**  

## Description:
Model is ResNet14_v2_mini  
Model is utilizing open-source face detector (bboxes) from https://github.com/TropComplique/FaceBoxes-tensorflow  
This detector was exported from tf1 to tf2  
The Model was trained on Celeba train set (utilizing info about attribute Eyeglasses) with validation on Celeba val set  
The Model was tested on Celeba test set and example_data_glasses (40 images)  

## Metrics:
Inference time on 1 sample is around 6ms on CPU  
Saved_model (tensorflow) of classificator takes less then 2MB space  
EER on example_data_glasses is 0.0  
EER on Celeba test is 0.02

## Retrain the Model:
Run /train/train.py  
Get the data in proper format (better to request from me as archive)  
Don't forget to install *pip install git+https://github.com/klapeyron5/klapeyron_py_utils*  
If u r running my code  

## Eval the Model
Run /train/eval.py  
Follow other points from **Retrain the Model** paragraph  
