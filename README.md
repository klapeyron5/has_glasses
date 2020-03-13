# has_glasses

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
Inference time on 1 sample is around 6ms on CPU  
Saved_model (tensorflow) takes less then 2MB space  
Model utilizing open-source face detector (bboxes) from https://github.com/TropComplique/FaceBoxes-tensorflow  
This detector was exported from tf1 to tf2  
The Model was trained on Celeba (utilizing info about attribute Eyeglasses)  
The Model was tested on Celeba test and example_data_glasses (40 images)  

## Metrics
