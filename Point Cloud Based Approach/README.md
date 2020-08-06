Point Cloud based approach is illustrated below

![Base Image](Modeldesign.jpg)

## Training Hyperparameters

* Class Balance Loss with Beta = 2.0, Gamma = 0.9999, loss_type = "softmax"
* Adam optimizer
* Learning rate = 5e-4 (for resnext 101) 
* Learning rate scheduler (StepLR , step_size = 15, gamma = 0.7)
* Batch size = 60
