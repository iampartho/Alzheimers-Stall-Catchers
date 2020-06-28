This is the list of submitted file related to 3D Cloud point classification pipeline.

## Description of Modification and Related Codes

Following table summarizes all the change in pipeline

| Serial | Model Description | Optimizer | Loss Function | Data Dimension | Dataset Relatd New Features | Other new features |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline Model | SGD(lr = 1e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 2 | Added two more dense layers | Adam(lr = 5e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 3 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 4 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | Adding Balance Batch in training |
| 5 | ResNet 3D 18 | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 6 | ResNet Mixed Convolution | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 7 | ResNet Mixed Convolution | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 8 | ResNet (2+1)D | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |


- Serial 1  (Baseline Pipeline) : [3DptCloudofAlzheimer_Baseline.ipynb](3DptCloudofAlzheimer_Baseline.ipynb) contains the baseline pipeline code
- Serial 2 : [3DptCloudofAlzheimer_modified.ipynb](3DptCloudofAlzheimer_modified.ipynb) contains that modified code
- Serial 3 : To add weight decay (adding L2 regularization to reduce overfitting) change code in **Model Hyperparameters** section like 
```bash
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
```
- Serial 4 : To add balance batch change code in **Dataloader** section 
```bash
train_loader = torch.utils.data.DataLoader(train,sampler=BalancedBatchSampler(train), batch_size = batch_size, num_workers=4)
```

N.B : If training stops somehow at the end of one epoch then add this line at the end of each epoch( in last line of epoch loop)
- Serial 5 : To implement ResNet 3D as model we can use torchvision.models library. We have to change code in **Model Code** section
```bash
from torchvision.models.video import r3d_18
model = r3d_18(pretrained = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.fc.out_features = 2
```
For changing the [Inference_3DptCloud.ipynb](Inference_3DptCloud.ipynb) we have to change similarly in **Model Code** section

```bash
from torchvision.models.video import r3d_18
model = r3d_18(pretrained = False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.fc.out_features = 2
model.load_state_dict(torch.load(checkpoint_model))
```
- Serial 6 : To implement ResNet Mixed Convolution we have to change similarly like **serial 5** which is just change of library function
```bash
from torchvision.models.video import mc3_18
model = mc3_18(pretrained = True)
```
And similar change will come into inference code(like Serial 5).
- Serial 7 : To implement State of the art ResNet (2+1)D we have to change library function similarly like **serial 6** 
```bash
from torchvision.models.video import r2plus1d_18
model = r2plus1d_18(pretrained = True)
```
And same goes for inference code(like Serial 5). All the details about 3D models can be found in <a href="https://pytorch.org/docs/stable/torchvision/models.html">Pytorch Model documentations </a> 

- Baseline Model (Baseline pipeline) : 



