This is the list of submitted file related to 3D Cloud point classification pipeline.

## Description of Modification and Related Codes

Following table summarizes all the change in pipeline

| Serial | Model Description | Optimizer | Loss Function | Data Dimension | Dataset Relatd New Features | Other new features |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline Model | SGD(lr = 1e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 2 | Added two more dense layers | Adam(lr = 5e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 3 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 4 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | Balance Batch added in training |
| 5 | ResNet18 3D or ResNet18 Mixed Convolution or ResNet18 (2+1)D | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 6 | ResNet18 3D | Adam(lr = 1e-3,w_d = 1e-4) | " | 32 X 64 X 64 (bigger size) | Normalization, Augmentation, Changed DataLoader format | -- |
| 7 | -- | Ranger(lr = 1e-3,w_d = 1e-4) | " | 32 X 64 X 64 | -- | -- |
| 8 | 3D ResNet34,50,101,152,200, 3D ResNeXt50,101 | -- | " | 32 X 64 X 64 | -- | -- |

- **Serial 1**  (Baseline Pipeline) : [3DptCloudofAlzheimer_Baseline.ipynb](3DptCloudofAlzheimer_Baseline.ipynb) contains the baseline pipeline code
- **Serial 2** : [3DptCloudofAlzheimer_modified.ipynb](3DptCloudofAlzheimer_modified.ipynb) contains that modified code
- **Serial 3** : To add weight decay (adding L2 regularization to reduce overfitting) change code in **Model Hyperparameters** section like 
```bash
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
```
- **Serial 4** : Adding balance batch in training can be easy as we need to import library from **sampler.py** and change code in **Dataloader** section
```bash
from sampler import BalancedBatchSampler  #importing library

train_loader = torch.utils.data.DataLoader(train,sampler=BalancedBatchSampler(train), batch_size = batch_size, num_workers=4)
```
N.B : If you face any problem at the end of one epoch then you can add this line at the end of one epoch(end line of epoch loop)

- **Serial 5** :
To implement ResNet 3D variants we can change code in **Model Code** section like
```bash
from torchvision.models.video import r3d_18
from torchvision.models.video import r2plus1d_18
from torchvision.models.video import mc3_18
#which model we want we have to select it
model = r3d_18(pretrained = True)
model = r2plus1d_18(pretrained = True)
model = mc3_18(pretrained = True)
model.fc.out_features = 2
```
For inference code [Inference_3DptCloud.ipynb](Inference_3DptCloud.ipynb) we have to change similary the **Model Code** section
```bash
from torchvision.models.video import r3d_18
from torchvision.models.video import r2plus1d_18
from torchvision.models.video import mc3_18
#select which model weight you have
model = r3d_18(pretrained = False)
model = r2plus1d_18(pretrained = False)
model = mc3_18(pretrained = False)
model.fc.out_features = 2
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_model))
```
Or you can use [Inference_3DptCloud_3D_Model.ipynb](Inference_3DptCloud_3D_Model.ipynb) for 3D model inference. All details regarding pytorch models documentation can be found <a href="https://pytorch.org/docs/stable/torchvision/models.html">here</a>


- **Serial 6** :

Custom Dataset Class:
```
import torch

class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, files_list, source_path):
        self.list_IDs = list(files_list.keys())
        self.labels = files_list
        self.path = source_path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        original_name = ID.replace(".mp4", "")
        f = self.path + original_name + ".pt"

        X = torch.load(f)

        y = self.labels[ID]
        y = torch.tensor(int(y))

        return X, y
```

Augmentation: Change to False for disabling augmentation
```
train_list = VoxelTensor(pc_path).save(files_list=train_list, dst_path=train_files_directory, dim=dimension, augment=True)
```
- **Serial 7** :
To use ranger optimizer(RAdam and LookAhead) change this in the section **Model Hyperparameters** . Must keep ranger.py in the directory.
``` 
from ranger import Ranger 
optimizer = Ranger(model.parameters(), lr=learning_rate, weight_decay=1e-4)
```
- **Serial 8** :
To use deeper model than the ResNet 18 we can use 3D ResNet 34, 50, 101, 152, 200 and 3D ResNeXt 50, 101. For ResNet keep 'resnet.py' and for ResNeXt keep 'resnext.py' in directory. Then change the section **Model Code** to

``` 
#For resnext 50/ 101/ 152
import resnext
model = resnext.resnext50(
                num_classes=2,
                shortcut_type='B',
                cardinality=32,
                sample_size=64,
                sample_duration=32)
``` 

``` 
#For resnet 10/ 18/ 50/ 101/ 152/ 200
import resnet
model = resnet.resnet101(
                num_classes=2,
                shortcut_type='B',
                sample_size=64,
                sample_duration=32)
``` 
To know more details you can see <a href="https://github.com/aia39/Efficient-3DCNNs">this</a> repo. 
