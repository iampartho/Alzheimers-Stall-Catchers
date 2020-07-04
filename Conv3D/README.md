This is the list of submitted file related to 3D Cloud point classification pipeline.

## Description of Modification and Related Codes

Following table summarizes all the change in pipeline

| Serial | Model Description | Optimizer | Loss Function | Data Dimension | Dataset Relatd New Features | Other new features |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline Model | SGD(lr = 1e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 2 | Added two more dense layers | Adam(lr = 5e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 3 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 4 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | Balance Batch added in training |
| 5 | ResNet 3D or ResNet Mixed Convolution or ResNet (2+1)D | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 6 | ResNet 3D | Adam(lr = 1e-3,w_d = 1e-4) | " | 32 X 64 X 64 | Normalization, Augmentation, Changed DataLoader format | --

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
Or you can use [Inference_3DptCloud_3D_Model.ipynb](Inference_3DptCloud_3D_Model.ipynb) for 3D model inference.

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

All details regarding pytorch models documentation can be found <a href="https://pytorch.org/docs/stable/torchvision/models.html">here</a>



