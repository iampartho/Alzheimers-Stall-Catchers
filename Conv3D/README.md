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
| 9 | 3D ResNets , 3D ResNeXts Pretrained weight | -- | " | 32 X 64 X 64 | -- | -- |
|10 | RESNET101 | Adam lr=5e-4 w_d=8e-4 | CrossEntropy | 32 X 64 X 64 | Point Cloud Mask applied to Imageset, Reshape to fit to dimension, Augmentation | Manual selection of LR, WD, Multistage Training |
| 11 | '' | '' | '' | 32 X 64 X 64 | -- | Using Automatic Mixed Precision |
| 12 | '' | '' | Class Balance Weight | 32 X 64 X 64 | -- | -- |

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

- **Serial 9** :
If we want to leverage 3D models then loading pretrained weight is the convenient way to start. To use pretrained weight(trained on Kinetics) we have to change as following. Change the section **Model Code** to
``` 
#For resnet 10/ 18/ 50/ 101/ 152/ 200
import resnet
model = resnet.resnet101(
                num_classes=600,
                shortcut_type='B',
                sample_size=64,
                sample_duration=32)
                
model = model.cuda()
model = nn.DataParallel(model, device_ids=None)
pytorch_total_params = sum(p.numel() for p in model.parameters() if
                        p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)

pretrain = torch.load(checkpoint_model, map_location=torch.device('cpu'))
#assert opt.arch == pretrain['arch']
model.load_state_dict(pretrain['state_dict'])
model.module.fc = nn.Linear(model.module.fc.in_features, num_classes)
model.module.fc = model.module.fc.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

Note that, you can find pretrained weight file <a href="https://drive.google.com/drive/folders/1eggpkmy_zjb62Xra6kQviLa67vzP_FR8">here</a>. For ResNet 101 use "kinetics_resnet_101_RGB_16_best.pth" and assign it to "checkpoint_model" variable. Similarly applicable for ResNeXt model.


- **Serial 10** :

Train Code: 3DptCloudofAlzheimer_networks.ipynb

Inference: Inference_3D_networks.ipynb

Data importing now occurs in partitions due to large size of dataset
```
!jar -xf "/content/drive/My Drive/SayeedColab/Alzheimer Data/micro_1.zip";
print("partition 1 imported")
```

Dataloader directly loads saved tensor files so training time has a significant improvement
```
X = torch.load(sequence_path + ".pt")
```

Augmentation now happens during training. While running dataloader, you can mention batch size, augmentation enabling and augmentation volume
```
batch_size = 128

split_number = 0

augmentation_enabled = True     # Set to false for no augmentation
augment_volume = 16             # How many times data should be augmented

if augmentation_enabled ==  True:       # If augmentation enabled, resize batch to fit in gpu
  batch_size = int(batch_size/augment_volume)
```

Augmenter Function takes a minibatch and returns an augmentated batch of augment_volume times data
```
def augment(images, labels, augment_volume=2):

  [instances, channel, depth, height, width] = images.shape

  images_aug = torch.zeros((instances*augment_volume, channel, depth, height, width))
  labels_aug = torch.zeros(instances*augment_volume)

  for aug_type in range(augment_volume):
    augmented = im_aug(images, aug_type)
    
    images_aug[aug_type*instances:(aug_type+1)*instances, :, :, :, :] = augmented
    labels_aug[aug_type*instances:(aug_type+1)*instances] = labels

  return images_aug.float(), labels_aug.long()
```

Model being used currently resnet101
```
import resnet
model = resnet.resnet101(
                num_classes=2,
                shortcut_type='B',
                sample_size=64,
                sample_duration=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_state_dict(torch.load(checkpoint_model))  # comment for training from scratch
```

**TIP:** This training could take several hours depending on how many iterations you chose in the .cfg file. You will want to let this run as you sleep or go to work for the day, etc. However, Colab Cloud Service kicks you off it's VMs if you are idle for too long (30-90 mins).

To avoid this hold (CTRL + SHIFT + i) at the same time to open up the inspector view on your browser.

Paste the following code into your console window and hit **Enter**
```
function ClickConnect(){
console.log("Working"); 
document
  .querySelector('#top-toolbar > colab-connect-button')
  .shadowRoot.querySelector('#connect')
  .click() 
}
setInterval(ClickConnect,60000)
```

- **Serial 11** :

Using Automatic Mixed Precision to change precision to cope up with bigger data. Use [3DptCloudimageofAlzheimer_networks_amp.ipynb](3DptCloudimageofAlzheimer_networks_amp.ipynb) for amp implementation. Code is organized to train in 2 stages where first stage is high lr, low weight_decay, augmentation disabled, with balance batch and 2nd stage is opposite of stage one. Here , ```IS_FIRST_STAGE = True```  and ```RESUME_TRAINING = False``` is for first stage where opposite for 2nd stage. Keep ```use_amp = True``` to use automatic mixed precision.

- **Serial 12** :

As this is hugely imbalanced data so we shifted our loss to 'Class Balance Loss' which can tackle imbalancy in dataset. To replace existing loss with this loss you can change the previous loss function like following:
**Step 1 :**
Add the Class balance loss function:
```
import torch.nn.functional as F
def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss
```

Then add necessary parameters for the loss function. For details you can see the <a href="https://arxiv.org/pdf/1901.05555.pdf">paper</a>. 
```
no_of_classes = 2
beta = 0.9999
gamma = 2.0
samples_per_cls = [35,15]   #class sample number in one batch
loss_type = "softmax"
```
Lastly don't forget to change loss function by adding :
```
outputs = model()   #make sure model returns logits 
loss = CB_loss(labels, outputs, samples_per_cls, no_of_classes,loss_type, beta, gamma)
```

