This is the list of submitted file related to 3D Cloud point classification pipeline.

## Description of Modification and Related Codes

Following table summarizes all the change in pipeline

| Serial | Model Description | Optimizer | Loss Function | Data Dimension | Dataset Relatd New Features | Other new features |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline Model | SGD(lr = 1e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 2 | Added two more dense layers | Adam(lr = 5e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 3 | '' | Adam(lr = 5e-3,w_d = 1e-4) | CrossEntropy | 32 X 64 X 64 | -- | -- |



- Serial 1  (Baseline Pipeline) : [3DptCloudofAlzheimer_Baseline.ipynb](3DptCloudofAlzheimer_Baseline.ipynb) contains the baseline pipeline code
- Serial 2 : [3DptCloudofAlzheimer_modified.ipynb](3DptCloudofAlzheimer_modified.ipynb) contains that modified code
- Serial 3 : To add weight decay (adding L2 regularization to reduce overfitting) change code in **Model Hyperparameters** section like 
```bash
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
```

- Serial 4 :

- Baseline Model (Baseline pipeline) : 



