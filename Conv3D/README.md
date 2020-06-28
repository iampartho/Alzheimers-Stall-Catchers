This is the list of submitted file related to 3D Cloud point classification pipeline.

## Description of Modification and Related Codes

Following table summarizes all the change in pipeline

| Serial | Model Description | Optimizer | Loss Function | Data Dimension | Dataset Relatd New Features | Other new features |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline Model | SGD(lr = 1e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |
| 2 | Added two more dense layers | Adam(lr = 5e-3) | CrossEntropy | 32 X 64 X 64 | -- | -- |


- Baseline Model (Baseline pipeline) : 


