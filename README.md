# Multi-task Learning Approach for Intracranial Hemorrhage Prognosis

This repository contains the code of the paper **Multi-task Learning Approach for Intracranial Hemorrhage Prognosis**, accepted at Machine Learning for Medical Imaging workshop @ MICCAI 2024 (MLMI). For our submitted manuscript with added funding acknowledgements and authors' names and affiliations, but without post submission improvements or corrections, please click [here] (https://arxiv.org/abs/2408.08784). The final version is not published yet.

In this study, we aim to enhance image-based prognosis by learning a robust feature representation shared between pronosis and the clinical and demographic variables most highly correlated with it. Our approach mimics clinical decision-making by reinforcing the model to learn valuable prognostic data embedded in the image. We propose a 3D multi-task image model to predict prognosis, Glasgow Coma Scale and age, improving accuracy and interpretability, as shown below.
![Proposed multi-task image model integrating GCS and age as outputs to regularize the learning and enhance the prognosis task. In the saliency maps, brighter colors mean higher importance.](Figures/Figure_Method.png)
Our method outperforms current state-of-the-art baseline image models, and demonstrates superior performance in ICH prognosis compared to four board-certified neuroradiologists using only CT scans as input.
![Saliency maps from the paper](Figures/Figure3_SM_guidedBackProp.png)
Contact: Miriam Cobo (cobocano@ifca.unican.es)

### Requirements

All experiments were carried out using the following:

```
imbalanced-learn==0.11.0
itk==5.3.0
matplotlib==3.6.2
matplotlib-inline==0.1.6
numpy==1.24.3
opencv-python==4.7.0.72
openpyxl==3.1.2
pandas==1.5.2
Pillow==9.4.0
pyplastimatch==0.1
scikit-learn==1.2.0
scikit-learn-extra==0.3.0
tensorboard==2.13.0
tensorboard-data-server==0.7.1
tensorboard-plugin-wit==1.8.1
tensorflow==2.13.0
tensorflow-estimator==2.13.0
tensorflow-io==0.33.0
tensorflow-io-gcs-filesystem==0.33.0
tensorrt==0.0.1
torch==1.13.1
torchsummary==1.5.1
torchvision==0.14.1
tornado==6.2
tqdm==4.42.1
```
### Dataset

All experiments were conducted using the open-source Head-CT 2D/3D images with and without
ICH prepared for Deep Learning dataset. You can ask for access to the dataset at this [link](https://digital.csic.es/handle/10261/275792).

### Training

To launch the training of one of the available multitask image models run the following:

```
python /PATH_TO_REPO/trainImageModels/train_"name_model".py 
```

To launch the training of one of the tabular models run the following:
```
python /PATH_TO_REPO/trainTabularModels/train_"name_model".py 
```

### Citation

If you use this code in your research, please cite our paper:
```
Final version not published yet.
```


### Aknowledgments

* Thanks to [Perez et al.](https://digital.csic.es/handle/10261/275792) for the original dataset repository. 
