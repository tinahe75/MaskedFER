# Facial Expression Recognition for Masked Faces

This project was developed by adapting the code from [ResidualMaskingNetwork](https://github.com/phamquiluan/ResidualMaskingNetwork).
The training and validation data was generously provided by Bo Yang and Jianming Wu via [LFW-emotion-dataset](https://github.com/KDDI-AI-Center/LFW-emotion-dataset).

Facial expression recognition (FER) in the wild is challenging, hence there is ongoing work to develop models that are robust to various environmental conditions. One challenge for FER is the partial occlusion of faces. Given the COVID-19 pandemic, mask-wearing in public spaces has become the norm, which makes FER for masked faces a new and relevant problem. The KDDI’s research team has released a new dataset (M-LFW) containing synthetically masked faces. This dataset has only been evaluated on VGG19 and MobileNet by Yang et al. We evaluate this dataset with two other architectures, ResMaskingNet and Cbam\_Resnet50, in order to further understand how real and synthetic face masks affect FER. Both architectures outperform the models reported by Yang et al. However, there remains a significant performance gap between synthetic and real masked faces. 


## Required libraries
* pytorch == 1.7.0
* torchvision == 0.8.0
* torchtext == 0.8.0
* pytorchcv
* imgaug==0.3.0
* For visualization: matploblib, seaborn, opencv

## Colab notebooks
Running these notebooks requires downloading the dataset to your Google Drive. The notebooks will mount the drive 
and directly read/write to it. Checkpoints are saved to your Drive folder too. During training, 
the best training and validation accuracies are displayed in the notebook.
### Training on LFW
* https://colab.research.google.com/drive/1C3C4tekMFi4PYQacWMLUckOg3iTT45C_?usp=sharing
### Training on M-LFW
* https://colab.research.google.com/drive/1PQz4r6fW0rCTKFXCUz5qJ08T8eIcSmq1?usp=sharing

## Saved checkpoints
* [cbam_resnet50, trained on M-LFW](https://drive.google.com/file/d/1-28bhnVTG7H-U1iMWSujU2SAGKPeXi4C/view?usp=sharing): cbam_resnet50__n_2021Apr18_23.46
* [resmaskingnet, trained on M-LFW](https://drive.google.com/file/d/1T-_yyCiyxHdmTuX285ifSGLdMB2UkVXE/view?usp=sharing): resmasking_dropout1__n_2021Apr18_21.31
* [cbam_resnet50, trained on LFW](https://drive.google.com/file/d/1zWPr7TnvP_0XzL3z48XXtgPrT_VOfm7W/view?usp=sharing): cbam_resnet50__n_2021Apr20_00.24
* [resmaskingnet, trained on LFW](https://drive.google.com/file/d/1M1GSLfuuwLmn8EaU9VeX3PXpQbOdn76C/view?usp=sharing): resmasking_dropout1__n_2021Apr18_22.13

## Project tasks:
  * [x] add data loader code to work with LFW images
  * [x] train ResMasking on M-LFW, default hyperparams 
  * [x] ResMasking hyperparam tuning, try different transfer learning configurations
  * [x] train cbam_resnet50 on M-LFW, default hyperparams 
  * [x] cbam_resnet50 hyperparam tuning, try different transfer learning configurations
  * [x] train ResMasking on LFW, test on real world masked faces 
  * [x] train cbam_resnet50 on LFW , test on real world masked faces 
  * [x] collect + annotate real world masked faces 


## Evaluation

To generate confusion matrices and samples of misclassified images, run gen_confusion_matrix.py.
The --type parameter specificies the data type (train, val, test). This allows generating confusion matrices for 
both validation and test data.

If not saving sample images, do not use --save_samples. The following code assumes that the checkpoints 
are already saved to LFW-FER/saved/checkpoints and M-LFW-FER/saved/checkpoints. To download them, follow the links provided above.

Saving sample images requires running the command twice, once for cbam_resnet50 and once fo resmaskingnet.
Only images misclassified by both models are saved. 

Run the following via Colab to produce samples and confusion matrices 
for both architectures, trained on LFW and M-LFW.

*Note: this script contains some hardcoded checkpoint names when saving samples. Make sure to modify if needed.*


```Shell

python gen_confusion_matrix.py --config configs/m_lfw_cbam_resnet50_config_colab.json --type test --model cbam_resnet50 --checkpoint cbam_resnet50__n_2021Apr18_23.46 --save_samples 0 

python gen_confusion_matrix.py --config configs/m_lfw_res_masking_config_colab.json --type test --model resmasking_dropout1 --checkpoint resmasking_dropout1__n_2021Apr18_21.31 --save_samples 1


python gen_confusion_matrix.py --config configs/lfw_cbam_resnet50_config_colab.json --type test --model cbam_resnet50 --checkpoint cbam_resnet50__n_2021Apr20_00.24 --save_samples 0 

python gen_confusion_matrix.py --config configs/lfw_res_masking_config_colab.json --type test --model resmasking_dropout1 --checkpoint resmasking_dropout1__n_2021Apr18_22.13 --save_samples 1

```





## References


L. Pham, H. Vu, T. A. Tran, "Facial Expression Recognition Using Residual Masking Network", IEEE 25th International Conference on Pattern Recognition, 2020, 4513-4519. Milan -Italia.


```
@inproceedings{luanresmaskingnet2020,
  title={Facial Expression Recognition using Residual Masking Network},
  author={Luan, Pham and Huynh, Vu and Tuan Anh, Tran},
  booktitle={IEEE 25th International Conference on Pattern Recognition},
  pages={4513--4519},
  year={2020}
}
```

LFW-FER and M-LFW-FER dataset provided by Yang et al.:

```BibTeX
@inproceedings{LFW-emotion,
  author = {Yang, Bo and Wu, Jianming and Hattori, Gen},
  title = {Facial Expression Recognition with the advent of human beings all behind face masks},
  year = {2020},
  publisher = {Association for Computing Machinery},
  address = {Essen, Germany},
  series = {MUM2020}
}
```
Their paper on M-LFW: https://dl.acm.org/doi/10.1145/3428361.3432075

Original LFW database:


```BibTeX
@TechReport{LFWTech,
  author = {Gary B. Huang and Manu Ramesh and Tamara Berg and Erik Learned-Miller},
  title = {Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments},
  institution = {University of Massachusetts, Amherst},
  year = {2007},
  number = {07-49},
  month = {October}
}
```
