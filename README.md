[![DOI](https://sandbox.zenodo.org/badge/812648004.svg)](https://handle.stage.datacite.org/10.5072/zenodo.155132)
## About
This is the source code of "Automated Recognition of Right Ventricular Dilatation and Prognosis Correlation", implemented with Keras and Tensorflow.
## Reproduction and model usage
1. All pretrained classification models were stored under the "model" folder.
2. run.ipynb: notebook of how you load and use the models or train your own model.
3. For model architecture and training, please refer to the "src" folder.
## PAH prognosis and survival analysis
Application was shown in survival_analysis.ipynb.
## Grad-CAM
![160019P48701_31_GradCAM](https://github.com/Urania880519/RV-dilatation-DL/assets/95178070/bda6e07d-e536-4dba-96af-eec731fb12ad)
## Segmentation 
The segmentation results came from our pretrained model. Adding both segmentation channels of left and right ventricle, the input arrays become larger.
We trained the A4C-seg model with these reproduced data, which is visualized and shown as the following gif.
![seg](https://github.com/Urania880519/RV-dilatation-DL/assets/95178070/3b737541-84e2-4c13-9082-a996e61ecba1)
