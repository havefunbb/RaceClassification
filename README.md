# RaceClassification
Race Classification for 4 races (African,Asian,Caucasian,South Asian (Indian)) on pytorch

## System Requirements

* Pytorch
* Scikit-learn
* Plotly

## Dataset Preparation

For race classification we use RFW Train set, we subsample the dataset and obtain about 40.000 images. Train set is race balance but not gender balance.

![Sampled Dataset Statistics](source/all_data.png)


## Model Overview

![Model Architecture](source/model.png)

## Results

| Model    |  Avarage | African | Asian | Caucasian  | Indian|
|----------|---------|----------|----------|---------|--------|
| Resnet50 | -  | -    | -    | -   | -  |
| Resnet101|-   | -   | -    |-   | -  |
| Inceptionvs3 | -   | -    | -    | -   | -  |
| WS-Dan | -   | -    | -    | -  | -  |

### WSDAN Confusion Matrix