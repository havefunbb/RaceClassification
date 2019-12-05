# RaceClassification
Race Classification for 4 races (African,Asian,Caucasian,South Asian (Indian)) on pytorch

## System Requirements

* Pytorch
* Scikit-learn
* Plotly

## Dataset Preparation

For race classification we use RFW Train set, we subsample the dataset, yielding about 40.000 images for training. Train set is race balance but not gender balance.

![Sampled Dataset Statistics](source/all_data.png)


## Model Train and Test


```cmd

python test_sota.py --model resnet50 --ckpt models/model.ckpt
python test_sota.py --model resnet101 --ckpt models/model.ckpt
python test_sota.py --model inception --ckpt models/model.ckpt

```
## Results

| Model Name | African | Asian | Caucasian | Indian | Average |
|------------|---------|-------|-----------|--------|---------|
| Resnet50   | 96,09   | 94,74 | 88,77     | 91,21  | 92,7025 |
| Resnet101  | 97,59   | 93,38 | 88,75     | 90,69  | 92,6025 |
| Inception  | 96,06   | 88,42 | 87,07     | 86,06  | 89,4025 |
| WSDAN      | 97,77   | 93,84 | 94,48     | 88,82  | 93,7275 |

### WSDAN Confusion Matrix

![Classifiers Confusion Matrices](source/classifiers_cf.png)