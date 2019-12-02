# RaceClassification
Race Classification for 4 races (African,Asian,Caucasian,South Asian (Indian))

## System Requirements

* Pytorch
* Scikit-learn
* Plotly

## Dataset Preparation

For race classification we use RFW Train set, we subsample the dataset and obtain about 40.000 images. Train set is race balance but not gender balance.

<object data="https://github.com/seymayucer/RaceClassification/blob/master/source/all_data.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/seymayucer/RaceClassification/blob/master/source/all_data.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="source/all_data.pdf">Download PDF</a>.</p>
    </embed>
</object>


## Model Overview

![Model Architecture](source/model.png)

## Results

| Model    | Loss              | Avarage | African Female | African Male | Cau Female | Cau Male|std| 
|----------|-------------------|---------|----------|----------|---------|--------|---------|
| Resnet18 | Cross Entropy     | 0.621   | 0.606    | 0.575    | 0.628   | 0.670  |0.0399|
| Resnet18 | Cross Entropy (pretrained)    | 0.561   | 0.547   | 0.523    |0.552   | 0.616  |0.0397|
| Resnet18 | Dynamic C.Entropy | 0.675   | 0.671    | 0.644    | 0.672   | 0.710  |0.0271|
| Resnet18 | Dynamic C.Entropy (pretrained) | 0.593   | 0.557    | 0.642    | 0.595   | 0.673  |0.051
| Resnet18 | Dynamic Arcface Loss     | -      | -       | -       | -      | -     |-|
