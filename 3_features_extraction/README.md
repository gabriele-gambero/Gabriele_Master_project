This folder contains the methodology for feature extraction from tiles using two models:

- KimiaNet &rarr; a Convolutional Neural Network ([paper](https://www.nature.com/articles/s41591-024-02857-3))<br>
- UNI2 &rarr; a general purpose foundation model ([paper](https://www.sciencedirect.com/science/article/pii/S1361841521000785))<br>

Content of the folder: <br>
`3_features_extraction`<br>
`├── models/` &rarr; folder with feature extraction models (KimiaNet and UNI2-h) and scripts <br>
`├── figures/` &rarr; figures derived from the scripts or Jupyter Notebooks organised per sample <br>
`├── utils/` &rarr; contains the used environments `.yaml` files and eventual useful scripts <br>
`└── output/` &rarr; contains the normalised tiles, organised per sample, WSI (original or normalised) and size, and results of metrics evaluation<br>


