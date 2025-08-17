## Feature Extraction

This folder contains the methodology to perform feature extraction from the previously created and color-normalised tiles. For this end, we have applied two histopathology pre-trained deep learning models without fine-tuning steps:

- **KimiaNet** &rarr; a Convolutional Neural Network ([paper](https://www.nature.com/articles/s41591-024-02857-3))<br>
- **UNI2** &rarr; a general purpose Foundation Model ([paper](https://www.sciencedirect.com/science/article/pii/S1361841521000785))<br>

Content of the folder: <br>
`3_features_extraction`<br>
`├── models/` &rarr; folder with feature extraction models (KimiaNet and UNI2-h) and scripts <br>
`├── figures/` &rarr; figures derived from the scripts or Jupyter Notebooks organised per sample <br>
`├── utils/` &rarr; contains the used environments `.yaml` files and eventual useful scripts <br>
`└── output/` &rarr; contains the normalised tiles, organised per sample, WSI (original or normalised) and size, and results of metrics evaluation<br>


