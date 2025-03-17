# Human Breast Cancer: Visium Fresh Frozen, Whole Transcriptome 

Link for the dataset:
https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard

Folder structure:
```
data/visium_2022_FF_WG
├── input_files
└── output_n_suppl_files
    └── spatial
```


Download:

- Output and supplemental files:
    ```
    # Output Files
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_molecule_info.h5
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.tar.gz
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_raw_feature_bc_matrix.h5
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_raw_feature_bc_matrix.tar.gz
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_analysis.tar.gz
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_spatial.tar.gz
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_spatial_enrichment.csv
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_metrics_summary.csv
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_web_summary.html
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_cloupe.cloupe
    ```

- Input files:
    ```
    # Input Files
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_image.tif
    wget https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_alignment_file.json
    ```

Actually, I didn't need to use files related to the genome, so, I haven't even downloaded them.