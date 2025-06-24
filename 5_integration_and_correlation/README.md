This folder represents the last step of the project where image-derived clusters' identity from both the modalities is checked.

In the notebooks, you'll see how the data has been processed, the quality checks, the spot annotation via CCA and scRNA-seq reference and the evaluation of image-based clusters identity.

---


Content of the folder:

`5_integration_and_correlation`<br>
`├── sATAC_analysis_and_integration.ipynb` &rarr; Spatial ATAC processing, QC, annotation and integration with image-based clusters<br>
`├── Visium_analysis_and_integration.ipynb` &rarr; Visium processing, QC, annotation and integration with image-based clusters<br>
respective notebook<br>
`├── figures/` &rarr; figures derived from the scripts or Jupyter Notebooks organised per sample <br>
`├── utils/` &rarr; contains the used environments `.yaml` files and eventual useful scripts <br>
`└── output/` &rarr; contains the normalised tiles, organised per sample, WSI (original or normalised) and size, and results of metrics evaluation<br>



As always, to create the working environments, use the `.yaml` files inside the `utils` folder. Otherwise, below there are instructions to create an R Conda envinoment for Jupyter Notebooks.




# Creating and managing the R Conda environment for this step


Name of the environment: `r_env_corr`.
```sh
conda create -n r_env_corr r-essentials r-base r-devtools -y
conda activate r_env_corr
# install Jupyter
conda install -c conda-forge jupyterlab -y
# install the R kernel
conda install -c r r-irkernel -y
# Check if the R kernel is installed
R -e "IRkernel::installspec(user = FALSE)"
# cheking for kernels. Should print something like `ir` 
jupyter kernelspec list
```

Let's install the packages and dependencies. Most of them work with BioConda.

```sh
conda install bioconda::bioconductor-ensdb.hsapiens.v86
conda install bioconda::r-signac
conda install bioconda::r-seurat
conda install bioconda::r-harmony
conda install bioconda::r-loomr
conda install bioconda::r-monocle3
conda install bioconda::r-archr
conda install conda-forge::r-gprofiler2
conda install bu_cnio::r-seuratwrappers 
conda install -n r_env_corr -c bioconda -c conda-forge     bioconductor-genomeinfodb     bioconductor-ensembldb     bioconductor-ensdb.hsapiens.v86     bioconductor-bsgenome.hsapiens.ucsc.hg38     bioconductor-dropletutils     bioconductor-genomicranges     bioconductor-genomicfeatures     bioconductor-annotationdbi     bioconductor-rtracklayer     bioconductor-rsamtools     bioconductor-biostrings     bioconductor-xvector     bioconductor-rhdf5     bioconductor-singlecellexperiment     bioconductor-delayedarray     bioconductor-delayedmatrixstats     bioconductor-hdf5array     bioconductor-beachmat     bioconductor-scuttle     r-rcurl     r-xml     r-matrix     r-patchwork     r-scales     r-viridis     r-purrr     r-ggplot2     r-dplyr     r-ica     r-spdep     r-jpeg     r-ggpubr     r-gplots     r-devtools
```

In R:

```r
BiocManager::install("EnsDb.Hsapiens.v86")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
remotes::install_github('satijalab/seurat-wrappers')
devtools::install_github("jbergenstrahle/STUtility")
devtools::install_github('cole-trapnell-lab/monocle3')
devtools::install_github("GreenleafLab/ArchR", ref="master", repos = BiocManager::repositories())
```

---
### DCA - Counts denoiser
**[Deep Counts Autoencoder (DCA)](https://github.com/theislab/dca)**

**Package description** from GitHub folder:
*A deep count autoencoder network to denoise scRNA-seq data and remove the dropout effect by taking the count structure, overdispersed nature and sparsity of the data into account using a deep autoencoder with zero-inflated negative binomial (ZINB) loss function.*

**Installation process:**

The installation and correct execution of this "package" (as it's not a proper and real package) took me 5 days, so, after checking in the "issue" section of the package GitHub page, I managed to install it and make it run but only on the CPU and with a single core. Because of this, the code takes quite some time.

In the `utils` folder you'll find a `.yml` file to create the proper `mamba` environment to run DCA, as the packages stated in the package `setup.py` file aren't compatible.\
The environment needs:
- `keras==2.4.0`
- `tensorflow==2.5.3`
- `xlrd==1.2.0`

In case, upon creationg of your `mamba_env` environment, run:
```sh
cd ~/the_project/5_integration_and_correlation

conda deactivate
conda activate mamba_env
# create the Mamba environment for DCA
mamba create -n dca_mamba_37 python==3.7

conda activate dca_mamba_37 

pip install keras==2.4.0 tensorflow==2.5.3
pip install xlrd==1.2.0
# move to your working folder
```

On terminal, in a Conda environment with TensorFlow and the DCA package installed, such as `dca_mamba_37` run:

<ul>

  - for **subsetted** prenormalised peaks data:
```sh
nohup dca output/sATAC_preprocessing/data_signac_<SAMPLE_NAME>_selected_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/dated_dca_<SAMPLE_NAME>_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_<SAMPLE_NAME>_data.out &
```
or, adapted to my case:
```sh
nohup dca output/sATAC_preprocessing/data_signac_220327_C1_selected_brca_q0_peak_bc_matrix2.csv output/sATAC_preprocessing/dated_dca_220327_C1_peaks_q0_brca --threads 6 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_220327_C1_data.out &
```
<br>

  - for **subsetted** counts data:
```sh
nohup dca output/sATAC_preprocessing/counts_signac_<SAMPLE_NAME>_selected_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/counted_dca_<SAMPLE_NAME>_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights > nohup_for_dca_<SAMPLE_NAME>_counts.out &
```
or, adapted to my case:
```sh
nohup dca output/sATAC_preprocessing/counts_signac_220327_C1_selected_brca_q0_peak_bc_matrix2.csv output/sATAC_preprocessing/counted_dca_220327_C1_peaks_q0_brca2 --threads 6 --nosizefactors --saveweights > nohup_for_dca_220327_C1_counts.out &
```
<br>

  - for **subsetted** gene activity:
```sh
nohup dca output/sATAC_preprocessing/gene_activity_selected_<SAMPLE_NAME>_brca_overlapping.csv output/sATAC_preprocessing/dca_gene_activity_selected_<SAMPLE_NAME>_brca_overlapping --threads 6 --saveweights > nohup_for_<SAMPLE_NAME>_dca_gene_activity.out &
```
or, adapted to my case:
```sh
nohup dca output/sATAC_preprocessing/gene_activity_selected_220327_C1_brca_overlapping.csv output/sATAC_preprocessing/dca_gene_activity_selected_220327_C1_brca_overlapping --threads 6 --saveweights > nohup_for_220327_C1_dca_gene_activity.out &
```
<br>

  - for **Visium** gene counts:
```sh
nohup dca output/Visium_preprocessing/<VISIUM_SAMPLE_NAME>/gene_counts_<VISIUM_SAMPLE_NAME>.csv output/Visium_preprocessing/<VISIUM_SAMPLE_NAME>/dca_gene_counts_<VISIUM_SAMPLE_NAME> --threads 6 --saveweights > nohup_for_<VISIUM_SAMPLE_NAME>_dca_gene_counts.out &
```
or, adapted to my case:

```sh
nohup dca output/Visium_preprocessing/visium_FFPE_dcis_idc_10X_img_not_changed_allspots/gene_counts.csv output/Visium_preprocessing/visium_FFPE_dcis_idc_10X_img_not_changed_allspots/dca_gene_counts --threads 6 --saveweights > nohup_for_visium_FFPE_dcis_idc_10X_img_not_changed_allspots_dca_gene_counts.out &
```
<br>

  - for **combined** prenormalised peaks data (this takes A LOT of time):
```sh
nohup dca output/sATAC_preprocessing/data_combined_brca_q0_peak_bc_matrix.csv \
  output/sATAC_preprocessing/dated_combined_dca_peaks_q0_brca2 \
  --threads 10 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts \
  > nohup_for_dca_data_combined2.out &
```
  - for **combined** gene activity:
```sh
nohup dca output/sATAC_preprocessing/gene_activity_combined_brca.csv \
  output/sATAC_preprocessing/dca_gene_activity_combined_brca2 \
  --threads 3 --saveweights > nohup_for_dca_gene_activity_combined.out &
```
</ul>
<br>
<br>

---

Other attempts with Docker containers and ScanPy implementation were tried but did not succeedeed.