## Creating an R environment with Conda.

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

Let's install the packages and dependencies. Most of them works with BioConda.

```sh
conda install -c conda-forge jupyterlab -y
conda install -c r r-irkernel -y

conda install bioconda::bioconductor-ensdb.hsapiens.v86
conda install bioconda::r-signac
conda install bioconda::r-seurat
conda install bioconda::r-harmony
conda install bioconda::r-loomr
# conda install bioconda::r-monocle3
# conda install bioconda::r-archr

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

The R script is not able to create the directories, differently to Python, so I had to manually create the output folder.


