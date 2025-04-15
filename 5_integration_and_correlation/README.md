# Creating and managing the R Conda environment

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
## Straight script execution
I've also exported the scripts derived from the Jupyter notebook:
```sh
conda deactivate
conda activate r_env_corr

nohup Rscript creation_satac_selected_object_script.R > nohup_script_for_selected_obj_creation.out &
nohup Rscript creation_satac_selected_object_script_v2.R > nohup_script_for_selected_obj_creation_v2.out &
nohup Rscript combined_creation_satac_object_script.R > nohup_script_for_combined_obj_creation.out &
nohup Rscript combined_creation_satac_object_script_v2.R > nohup_script_for_combined_obj_creation_v2.out &
nohup Rscript temp_until_LSI.R > nohup_until_LSI.out &
nohup Rscript temp_for_linked_peaks.R > nohup_for_temp_linked_peaks.out &
nohup Rscript script_cross_modality_correct_object_creation.R > nohup_script_cross_modality_correct_object_creation.out &
```



---
# DCA - Peaks denoiser
**[Deep Counts Autoencoder](https://github.com/theislab/dca)**

ADD PACKAGE DESCRIPTION.

The installation and correct execution of this "package" (as it's not a proper and real package) took me two days, so, after checking in the "issue" section of the package GitHub page, I managed to install it and make it run but only on the CPU and with a single core. Because of this, the code takes quite some time.

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
  - for **subsetted** prenormalised peaks data:
```sh
nohup dca output/sATAC_preprocessing/data_signac_<SAMPLE_NAME>_selected_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/dated_dca_<SAMPLE_NAME>_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_<SAMPLE_NAME>_data.out &
```
    or, adapted to my case:
```sh
nohup dca output/sATAC_preprocessing/data_signac_220327_C1_selected_brca_q0_peak_bc_matrix2.csv output/sATAC_preprocessing/dated_dca_220327_C1_peaks_q0_brca --threads 6 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_220327_C1_data.out &
```


  - for **subsetted** counts data:
```sh
nohup dca output/sATAC_preprocessing/counts_signac_<SAMPLE_NAME>_selected_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/counted_dca_<SAMPLE_NAME>_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights > nohup_for_dca_<SAMPLE_NAME>_counts.out &
```
    or, adapted to my case:
```sh
nohup dca output/sATAC_preprocessing/counts_signac_220327_C1_selected_brca_q0_peak_bc_matrix2.csv output/sATAC_preprocessing/counted_dca_220327_C1_peaks_q0_brca2 --threads 6 --nosizefactors --saveweights > nohup_for_dca_220327_C1_counts.out &
```


  - for **subsetted** gene activity:
```sh
nohup dca output/sATAC_preprocessing/gene_activity_selected_<SAMPLE_NAME>_brca.csv output/sATAC_preprocessing/dca_gene_activity_selected_<SAMPLE_NAME>_brca --threads 6 --saveweights > nohup_for_<SAMPLE_NAME>_dca_gene_activity.out &
```
    or, adapted to my case:
```sh
nohup dca output/sATAC_preprocessing/gene_activity_selected_220327_C1_brca.csv output/sATAC_preprocessing/dca_gene_activity_selected_220327_C1_brca --threads 6 --saveweights > nohup_for_220327_C1_dca_gene_activity.out &
```


  - for **Visium** gene counts:
```sh
nohup dca output/Visium_preprocessing/gene_counts_<VISIUM_SAMPLE_NAME>.csv output/Visium_preprocessing/dca_gene_counts_<VISIUM_SAMPLE_NAME> --threads 6 --saveweights > nohup_for_<VISIUM_SAMPLE_NAME>_dca_gene_counts.out &
```
    or, adapted to my case:
```sh
nohup dca output/Visium_preprocessing/gene_counts_FFPE.csv output/Visium_preprocessing/dca_gene_counts_FFPE2 --threads 6 --saveweights > nohup_for_FFPE_dca_gene_counts.out &
```


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

---










\
\
As DCA is part of the unning the DCA denoiser in a ScanPy script. DOESN'T WORK
- On the raw counts:
```sh
python3 dca_denoiser.py output/sATAC_preprocessing/counts_signac_filtered_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/denoised_counts_output.h5ad --threads 4
```

- On the normalised counts:
```sh
python3 dca_denoiser.py output/sATAC_preprocessing/data_signac_filtered_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/denoised_data_output.h5ad --threads 4
```










---
## DCA with Docker attempt
It never worked out...
  - with Docker container:
```sh
docker run --rm -it
-v /disk2/user/gabgam/the_project/5_integration_and_correlation/output/sATAC_preprocessing/
quay.io/biocontainers/dca:0.3.4--pyhdfd78af_0
dca data_signac_filtered_brca_q0_peak_bc_matrix.csv docker_dated_dca_peaks_q0_brca
```
  or:
```sh
docker exec -it dca_container_034 dca /disk2/user/gabgam/the_project/5_integration_and_correlation/output/sATAC_preprocessing/data_signac_filtered_brca_q0_peak_bc_matrix.csv /disk2/user/gabgam/the_project/5_integration_and_correlation/output/sATAC_preprocessing/docker_dated_dca_peaks_q0_brca
```

Check if it works:
```sh 
docker exec -it dca_container_034 dca ~/the_project/5_integration_and_correlation/output/sATAC_preprocessing/data_signac_filtered_brca_q0_peak_bc_matrix.csv ~/the_project/5_integration_and_correlation/output/sATAC_preprocessing/docker_dated_dca_peaks_q0_brca
```
```sh 
docker exec -it dca_container_034 dca data/data_signac_filtered_brca_q0_peak_bc_matrix.csv data/docker_dated_dca_peaks_q0_brca
```





screen:

[I 2025-03-26 19:14:36.999 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2025-03-26 19:14:37.005 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2025-03-26 19:14:37.012 ServerApp] jupyterlab | extension was successfully linked.
[I 2025-03-26 19:14:37.016 ServerApp] notebook | extension was successfully linked.
[I 2025-03-26 19:14:40.919 ServerApp] notebook_shim | extension was successfully linked.
[I 2025-03-26 19:14:41.191 ServerApp] notebook_shim | extension was successfully loaded.
[I 2025-03-26 19:14:41.195 ServerApp] jupyter_lsp | extension was successfully loaded.
[I 2025-03-26 19:14:41.196 ServerApp] jupyter_server_terminals | extension was successfully loaded.
[I 2025-03-26 19:14:41.198 LabApp] JupyterLab extension loaded from /disk2/user/gabgam/miniconda3/envs/r_env_corr/lib/python3.13/site-packages/jupyterlab
[I 2025-03-26 19:14:41.198 LabApp] JupyterLab application directory is /disk2/user/gabgam/miniconda3/envs/r_env_corr/share/jupyter/lab
[I 2025-03-26 19:14:41.212 LabApp] Extension Manager is 'pypi'.
[I 2025-03-26 19:14:41.789 ServerApp] jupyterlab | extension was successfully loaded.
[I 2025-03-26 19:14:41.792 ServerApp] notebook | extension was successfully loaded.
[I 2025-03-26 19:14:41.793 ServerApp] Serving notebooks from local directory: /disk2/user/gabgam/the_project/5_integration_and_correlation
[I 2025-03-26 19:14:41.793 ServerApp] Jupyter Server 2.15.0 is running at:
[I 2025-03-26 19:14:41.793 ServerApp] http://cbb.medh.ki.se:8888/tree?token=f1a2abcc2ffe168b8bb272b95f19e63b891f0721f2376b43
[I 2025-03-26 19:14:41.793 ServerApp]     http://127.0.0.1:8888/tree?token=f1a2abcc2ffe168b8bb272b95f19e63b891f0721f2376b43
[I 2025-03-26 19:14:41.793 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-03-26 19:14:41.824 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///disk2/user/gabgam/.local/share/jupyter/runtime/jpserver-885623-open.html
    Or copy and paste one of these URLs:
        http://cbb.medh.ki.se:8888/tree?token=f1a2abcc2ffe168b8bb272b95f19e63b891f0721f2376b43
        http://127.0.0.1:8888/tree?token=f1a2abcc2ffe168b8bb272b95f19e63b891f0721f2376b43
[I 2025-03-26 19:14:42.223 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server