# %% [markdown]
# Environment name: `r_env_corr`

# %%
R.version.string


# %%
setwd(dir = "/disk2/user/gabgam/the_project/5_integration_and_correlation/")

# %%
source("./utils/source_satac.R")
library(biovizBase)

# %%
# table <- list()

# PATH_TO_sATAC_DATA_FOLDER <- "../data/spatial_atac/"
# sATAC_SAMPLE_NAME <- "220327_C1"

# # create a table containing paths to raw and meta data (i.e., output from cellranger + spatial info)
# table <- list(
#         singlecell = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", sATAC_SAMPLE_NAME, "/singlecell.csv"),
#         fragments = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", sATAC_SAMPLE_NAME, "/fragments.tsv.gz"),
#         tissue_paths = paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/", sATAC_SAMPLE_NAME, "_tissue.tsv"),
#         spotfiles = paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/", sATAC_SAMPLE_NAME, "_tissue.csv")
#         )

# # as.data.frame(table)
# table

# %%
# set paths to raw data and metadata
dirs <- list.dirs("../data/spatial_atac/outs/", recursive = F, full.names = F)
table <- list()

PATH_TO_sATAC_DATA_FOLDER <- "../data/spatial_atac/"
sATAC_SAMPLE_NAME <- "220327_C1"
# dirs <- list("220327_C1")

# create a table containing paths to raw and meta data (i.e., output from cellranger + spatial info)
for(i in dirs){
  table[[i]] <- list(samples = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", i, "/raw_peak_bc_matrix.h5"),
                  singlecell = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", i, "/singlecell.csv"),
                  fragments = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", i, "/fragments.tsv.gz"),
                  tissue_paths = paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/", i, "_tissue.tsv"),
                  spotfiles = paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/", i, "_tissue.csv"))
}

infoTable <- do.call("rbind", table) %>% as.data.frame()
#infoTable <- as.data.frame(table, stringsAsFactors = FALSE)

object <- list(md = list(),
               frag = list(),
               counts = list())

# build new matrices with new peak set
# peaks obtained from https://www.science.org/doi/10.1126/science.aav1898
# load peaks and create genomicranges object
BRCA_PEAK_CALLS <- paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/BRCA_peakCalls.csv")

gr <- read.csv(
  file = BRCA_PEAK_CALLS,
  col.names = c("chr", "start", "end"),
  sep = ";"
) %>% makeGRangesFromDataFrame()

# create fragment objects and count matrices for each section separately
for(i in seq_along(dirs)){
  folder = paste0(PATH_TO_sATAC_DATA_FOLDER, "combined/")
  # load metadata
  object$md[[i]] <- read.table(
    file = infoTable$singlecell[[i]],
    stringsAsFactors = FALSE,
    sep = ",",
    header = TRUE,
    row.names = 1
  )[-1, ] # remove the first row (NO_BARCODE)
  
  # create fragment objects
  object$frag[[i]] <- CreateFragmentObject(
    path = infoTable$fragments[[i]],
    cells = rownames(object$md[[i]])
  )
  
  if (file.exists(paste0(folder, "brca_peak_bc_matrix.h5"))) {
    message("File exists.")
    object$counts[[i]] <- Read10X_h5(paste0(folder, "brca_peak_bc_matrix.h5"))
    
  } else {
    
    # make count matrix
    object$counts[[i]] <- FeatureMatrix(
      fragments = object$frag[[i]],
      features = gr,
      cells = rownames(object$md[[i]])
    )
    
    write10xCounts(path = paste0(folder, "brca_peak_bc_matrix.h5"),
                   x = object$counts[[i]],
                   type = "HDF5") # save new count matrix
  }
  
}


# %%
# save unmerged objects and metadata
saveRDS(object, "output/sATAC_preprocessing/1_combined_brca_unmerged.rds")

# %% [markdown]
# building the correct tissue_list file.

# %%
# make tissue position files
for(i in seq_along(infoTable$tissue_paths)){
  spotfile <- read.table(infoTable$tissue_paths[[i]], sep = "\t", header = T)
  
  spotfile_tissue <- cbind(
    spotfile$row, #array_row
    spotfile$col, #array_col
    spotfile$x, #pxl_row_in_fullres
    spotfile$y #pxl_col_in_fullres
  ) %>% as.data.frame()
  tissue <- rep(1, nrow(spotfile_tissue))
  spotfile_tissue <- cbind(tissue, spotfile_tissue)
  rownames(spotfile_tissue) <- paste0(spotfile$barcode, "-1")
  
  # save new tissue position file with the correct columns for creating STutility object as .csv files
  write.csv(spotfile_tissue, 
            paste0(strsplit(infoTable$tissue_paths[[i]], "tsv"), "csv"), 
            sep = ",")
  
}

# %% [markdown]
# loading the files just created and 

# %%
# load spotfiles
tissue_md <- list()
object$mtx <- list()

for(i in seq_along(infoTable$spotfiles)){
  spotfile <- read.csv(infoTable$spotfiles[[i]],
                       col.names = c("barcode", "tissue", "y", "x", "pixel_y", "pixel_x")
  )
  
  # filter count matrix to retain only spots overlaying tissue
  mtx <- object$counts[[i]] # barcodes extracted from here
  mtx <- mtx[,colnames(mtx) %in% spotfile$barcode] 
  
  object$mtx[[i]] <- mtx
  
  #save spot info in a list
  tissue_md[[i]] <- spotfile
}


# %% [markdown]
# why do they perform normalisation here???

# %%
signac_object <- list()

# create signac objects for each section and run normalization and dimensionality reduction
for(i in seq_along(infoTable$spotfiles)){
  assay <- CreateChromatinAssay(object$mtx[[i]], 
                                fragments = object$frag[[i]])
  
  signac_object[[i]] <- CreateSeuratObject(assay, 
                                           assay = "peaks", 
                                           meta.data=object$md[[i]])
  signac_object[[i]]$section <- rownames(infoTable)[i]
  # signac_object[[i]]$sample <- i
  signac_object[[i]]$sample <- rownames(infoTable)[i]
  # compute LSI
  signac_object[[i]] <- FindTopFeatures(signac_object[[i]], min.cutoff = 10)
  signac_object[[i]] <- RunTFIDF(signac_object[[i]])
  signac_object[[i]] <- RunSVD(signac_object[[i]])
}

# %% [markdown]
# normalised with LSI.
# 
# The normalisation happens on the `data` object.

# %%
# merge objects
combined <- merge(
  x = signac_object[[1]],
  y = c(signac_object[2:length(dirs)])
)

table(combined$section)

# add spatial data to meta
for(i in seq_along(infoTable$tissue_paths)){
  tissue_md[[i]]$barcode <- paste0(tissue_md[[i]]$barcode, "_", i)
  rownames(tissue_md[[i]]) <- tissue_md[[i]]$barcode 
}

tissue_md_combined <- do.call("rbind", tissue_md) 
combined <- AddMetaData(combined, tissue_md_combined)


# %% [markdown]
# Don't add the image as mine has already been processed.
# %%
table(combined$sample)

# %%
dim(combined)

# %% [markdown]
# I don't think that I should normalise once again.

# %%
#preprocess raw data
combined <- combined %>% 
  RunTFIDF() %>%
  FindTopFeatures(min.cutoff = 'q0') %>%
  RunSVD() %>%
  RunUMAP(reduction = 'lsi', dims = 2:7) %>%
  FindNeighbors(reduction = 'lsi', dims = 2:7) %>%
  FindClusters(algorithm = 3, resolution = 0.5)


# %% [markdown]
# in the original code they use `data`, but that's the normalised matrix.

# %%
# #save matrix for denoising
write.csv(combined@assays$peaks@data,
          "./output/sATAC_preprocessing/data_combined_brca_q0_peak_bc_matrix.csv")

# # %% [markdown]
# # # DCA - Peaks denoiser
# # 
# # On terminal, in a Python environment with TensorFlow and the DCA package installed, such as `dca_mamba_37` run:\
# #     - for counts
# # ```sh
# # nohup dca output/sATAC_preprocessing/counts_combined_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/counted_dca_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights > nohup_for_dca_counts.out &
# # ```
# #   - for prenormalised data:
# # ```sh
# # nohup dca output/sATAC_preprocessing/data_combined_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/dated_dca_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_data.out &
# # ```
# # 
# #   - with Docker container:
# # ```sh
# # docker run --rm -it
# # -v /disk2/user/gabgam/the_project/5_integration_and_correlation/output/sATAC_preprocessing/
# # quay.io/biocontainers/dca:0.3.4--pyhdfd78af_0
# # dca data_combined_brca_q0_peak_bc_matrix.csv docker_dated_dca_peaks_q0_brca
# # ```
# #     or:
# # ```sh
# # docker exec -it dca_container_034 dca /disk2/user/gabgam/the_project/5_integration_and_correlation/output/sATAC_preprocessing/data_combined_brca_q0_peak_bc_matrix.csv /disk2/user/gabgam/the_project/5_integration_and_correlation/output/sATAC_preprocessing/docker_dated_dca_peaks_q0_brca
# # ```
# # 
# # Check if it works:
# # ```sh 
# # docker exec -it dca_container_034 dca ~/the_project/5_integration_and_correlation/output/sATAC_preprocessing/data_combined_brca_q0_peak_bc_matrix.csv ~/the_project/5_integration_and_correlation/output/sATAC_preprocessing/docker_dated_dca_peaks_q0_brca
# # ```
# # ```sh 
# # docker exec -it dca_container_034 dca data/data_combined_brca_q0_peak_bc_matrix.csv data/docker_dated_dca_peaks_q0_brca
# # ```
# # 
# # ~/the_project/5_integration_and_correlation/output/sATAC_preprocessing
# # 
# # Successfully installed keras-2.4.0 tensorflow-2.5.3
# # 
# # pip install keras==2.4.0 tensorflow==2.5.3
# # pip install xlrd==1.2.0
# # 

# # %% [markdown]
# # ---

# %%
# load denoised matrices and save objects - separately for peaks and gene activity due to large size
# peaks
denoised_counts <- read.table("./output/sATAC_preprocessing/dated_combined_dca_peaks_q0_brca/mean.tsv", row.names = 1) %>% 
  as.matrix()
colnames(denoised_counts) <- sub("\\.", "-", colnames(denoised_counts))


# %%
# combined@assays$peaks

# %%
# combined@assays$peaks

# %%
# combined@assays$dca

# %%
# add denoised matrix to object
# filter to retain same spots and features
combined[["dca"]] <- combined[["peaks"]]

# %%
dim(combined[["dca"]]@data)
dim(combined[["peaks"]]@data)

# %%
combined@assays$dca@data <- denoised_counts
combined@assays$dca@counts <- combined@assays$dca@counts[rownames(combined@assays$dca@counts) %in%
                                                           rownames(combined@assays$dca@data),]
combined@assays$dca@var.features <- combined@assays$dca@var.features[combined@assays$dca@var.features %in%
                                                                       rownames(combined@assays$dca@data)]
combined@assays$dca@meta.features <- combined@assays$dca@meta.features[rownames(combined@assays$dca@meta.features) %in% 
                                                                        rownames(combined@assays$dca@data), , drop = FALSE]

# Keep only fragment objects with cells
combined@assays$peaks@fragments <- combined@assays$dca@fragments[
  sapply(combined@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

combined@assays$dca@fragments <- combined@assays$dca@fragments[
  sapply(combined@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

rm(denoised_counts)

# %%
## compare DCA clustering
# clustering between DCA and peaks
# no need for harmony integration as the sections come from the same sample
DefaultAssay(combined) <- "dca"
combined <- RunTFIDF(combined, verbose = TRUE) %>%
  FindTopFeatures(min.cutoff = 'q0') %>%
  RunSVD() %>%
  RunUMAP(reduction = "lsi", dims = 2:9) %>%
  FindNeighbors(reduction = "lsi", dims = 2:9) %>%
  FindClusters(resolution = 0.5)

message("FINISHED DCA CLUSTERING!")

saveRDS(combined, "output/sATAC_preprocessing/2_combined_n_clustered_brca_denoised_peaks.rds")

combined$dca_snn_res.0.5 <- combined$seurat_clusters

levels(combined$peaks_snn_res.0.5) <- c("2", "4", "1", "3", "0")
levels(combined$dca_snn_res.0.5) <- c("2", "3", "0", "4", "1")

# %% [markdown]
#   - `FeatureMatrix` → Construct a feature x cell matrix from a genomic fragments file.
#   - `write10xCounts` → `rdrr` package function → Create a directory containing the count matrix and cell/gene annotation from a sparse matrix of UMI counts, in the format produced by the CellRanger  software suite. 
#   - `GeneActivity` → Compute counts per cell in gene body and promoter region.
#   - `seqlevelsStyle` → The `seqlevelsStyle` getter and setter can be used to get the current seqlevels style of an object and to rename its seqlevels according to a given style. 
#   - `FindAllMarkers` → `Seurat` package function → Finds markers (differentially expressed genes) for each of the identity classes in a dataset (peaks in our case).
#   - `RegionStats` → Compute the GC content, region lengths, and dinucleotide base frequencies for regions in the assay and add to the feature metadata.
#   - `LinkPeaks` → Find peaks that are correlated with the expression of nearby genes. For each gene, this function computes the correlation coefficient between the gene expression and accessibility of each peak within a given distance from the gene TSS, and computes an expected correlation coefficient for each peak given the GC content, accessibility, and length of the peak. The expected coefficient values for the peak are then used to compute a z-score and p-value.

# %% [markdown]
# adapted cluster comparison for just my section

# %%
# Check the proportion of spots assigned to each cluster when using denoised vs original data
prop <- prop.table(table(combined$peaks_snn_res.0.5, combined$dca_snn_res.0.5), 1)*100
heatmap(prop[order(nrow(prop):1),], 
        Colv = NA, Rowv = NA, scale="none", 
        xlab="peaks cluster", ylab="dca clusters", 
        col = hm_colors, RowSideColors=rev(colors_okate), ColSideColors = colors_okate)

# %%
saveRDS(combined, "output/sATAC_preprocessing/3_combined_compared_brca_denoised_peaks.rds")

# %%

# calculate gene activity
DefaultAssay(combined) <- "peaks"
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- 'UCSC'
Annotation(combined) <- annotations
gene.activities <- GeneActivity(combined)

# add the gene activity matrix to the Seurat object as a new assay and normalize it
#remove PCDH and UGT genes
gene.activities <- gene.activities[-grep("PCDH", rownames(gene.activities)),]
gene.activities <- gene.activities[-grep("UGT", rownames(gene.activities)),]
combined[['RNA']] <- CreateAssayObject(counts = gene.activities)
write.csv(combined@assays$RNA@counts, "output/sATAC_preprocessing/gene_activity_combined_brca.csv")
