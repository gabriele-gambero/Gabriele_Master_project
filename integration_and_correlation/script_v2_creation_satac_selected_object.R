# %% [markdown]
# Environment name: `r_env_corr`

# %% [markdown]
# # 0. - Imports and paths

# %%
setwd(dir = "/disk2/user/gabgam/the_project/5_integration_and_correlation/")

# %%
source("./utils/source_satac.R")
library(biovizBase)
library(extrafont) # install.packages("extrafont")

# %% [markdown]
# Loading the fonts.

# %%
# Fetch DM Sans font file from https://github.com/google/fonts/blob/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf 
# or, better, https://fonts.google.com/specimen/DM+Sans
# Import only DMSans-Medium
# font_import(paths = "../fonts/static/", prompt = FALSE, recursive = FALSE, pattern = "DMSans-Medium.ttf") # select just the Medium one
# loadfonts()  # Load registered fonts

# # verify if the font is available:
# fonts()

# # %%
# # for font in Base R plots
# par(family = "DM Sans Medium")
# plot(1: 2, main = "Title in Custom Font")

# %%
PATH_TO_sATAC_DATA_FOLDER <- "../data/spatial_atac/"
PATH_sATAC_PREPROCESSING_OUTPUT <- "output/sATAC_preprocessing/"
PATH_TO_FIGURES <- "figures/sATAC_preprocessing/"

SELECTED_sATAC_SAMPLE <- "220327_C1"

# peaks obtained from https://www.science.org/doi/10.1126/science.aav1898
BRCA_PEAK_CALLS <- paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/BRCA_peakCalls.csv")

# %% [markdown]
# ---
# # 1. - Creating the initial Object

# %% [markdown]
# ## 1.1 - Creating table with paths to all files

# %%
# set paths to raw data and metadata
dirs <- list.dirs(paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/"), recursive = F, full.names = F)
table <- list()


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


# %% [markdown]
# ## 1.2 - Creating count Object for peaks

# %% [markdown]
# From the Signac [tutorial](https://stuartlab.org/signac/articles/pbmc_vignette):
# ```r
# fragpath <- '10k_pbmc_ATACv2_nextgem_Chromium_Controller_fragments.tsv.gz'
# 
# # Define cells
# # If you already have a list of cell barcodes to use you can skip this step
# total_counts <- CountFragments(fragpath)
# cutoff <- 1000 # Change this number depending on your dataset!
# barcodes <- total_counts[total_counts$frequency_count > cutoff, ]$CB
# 
# # Create a fragment object
# frags <- CreateFragmentObject(path = fragpath, cells = barcodes)
# 
# # First call peaks on the dataset
# # If you already have a set of peaks you can skip this step
# peaks <- CallPeaks(frags)
# 
# # Quantify fragments in each peak
# counts <- FeatureMatrix(fragments = frags, features = peaks, cells = barcodes)
# ```
# 

# %%

object <- list(md = list(),
               frag = list(),
               counts = list())

# build new matrices with new peak set
# peaks obtained from https://www.science.org/doi/10.1126/science.aav1898
# load peaks and create genomicranges object
gr <- read.csv(
  file = BRCA_PEAK_CALLS,
  col.names = c("chr", "start", "end"),
  sep = ";"
) %>% makeGRangesFromDataFrame()

# create fragment objects and count matrices for each section separately
for(i in seq_along(dirs)){
  # folder = paste0(PATH_TO_sATAC_DATA_FOLDER, "combined/")
  folder = strsplit(infoTable$samples[[i]], "raw_peak_bc_matrix.h5") %>% unlist()
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

# %% [markdown]
# ## 1.3 - Building the correct tissue list file

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
# Loading the files just created for each of the sections.

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
# ---
# # 2. - Chromatin Assay object and LSI computing
# why do they perform normalisation here???

# %% [markdown]
# No Quality control when importing the fragments. In the Signac [tutorial](https://stuartlab.org/signac/articles/pbmc_vignette) they do:
# ```r
# chrom_assay <- CreateChromatinAssay(
#   counts = counts,
#   sep = c(":", "-"),
#   fragments = "10k_pbmc_ATACv2_nextgem_Chromium_Controller_fragments.tsv.gz",
#   min.cells = 10,
#   min.features = 200
# )
# ```

# %% [markdown]
# They didn't even remove the features that correspond to chromosome scaffolds or sequences that aren't part of standard chromosomes like in the tutorial.

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

# %% [markdown]
# ## 2.1 - Merging all the sections

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
infoTable$spotfiles

# %% [markdown]
# ## 2.2 - Adding the images

# %%
# add image
# image is manually cropped so that it only shows the capture area
table <- list()
for(i in dirs){
  table[[i]] <- c(samples = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", i, "/brca_peak_bc_matrix.h5"),
                  spotfiles = paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/", i, "_tissue.csv"),
                  imgs = paste0(PATH_TO_sATAC_DATA_FOLDER, "spatial/", i, "_cropped.jpg"))
}

infoTable <- do.call("rbind", table) %>% as.data.frame()

for (i in 1:length(dirs)) {
  # Read *_tissue.csv file
  xy.raw <- setNames(read.csv(file = infoTable$spotfiles[i]), 
                     nm = c("barcode", "tissue", "y", "x", "pixel_y", "pixel_x"))
  xy <- xy.raw[, c("x", "y")]
  
  img_path <- infoTable$imgs[i]
  img <- readJPEG(img_path)
  
  sf <- c(ncol(img)/128, nrow(img)/78)
  xy$x <- xy$x*sf[1]
  xy$y <- xy$y*sf[2]
  
  # Create a new spot selection table with proper image pixel coordinates which match the cropped images
  spotfile <- data.frame(xy.raw$barcode, xy.raw$tissue, xy.raw$y, xy.raw$x, round(xy$y), round(xy$x))
  write.table(spotfile, file = paste0(strsplit(infoTable$spotfiles[i], ".csv"), "_positions_list.csv"), 
              sep = ",", quote = F, row.names = F, col.names = F)
}

infoTable$spotfiles <- paste0(strsplit(infoTable$spotfiles, ".csv"), "_positions_list.csv")
se <- InputFromTable(infoTable, scaleVisium = 1)
se <- LoadImages(se, time.resolve = F)
combined@tools[["Staffli"]] <- se@tools$Staffli # create STutility object for spatial plots

# %% [markdown]
# ---

# %% [markdown]
# ## 2.3. - Selecting just my sample

# %%
# Extract Only `220327_C1`
signac_filtered <- subset(combined, subset = sample == SELECTED_sATAC_SAMPLE)


# Save Processed Object
# saveRDS(signac_filtered, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, paste0("2", SELECTED_sATAC_SAMPLE, "_signac_object.rds")))
# saveRDS(signac_filtered, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, paste0("2_", SELECTED_sATAC_SAMPLE, "_selected_signac_object.rds")))

# Print Summary
print(signac_filtered)

# %%
table(combined$sample)

# %%
table(signac_filtered$sample)

# %%
dim(combined)
dim(signac_filtered)

# %% [markdown]
# # 3. - Computing LSI normalisation and clustering
# I don't think that I should normalise once again.

# %%
#preprocess raw data
signac_filtered <- signac_filtered %>% 
  RunTFIDF() %>%
  FindTopFeatures(min.cutoff = 'q0') %>%
  RunSVD()

# %%
DepthCor(signac_filtered, n = 50)

# %% [markdown]
# Here we see there is a very strong correlation between the first LSI component and the total number of counts for the cell. We will perform downstream steps without this component as we don’t want to group cells together based on their total sequencing depth, but rather by their patterns of accessibility at cell-type-specific peaks.
# 
# And then they choose 7 components, while in the tutorial, they go for `2:30`.

# %%
signac_filtered <- signac_filtered %>%
  RunUMAP(reduction = 'lsi', dims = 2:25, verbose = FALSE) %>%
  FindNeighbors(reduction = 'lsi', dims = 2:25, verbose = FALSE) %>%
  FindClusters(algorithm = 3, resolution = 0.5, verbose = FALSE)

# %% [markdown]
# ## 3.1 - Plotting by clusters

# %%
DimPlot(object = signac_filtered, reduction = "umap", cols=colors_okate)# + NoLegend()

# %%
DimPlot(object = signac_filtered, reduction = "umap", group.by = "section")# + NoLegend()

# %%
for (sec in unique(signac_filtered$section)) {
    p <- ST.FeaturePlot(subset(signac_filtered, section == sec), 
                        features = "seurat_clusters", 
                        cols = colors_okate,
                        pt.size = 1.8) +
         ggtitle(paste("Section:", sec)) +
         theme(plot.title = element_text(hjust = 0.5, size = 20), # ,family = "DM Sans Medium"
                plot.subtitle = element_blank(),  # Remove subtitle (if that's the small number))
                legend.text = element_text(size = 11),
                legend.title = element_text(size = 13))
    
    print(p)  # Display the plot
    plot_file_name <- paste0(PATH_TO_FIGURES, "Cluster_plot_for_selected_", sec, ".pdf")
    ggsave(filename = plot_file_name, plot = p, width = 5, height = 5, dpi = 300)
}

# %% [markdown]
# ## 3.2 - Saving the normalised output

# %% [markdown]
# in the original code they use `data`, but that's the normalised matrix.

# %%
# #save normalised data matrix for denoising
# write.csv(signac_filtered@assays$peaks@data,
#         paste0("./output/sATAC_preprocessing/data_signac_", SELECTED_sATAC_SAMPLE, "_filtered_brca_q0_peak_bc_matrix.csv"))

# %% [markdown]
# Save the R session as `.RDS` file.

# %%
saveRDS(signac_filtered, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "1_selected_", SELECTED_sATAC_SAMPLE, "_brca.rds"))

# %% [markdown]
# ---
# # 4. - DCA - Peaks denoiser
# 
# On terminal, in a Python environment with TensorFlow and the DCA package installed, such as `dca_mamba_37` run:\
#   - for prenormalised data:
# ```sh
# nohup dca output/sATAC_preprocessing/data_signac_<SECTION_NAME>_filtered_brca_q0_peak_bc_matrix.csv output/sATAC_preprocessing/dated_dca_<SECTION_NAME>_peaks_q0_brca --threads 3 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_data.out &
# ```

# %% [markdown]
# ---

# %%
# load denoised matrices and save objects - separately for peaks and gene activity due to large size
# peaks
denoised_counts <- read.table(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "dated_dca_", SELECTED_sATAC_SAMPLE, "_peaks_q0_brca/mean.tsv"), row.names = 1) %>% 
  as.matrix()
colnames(denoised_counts) <- sub("\\.", "-", colnames(denoised_counts))

# %%
signac_filtered

# %%
# add denoised matrix to object
# filter to retain same spots and features
signac_filtered[["dca"]] <- signac_filtered[["peaks"]]

# %% [markdown]
# Adding the the denoised counts (still floats, not integers).

# %%
signac_filtered@assays$dca@data <- denoised_counts

# %%
# filtering in counts
signac_filtered@assays$dca@counts <- signac_filtered@assays$dca@counts[rownames(signac_filtered@assays$dca@counts) %in%
                                                           rownames(signac_filtered@assays$dca@data),]
# filtering in var.features
signac_filtered@assays$dca@var.features <- signac_filtered@assays$dca@var.features[signac_filtered@assays$dca@var.features %in%
                                                                       rownames(signac_filtered@assays$dca@data)]
# filtering in meta.features
signac_filtered@assays$dca@meta.features <- signac_filtered@assays$dca@meta.features[rownames(signac_filtered@assays$dca@meta.features) %in% 
                                                                        rownames(signac_filtered@assays$dca@data), , drop = FALSE]

# %%
# Keep only fragment objects with cells
signac_filtered@assays$peaks@fragments <- signac_filtered@assays$dca@fragments[
  sapply(signac_filtered@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

signac_filtered@assays$dca@fragments <- signac_filtered@assays$dca@fragments[
  sapply(signac_filtered@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

# %% [markdown]
# Double checking the filtering

# %%
dim(signac_filtered[["dca"]]@meta.features)
dim(signac_filtered[["dca"]]@data)
length(signac_filtered[["dca"]]@var.features)

extra_features <- setdiff(rownames(signac_filtered[["dca"]]@meta.features),
                          rownames(signac_filtered[["dca"]]@data))

length(extra_features) # should print 0
print(extra_features[1:10]) # should print just NAs

# %%
rm(denoised_counts)

# %%
## compare DCA clustering
# clustering between DCA and peaks
# no need for harmony integration as the sections come from the same sample
DefaultAssay(signac_filtered) <- "dca"
signac_filtered <- RunTFIDF(signac_filtered, verbose = TRUE) %>%
  FindTopFeatures(min.cutoff = 'q0') %>%
  RunSVD() %>%
  RunUMAP(reduction = "lsi", dims = 2:9) %>%
  FindNeighbors(reduction = "lsi", dims = 2:9) %>%
  FindClusters(resolution = 0.5)

# %%
message("FINISHED DCA CLUSTERING!")

saveRDS(signac_filtered, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "2_selected_", SELECTED_sATAC_SAMPLE, "_n_clustered_brca_denoised_peaks.rds"))

# %%
levels(signac_filtered$peaks_snn_res.0.5)
levels(signac_filtered$dca_snn_res.0.5)

# %%
signac_filtered$dca_snn_res.0.5 <- signac_filtered$seurat_clusters

levels(signac_filtered$peaks_snn_res.0.5) <- c("2", "4", "1", "3", "0")
levels(signac_filtered$dca_snn_res.0.5) <- c("2", "3", "0", "4", "1")

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
prop.table(table(signac_filtered$peaks_snn_res.0.5, signac_filtered$dca_snn_res.0.5), 1)*100

# %%
# Check the proportion of spots assigned to each cluster when using denoised vs original data
prop <- prop.table(table(signac_filtered$peaks_snn_res.0.5, signac_filtered$dca_snn_res.0.5), 1)*100
heatmap(prop[order(nrow(prop):1),], 
        Colv = NA, Rowv = NA, scale="none", 
        xlab="peaks cluster", ylab="dca clusters", 
        col = hm_colors, RowSideColors=rev(colors_okate), ColSideColors = colors_okate)

# %%
saveRDS(signac_filtered, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "3_selected_", SELECTED_sATAC_SAMPLE,"_compared_brca_denoised_peaks.rds"))

# %% [markdown]
# ---
# # Gene activity

# %%
# signac_filtered <- readRDS("output/sATAC_preprocessing/filtered_compared_brca_denoised_peaks.rds")

# %%
# calculate gene activity
DefaultAssay(signac_filtered) <- "peaks"
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- 'UCSC'
Annotation(signac_filtered) <- annotations
gene.activities <- GeneActivity(signac_filtered)

# %%
# add the gene activity matrix to the Seurat object as a new assay and normalize it
#remove PCDH and UGT genes
gene.activities <- gene.activities[-grep("PCDH", rownames(gene.activities)),]
gene.activities <- gene.activities[-grep("UGT", rownames(gene.activities)),]
signac_filtered[['RNA']] <- CreateAssayObject(counts = gene.activities)

# %%
write.csv(signac_filtered@assays$RNA@counts, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "gene_activity_selected_", SELECTED_sATAC_SAMPLE, "_brca.csv"))

# %% [markdown]
# ## DCA for gene activity
# Run DCA on terminal
# ```sh
# nohup dca output/sATAC_preprocessing/gene_activity_selected_<SAMPLE_NAME>_brca.csv output/sATAC_preprocessing/dca_gene_activity_selected_<SAMPLE_NAME>_brca --threads 6 --saveweights > nohup_for_<SAMPLE_NAME>_dca_gene_activity.out &
# ```

# # %%
# signac_filtered

# # %%
# # gene activity
# signac_filtered[["dca"]] <- NULL
# mtx <- read.table(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "dca_gene_activity_selected_", SELECTED_sATAC_SAMPLE, "_brca/mean.tsv"))

# colnames(mtx) <- gsub("\\.", "-", colnames(mtx))
# DefaultAssay(signac_filtered) <- "RNA"
# signac_filtered <- subset(signac_filtered, cells = colnames(mtx))

# # add denoised matrix to object
# signac_filtered[["RNA_dca"]] <- CreateAssayObject(counts = as.matrix(mtx))
# rm(mtx)

# # %%
# saveRDS(signac_filtered, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "4_selected_", SELECTED_sATAC_SAMPLE, "_brca_denoised_RNA.rds"))


