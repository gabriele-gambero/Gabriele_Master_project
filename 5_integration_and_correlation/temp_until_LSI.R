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

# %%
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
# # 1. - Creating the initial sATAC object

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
# In the Signac [tutorial](https://stuartlab.org/signac/articles/pbmc_vignette) they do:
# ```r
# chrom_assay <- CreateChromatinAssay(
#   counts = counts,
#   sep = c(":", "-"),
#   fragments = "10k_pbmc_ATACv2_nextgem_Chromium_Controller_fragments.tsv.gz",
#   min.cells = 10,
#   min.features = 200
# )
# ```

# %%
signac_object <- list()

# create signac objects for each section and run normalization and dimensionality reduction
for(i in seq_along(infoTable$spotfiles)){
  
  # creating the Chromatin Assay
  assay <- CreateChromatinAssay(object$mtx[[i]], 
                                fragments = object$frag[[i]])
  
  # creating the signac object and performing an initial filtering 
  signac_object[[i]] <- CreateSeuratObject(assay, 
                                           assay = "peaks", 
                                           meta.data=object$md[[i]],
                                           min.cells = 10,
                                           min.features = 200)

  signac_object[[i]]$section <- rownames(infoTable)[i]
  signac_object[[i]]$sample <- i
  
  # compute LSI
  # signac_object[[i]] <- FindTopFeatures(signac_object[[i]], min.cutoff = 10)
  # signac_object[[i]] <- RunTFIDF(signac_object[[i]])
  # signac_object[[i]] <- RunSVD(signac_object[[i]])
}

# naming the objects in the correct way
names(signac_object) <- rownames(infoTable)
signac_object

# %% [markdown]
# Not normalised with LSI in this case.
# 
# The normalisation happens on the `data` object.

# %% [markdown]
# ## 2.1 - Adding the spatial data to meta

# %%
# add spatial data to meta
for(i in seq_along(infoTable$tissue_paths)){
  # tissue_md[[i]]$barcode <- paste0(tissue_md[[i]]$barcode, "_", i)
  tissue_md[[i]]$barcode <- paste0(tissue_md[[i]]$barcode, "_", i)
  rownames(tissue_md[[i]]) <- tissue_md[[i]]$barcode
  signac_object[[i]] <- AddMetaData(signac_object[[i]], tissue_md[[i]])
}

# %%
names(signac_object)

# %% [markdown]
# ## 2.2 - Adding the images

# %%
# Create a list to store Seurat objects, each with its own spatial image
seurat_list <- list()

# Create a metadata table (infoTable) containing file paths for each sample
table <- list()
for(i in names(signac_object)){
  table[[i]] <- c(samples = paste0(PATH_TO_sATAC_DATA_FOLDER, "outs/", i, "/brca_peak_bc_matrix.h5"),
                  spotfiles = paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/", i, "_tissue.csv"),
                  imgs = paste0(PATH_TO_sATAC_DATA_FOLDER, "spatial/", i, "_cropped.jpg"))
}

infoTable <- do.call("rbind", table) %>% as.data.frame()

# Iterate over all samples to process them separately
for (i in seq_along(dirs)) {
  sample_name <- dirs[i]
  
  # Read spatial barcode positions
  xy.raw <- setNames(read.csv(file = infoTable$spotfiles[i]), 
                     nm = c("barcode", "tissue", "y", "x", "pixel_y", "pixel_x"))
  xy <- xy.raw[, c("x", "y")]
  
  # Load and scale the corresponding image
  img_path <- infoTable$imgs[i]
  img <- readJPEG(img_path)
  
  sf <- c(ncol(img)/128, nrow(img)/78)
  xy$x <- xy$x * sf[1]
  xy$y <- xy$y * sf[2]
  
  # Create a spot selection table with updated pixel coordinates
  spotfile <- data.frame(xy.raw$barcode, xy.raw$tissue, xy.raw$y, xy.raw$x, round(xy$y), round(xy$x))
  spotfile_path <- paste0(strsplit(infoTable$spotfiles[i], ".csv"), "_positions_list.csv")
  write.table(spotfile, file = spotfile_path, sep = ",", quote = F, row.names = F, col.names = F)
  
  # Update infoTable to reference the new spotfile
  infoTable$spotfiles[i] <- spotfile_path
  
  # Create a Seurat object for each sample
  se <- InputFromTable(infoTable[i, , drop = FALSE], scaleVisium = 1)
  se <- LoadImages(se, time.resolve = FALSE)
  
  # Store the Seurat object with the sample name
  # seurat_list[[sample_name]] <- se
  signac_object[[i]]@tools[["Staffli"]] <- se@tools$Staffli # create STutility object for spatial plots
}


# %% [markdown]
# ---

# %% [markdown]
# ## 2.3 - Selecting just my sample

# %%
# Extract Only `220327_C1`
signac_selected <- signac_object[[SELECTED_sATAC_SAMPLE]]

# Save Processed Object
# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, paste0("2", SELECTED_sATAC_SAMPLE, "_signac_object.rds")))
# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, paste0("2_", SELECTED_sATAC_SAMPLE, "_selected_signac_object.rds")))

# Print Summary
signac_selected

# %%
signac_selected@tools[["Staffli"]]

# %%
dim(signac_selected)

# %% [markdown]
# ---
# # 3. - Quality Control
# 
# As a part of the preprocesing, first of all, we remove the features that correspond to chromosome scaffolds e.g. (KI270713.1) or other sequences instead of the (22+2) standard chromosomes.

# %% [markdown]
# ## 3.1 - Keeping only normal chromosomes

# %%
peaks.keep <- seqnames(granges(signac_selected)) %in% standardChromosomes(granges(signac_selected))
signac_selected <- signac_selected[as.vector(peaks.keep), ]

# %% [markdown]
# ## 3.2 - Attaching the correct annotation

# %% [markdown]
# We can also add gene annotations to the object for the human genome. This will allow downstream functions to pull the gene annotation information directly from the object.
# 
# Multiple patches are released for each genome assembly. When dealing with mapped data, it is advisable to use the annotations from the same assembly patch that was used to perform the mapping.
# 
# Usually, you can recover the used reference from the dataset summary; in this case, we will retrieve the information from the spatial ATAC article or from the code itself (where they just say "hg38").\
# Thanks to this, we can see that the reference used to perform the mapping was “GRCh38-2020-A”, which corresponds to the Ensembl v98 patch release.

# %%
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86) #GRCh38.p7, Oct 2016 -> v86
seqlevels(annotations) <- paste0('chr', seqlevels(annotations)) # just like UCSC genome annotations
genome(annotations) <- "hg38"
suppressWarnings(Annotation(signac_selected) <- annotations)

# %% [markdown]
# ## 3.3 - Computing QC Metrics
# We can now compute some QC metrics for the scATAC-seq experiment. We currently suggest the following metrics below to assess data quality. As with scRNA-seq, the expected range of values for these parameters will vary depending on your biological system, cell viability, and other factors.
# 
# Remember: Signac extracts the fragment counts per peak for each spot. Then:
# 1.  `CreateChromatinAssay()` stores the peak x spot count matrix.
# 2.  `CreateSeuratObject()` automatically computes:
# 
#     - `nCount_peaks` = Total number of fragments per spot.
#     - `nFeature_peaks` = The number of distinct peaks (open chromatin regions) that contain at least one read in a given spot.
# 
# So, we are going to check:
# -   Nucleosome banding pattern: The histogram of DNA fragment sizes (determined from the paired-end sequencing reads) should exhibit a strong nucleosome banding pattern corresponding to the length of DNA wrapped around a single nucleosome. We calculate this per single cell, and quantify the approximate ratio of mononucleosomal to nucleosome-free fragments (stored as nucleosome_signal)
# 
# -   Transcriptional start site (TSS) enrichment score: The ENCODE project has defined an ATAC-seq targeting score based on the ratio of fragments centered at the TSS to fragments in TSS-flanking regions (see https://www.encodeproject.org/data-standards/terms/). Poor ATAC-seq experiments typically will have a low TSS enrichment score. We can compute this metric for each cell with the `TSSEnrichment()` function, and the results are stored in metadata under the column name TSS.enrichment.
# 
# -   Total number of fragments in peaks: A measure of cellular sequencing depth / complexity. Cells with very few reads may need to be excluded due to low sequencing depth. Cells with extremely high levels may represent doublets, nuclei clumps, or other artefacts.
# 
# -   Fraction of fragments in peaks: Represents the fraction of all fragments that fall within ATAC-seq peaks. Cells with low values (i.e. <15-20%) often represent low-quality cells or technical artifacts that should be removed. Note that this value can be sensitive to the set of peaks used.
# 
# -   Ratio reads in genomic blacklist regions: The ENCODE project has provided a list of [blacklist regions](https://github.com/Boyle-Lab/Blacklist), representing reads which are often associated with artefactual signal. Cells with a high proportion of reads mapping to these areas (compared to reads mapping to peaks) often represent technical artifacts and should be removed. ENCODE blacklist regions for human (hg19 and hg38), mouse (mm9 and mm10), Drosophila (dm3 and dm6), and C. elegans (ce10 and ce11) are included in the Signac package. The `FractionCountsInRegion()` function can be used to calculate the fraction of all counts within a given set of regions per cell. We can use this function and the blacklist regions to find the fraction of blacklist counts per cell.
# 
# Note that the last three metrics can be obtained from the output of CellRanger (which is stored in the object metadata).

# %%
print(names(signac_selected@meta.data))

# %%
# compute nucleosome signal score per cell
signac_selected <- NucleosomeSignal(object = signac_selected)

# compute TSS enrichment score per cell
signac_selected <- TSSEnrichment(object = signac_selected, fast = FALSE)

# add fraction of reads in peaks
signac_selected$fract_reads_in_peaks <- signac_selected$peak_region_fragments / signac_selected$passed_filters * 100

# Fraction of Reads in Peaks (FRiP Score)
signac_selected$FRiP <- signac_selected$nFeature_peaks / signac_selected$nCount_peaks
# Mitochondrial Read Percentage
signac_selected$mitoRatio <- signac_selected$mitochondrial / signac_selected$total

# add blacklist ratio
signac_selected$blacklist_ratio <- FractionCountsInRegion(
  object = signac_selected, 
  assay = 'peaks',
  regions = blacklist_hg38_unified
)

# %% [markdown]
# The relationship between variables stored in the object metadata can be visualized using the `DensityScatter()` function. This can also be used to quickly find suitable cutoff values for different QC metrics by setting `quantiles=TRUE`:

# %% [markdown]
# ### 3.3.1 - Density plot

# %%
DensityScatter(signac_selected, x = 'nCount_peaks', y = 'TSS.enrichment', log_x = TRUE, quantiles = TRUE)

# %% [markdown]
# ### 3.3.2 - TSS Enrichment
# **TSS Enrichment Score**
# 
# -   Measures accessibility at transcription start sites (TSS).
# -   Similar to bulk/single-cell ATAC, high scores indicate open chromatin around TSS.
# -   Good threshold: TSS enrichment > 8-10 in snATAC-seq, but might be lower in spatial ATAC.

# %%
p2 <- TSSPlot(signac_selected)
p2 + geom_smooth(method = "loess", span = 0.15, se = FALSE, color = "black", linewidth = 0.8)

# %% [markdown]
# We can also look at the fragment length periodicity for all the cells, and group by cells with high or low nucleosomal signal strength. You can see that cells that are outliers for the mononucleosomal / nucleosome-free ratio (based on the plots above) have different nucleosomal banding patterns. The remaining cells exhibit a pattern that is typical for a successful ATAC-seq experiment.

# %% [markdown]
# ### 3.3.3 - Fragments length

# %%
FragmentHistogram(object = signac_selected)

# %% [markdown]
# ### 3.3.4 - FRiP score
# Fraction of Reads in Peaks (FRiP Score)
# -   Measures signal-to-noise ratio by computing the fraction of fragments that fall within called peaks.
# -   High FRiP (>0.2-0.3) indicates good quality; low FRiP (<0.1) suggests background noise.

# %%
VlnPlot(
  object = signac_selected,
  features = c('FRiP'),
  ncol = 1,
  pt.size = 0,
  group.by = "orig.ident"
  ) + NoLegend()

# %% [markdown]
# ### 3.3.5 - Total fragments per Spot
# -   Similar to fragments per cell in snATAC-seq, this measures sequencing depth per spatial location.
# -   Expected distribution: A log-normal or heavy-tailed distribution.
# -   Possible threshold: Spots with very low fragments (<1000) might be discarded.

# %%
hist(signac_selected$nCount_peaks, breaks = 50, main = "Total Fragments per Spot", col = colors_okate[4], border = "#1f8bca")


# %% [markdown]
# ### 3.3.5 - Other metrics

# %%
VlnPlot(signac_selected, features = c('nCount_peaks', 'TSS.enrichment', 'mitoRatio'), 
        ncol = 3, pt.size = 0, log = TRUE, group.by = "orig.ident")


# %% [markdown]
# ---
# # 4. - Computing LSI normalisation and clustering
# I don't think that I should normalise once again.

# %%
#preprocess raw data
signac_selected <- signac_selected %>% 
  RunTFIDF() %>%
  FindTopFeatures(min.cutoff = 'q0') %>%
  RunSVD()

# %%
DepthCor(signac_selected, n = 50)

# %%
lsi_singular_values <- signac_selected@reductions$lsi@stdev

variance_explained <- (lsi_singular_values^2) / sum(lsi_singular_values^2)
cumulative_variance <- cumsum(variance_explained)

ggplot(data.frame(PC = 1:length(cumulative_variance), Variance = cumulative_variance), aes(x = PC, y = Variance)) +
  geom_point() + geom_line() +
  xlab("LSI Components") + ylab("Cumulative Variance Explained") +
  ggtitle("LSI Scree Plot") +
  theme_minimal()

# %% [markdown]
# Here we see there is a very strong correlation between the first LSI component and the total number of counts for the cell. We will perform downstream steps without this component as we don’t want to group cells together based on their total sequencing depth, but rather by their patterns of accessibility at cell-type-specific peaks.
# 
# And then they choose 7 components, while in the tutorial, they go for `2:30`.

# %%
signac_selected <- signac_selected %>%
  RunUMAP(reduction = 'lsi', dims = 2:30, verbose = FALSE) %>%
  FindNeighbors(reduction = 'lsi', dims = 2:30, verbose = FALSE) %>%
  FindClusters(algorithm = 3, resolution = 0.5, verbose = FALSE)

# %% [markdown]
# ## 4.1 - Plotting by clusters

# %%
DimPlot(object = signac_selected, reduction = "umap", cols=colors_okate, label = FALSE)# + NoLegend()

# %%
table(signac_selected$seurat_clusters)

# %%
for (sec in unique(signac_selected$section)) {
    p <- ST.FeaturePlot(subset(signac_selected, section == sec), 
                        features = "seurat_clusters", 
                        cols = colors_okate,
                        pt.size = 1.8) +
         ggtitle(paste("Section:", sec)) +
         theme(plot.title = element_text(hjust = 0.5, size = 20), # ,family = "DM Sans Medium"
                plot.subtitle = element_blank(),  # Remove subtitle (if that's the small number))
                legend.text = element_text(size = 11),
                legend.title = element_text(size = 13))
    
    print(p)  # Display the plot
    plot_file_name <- paste0(PATH_TO_FIGURES, "Cluster_plot_for_selected_", sec, ".png")
    ggsave(filename = plot_file_name, plot = p, width = 5, height = 5, dpi = 300)
}

# %% [markdown]
# ## 4.2 - Plotting by chromatin accessibility

# %%
ST.FeaturePlot(signac_selected, 
                'nCount_peaks', 
                cols = magenta_scale,
                min.cutoff = 'q5', 
                max.cutoff = 'q95',
                pt.size = 1.8, indices = 1) +
        ggtitle(paste("Section:", signac_selected$section, "- spatial distribution of fragments number"))

# %%
# grep("chr17-397", rownames(signac_selected), value = TRUE)

ST.FeaturePlot(signac_selected, 
                'chr17-39699726-39700227', 
                cols = magenta_scale,
                min.cutoff = 'q1', 
                max.cutoff = 'q99',
                pt.size = 1.8, indices = 1) +
        ggtitle(paste("Section:", signac_selected$section, "- ERBB2 original closest accessible region"))

    # plot_file_name <- paste0(PATH_TO_FIGURES, "Cluster_plot_for_selected_", sec, ".png")
    # ggsave(filename = plot_file_name, plot = p, width = 5, height = 5, dpi = 300)


# %% [markdown]
# ## 4.3 - Saving the normalised output

# %% [markdown]
# in the original code they use `data`, which is the normalised matrix.

# %%
# #save normalised data matrix for denoising
# write.csv(signac_selected@assays$peaks@data,
#         paste0("./output/sATAC_preprocessing/data_signac_", SELECTED_sATAC_SAMPLE, "_selected_brca_q0_peak_bc_matrix2.csv"))

# %% [markdown]
# Save the R session as `.RDS` file.

# %%
# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "1_cross_mod_selected_", SELECTED_sATAC_SAMPLE, "_brca.rds"))

# %% [markdown]
# ---
# # 5. - DCA - Peaks denoiser
# 
# On terminal, in a Python environment with TensorFlow and the DCA package installed, such as `dca_mamba_37` run:\
#   - for prenormalised data:
# ```sh
# nohup dca output/sATAC_preprocessing/data_signac_<SECTION_NAME>_selected_brca_q0_peak_bc_matrix2.csv output/sATAC_preprocessing/dated_dca_<SECTION_NAME>_peaks_q0_brca --threads 6 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_data.out &
# ```

# %% [markdown]
# ---

# %%
# signac_selected <- readRDS("output/sATAC_preprocessing/1_cross_mod_selected_220327_C1_brca.rds")

# %%
# load denoised matrices and save objects - separately for peaks and gene activity due to large size
# peaks
denoised_counts <- read.table(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "dated_dca_", SELECTED_sATAC_SAMPLE, "_peaks_q0_brca/mean.tsv"), row.names = 1) %>% 
  as.matrix()
colnames(denoised_counts) <- sub("\\.", "-", colnames(denoised_counts))

# %%
dim(denoised_counts)
head(denoised_counts)

# %%
signac_selected

# %%
head(signac_selected$nCount_peaks)

# %%
# copy the assay
signac_selected[["dca"]] <- signac_selected[["peaks"]]

# %% [markdown]
# Adding the the denoised counts (still floats, not integers).

# %%
# add denoised matrix to object
signac_selected@assays$dca@data <- denoised_counts

# %%
dim(signac_selected@assays$dca@data)
head(signac_selected@assays$dca@data)

# %%
head(as.matrix(signac_selected@assays$peaks@data))

# %%
setdiff(rownames(signac_selected@assays$dca@data), rownames(signac_selected@assays$peaks@data))

# %%
length(setdiff(rownames(signac_selected@assays$peaks@data), rownames(signac_selected@assays$dca@data)))

# %%
# filter to retain same spots and features

# filtering in counts
signac_selected@assays$dca@counts <- signac_selected@assays$dca@counts[rownames(signac_selected@assays$dca@counts) %in%
                                                           rownames(signac_selected@assays$dca@data),]
# filtering in var.features
signac_selected@assays$dca@var.features <- signac_selected@assays$dca@var.features[signac_selected@assays$dca@var.features %in%
                                                                       rownames(signac_selected@assays$dca@data)]
# filtering in meta.features
signac_selected@assays$dca@meta.features <- signac_selected@assays$dca@meta.features[rownames(signac_selected@assays$dca@meta.features) %in% 
                                                                        rownames(signac_selected@assays$dca@data), , drop = FALSE]

# %%
# Keep only fragment objects with cells
signac_selected@assays$peaks@fragments <- signac_selected@assays$dca@fragments[
  sapply(signac_selected@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

signac_selected@assays$dca@fragments <- signac_selected@assays$dca@fragments[
  sapply(signac_selected@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

# %% [markdown]
# Double checking the filtering

# %%
dim(signac_selected[["dca"]]@meta.features)
dim(signac_selected[["dca"]]@data)
length(signac_selected[["dca"]]@var.features)

extra_features <- setdiff(rownames(signac_selected[["dca"]]@meta.features),
                          rownames(signac_selected[["dca"]]@data))

length(extra_features) # should print 0
print(extra_features[1:10]) # should print just NAs

# %%
rm(denoised_counts)

# %%
print(DefaultAssay(signac_selected))

# %%
dim(signac_selected@assays$dca@data)
head(signac_selected@assays$dca@data)

# %%
## compare DCA clustering
# clustering between DCA and peaks
# no need for harmony integration as the sections come from the same sample
DefaultAssay(signac_selected) <- "dca"
print(DefaultAssay(signac_selected))

# %%
signac_selected[["dca_1"]] <- CreateAssayObject(counts = denoised_counts)

# %%
DefaultAssay(signac_selected) <- "dca_1"
print(DefaultAssay(signac_selected))

# %%
signac_selected <- RunTFIDF(signac_selected, verbose = FALSE) %>%
  FindTopFeatures(min.cutoff = 'q0') %>%
  RunSVD()

# %%
saveRDS(signac_selected, "output/sATAC_preprocessing/temp_after_LSI.rds")

# %% [markdown]
# It's identical

# %%
# DepthCor(signac_selected, n = 50)

# # %%
# lsi_singular_values <- signac_selected@reductions$lsi@stdev

# variance_explained <- (lsi_singular_values^2) / sum(lsi_singular_values^2)
# cumulative_variance <- cumsum(variance_explained)

# ggplot(data.frame(PC = 1:length(cumulative_variance), Variance = cumulative_variance), aes(x = PC, y = Variance)) +
#   geom_point() + geom_line() +
#   xlab("LSI Components") + ylab("Cumulative Variance Explained") +
#   ggtitle("LSI Scree Plot") +
#   theme_minimal()

# # %%
# signac_selected <- signac_selected %>%
#   RunUMAP(reduction = "lsi", dims = 2:30, verbose = FALSE) %>%
#   FindNeighbors(reduction = "lsi", dims = 2:30, verbose = FALSE) %>%
#   FindClusters(resolution = 0.5, verbose = FALSE, cluster.name = 'seurat_clusters_dca')

# # %%
# signac_selected$dca_snn_res.0.5 <- signac_selected$seurat_clusters_dca

# # %%
# # the first two are identical
# table(signac_selected$peaks_snn_res.0.5)
# table(signac_selected$seurat_clusters)
# table(signac_selected$seurat_clusters_dca)
# table(signac_selected$dca_snn_res.0.5)

# # %%
# head(as.matrix(signac_selected@assays$dca@data))

# # %%
# head(as.matrix(signac_selected@assays$peaks@data))

# # %%
# message("FINISHED DCA CLUSTERING!")

# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "2_cross_mod_selected_", SELECTED_sATAC_SAMPLE, "_n_clustered_brca_denoised_peaks.rds"))

# # %% [markdown]
# # Still don't know why I have to do this, it's literally the same object.

# # %%
# # levels(signac_selected$peaks_snn_res.0.5) <- c("2", "4", "1", "3", "0")
# # levels(signac_selected$dca_snn_res.0.5) <- c("2", "3", "0", "4", "1")

# # %% [markdown]
# #   - `FeatureMatrix` → Construct a feature x cell matrix from a genomic fragments file.
# #   - `write10xCounts` → `rdrr` package function → Create a directory containing the count matrix and cell/gene annotation from a sparse matrix of UMI counts, in the format produced by the CellRanger  software suite. 
# #   - `GeneActivity` → Compute counts per cell in gene body and promoter region.
# #   - `seqlevelsStyle` → The `seqlevelsStyle` getter and setter can be used to get the current seqlevels style of an object and to rename its seqlevels according to a given style. 
# #   - `FindAllMarkers` → `Seurat` package function → Finds markers (differentially expressed genes) for each of the identity classes in a dataset (peaks in our case).
# #   - `RegionStats` → Compute the GC content, region lengths, and dinucleotide base frequencies for regions in the assay and add to the feature metadata.
# #   - `LinkPeaks` → Find peaks that are correlated with the expression of nearby genes. For each gene, this function computes the correlation coefficient between the gene expression and accessibility of each peak within a given distance from the gene TSS, and computes an expected correlation coefficient for each peak given the GC content, accessibility, and length of the peak. The expected coefficient values for the peak are then used to compute a z-score and p-value.

# # %% [markdown]
# # adapted cluster comparison for just my section

# # %%
# prop.table(table(signac_selected$peaks_snn_res.0.5, signac_selected$dca_snn_res.0.5), 1)*100

# # %%
# # Check the proportion of spots assigned to each cluster when using denoised vs original data
# prop <- prop.table(table(signac_selected$peaks_snn_res.0.5, signac_selected$dca_snn_res.0.5), 1)*100

# # Ensure colors match the number of clusters
# row_colors <- rev(colors_okate[1 : nrow(prop)])
# col_colors <- colors_okate[1 : ncol(prop)]

# heatmap(prop[order(nrow(prop):1),], 
#         Colv = NA, Rowv = NA, scale="none", 
#         xlab="Original peaks clusters", ylab="DCA denoised clusters", 
#         col = hm_colors, 
#         RowSideColors = row_colors, 
#         ColSideColors = col_colors)

# # %% [markdown]
# # ## 4.1 - Plotting by clusters

# # %%
# DimPlot(object = signac_selected, reduction = "umap", cols=colors_okate, label = FALSE)# + NoLegend()

# # %%
# for (sec in unique(signac_selected$section)) {
#     p <- ST.FeaturePlot(subset(signac_selected, section == sec), 
#                         features = "seurat_clusters_dca", 
#                         cols = colors_okate,
#                         pt.size = 1.8) +
#          ggtitle(paste("Section:", sec)) +
#          theme(plot.title = element_text(hjust = 0.5, size = 20), # ,family = "DM Sans Medium"
#                 plot.subtitle = element_blank(),  # Remove subtitle (if that's the small number))
#                 legend.text = element_text(size = 11),
#                 legend.title = element_text(size = 13))
    
#     print(p)  # Display the plot
#     plot_file_name <- paste0(PATH_TO_FIGURES, "Cluster_plot_for_selected_", sec, ".png")
#     ggsave(filename = plot_file_name, plot = p, width = 5, height = 5, dpi = 300)
# }

# # %% [markdown]
# # ## 4.1 - Plotting by clusters

# # %%
# # grep("chr17-397", rownames(signac_selected), value = TRUE)

# ST.FeaturePlot(signac_selected, 
#                 'chr17-39699726-39700227', 
#                 cols = magenta_scale,
#                 min.cutoff = 'q1', 
#                 max.cutoff = 'q99',
#                 pt.size = 1.8, indices = 1) +
#         ggtitle(paste("Section:", signac_selected$section, "- ERBB2 closest accessible region"))

#     # plot_file_name <- paste0(PATH_TO_FIGURES, "Cluster_plot_for_selected_", sec, ".png")
#     # ggsave(filename = plot_file_name, plot = p, width = 5, height = 5, dpi = 300)


# # %% [markdown]
# # ## 4.3 - Saving the normalised output

# # %% [markdown]
# # in the original code they use `data`, which is the normalised matrix.

# # %%
# # #save normalised data matrix for denoising
# write.csv(signac_selected@assays$peaks@data,
#         paste0("./output/sATAC_preprocessing/data_signac_", SELECTED_sATAC_SAMPLE, "_selected_brca_q0_peak_bc_matrix2.csv"))

# # %% [markdown]
# # Save the R session as `.RDS` file.

# # %%
# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "1_cross_mod_selected_", SELECTED_sATAC_SAMPLE, "_brca.rds"))

# # %% [markdown]
# # ---
# # # 5. - DCA - Peaks denoiser
# # 
# # On terminal, in a Python environment with TensorFlow and the DCA package installed, such as `dca_mamba_37` run:\
# #   - for prenormalised data:
# # ```sh
# # nohup dca output/sATAC_preprocessing/data_signac_<SECTION_NAME>_selected_brca_q0_peak_bc_matrix2.csv output/sATAC_preprocessing/dated_dca_<SECTION_NAME>_peaks_q0_brca --threads 6 --nosizefactors --nonorminput --nologinput --saveweights --nocheckcounts > nohup_for_dca_data.out &
# # ```

# # %% [markdown]
# # ---

# # %%
# # load denoised matrices and save objects - separately for peaks and gene activity due to large size
# # peaks
# denoised_counts <- read.table(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "dated_dca_", SELECTED_sATAC_SAMPLE, "_peaks_q0_brca/mean.tsv"), row.names = 1) %>% 
#   as.matrix()
# colnames(denoised_counts) <- sub("\\.", "-", colnames(denoised_counts))

# # %%
# signac_selected

# # %%
# # copy the assay
# signac_selected[["dca"]] <- signac_selected[["peaks"]]

# # %% [markdown]
# # Adding the the denoised counts (still floats, not integers).

# # %%
# # add denoised matrix to object
# signac_selected@assays$dca@data <- denoised_counts

# # %%
# # filter to retain same spots and features

# # filtering in counts
# signac_selected@assays$dca@counts <- signac_selected@assays$dca@counts[rownames(signac_selected@assays$dca@counts) %in%
#                                                            rownames(signac_selected@assays$dca@data),]
# # filtering in var.features
# signac_selected@assays$dca@var.features <- signac_selected@assays$dca@var.features[signac_selected@assays$dca@var.features %in%
#                                                                        rownames(signac_selected@assays$dca@data)]
# # filtering in meta.features
# signac_selected@assays$dca@meta.features <- signac_selected@assays$dca@meta.features[rownames(signac_selected@assays$dca@meta.features) %in% 
#                                                                         rownames(signac_selected@assays$dca@data), , drop = FALSE]

# # %%
# # Keep only fragment objects with cells
# signac_selected@assays$peaks@fragments <- signac_selected@assays$dca@fragments[
#   sapply(signac_selected@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

# signac_selected@assays$dca@fragments <- signac_selected@assays$dca@fragments[
#   sapply(signac_selected@assays$dca@fragments, function(x) length(Cells(x)) > 0)]

# # %% [markdown]
# # Double checking the filtering

# # %%
# dim(signac_selected[["dca"]]@meta.features)
# dim(signac_selected[["dca"]]@data)
# length(signac_selected[["dca"]]@var.features)

# extra_features <- setdiff(rownames(signac_selected[["dca"]]@meta.features),
#                           rownames(signac_selected[["dca"]]@data))

# length(extra_features) # should print 0
# print(extra_features[1:10]) # should print just NAs

# # %%
# rm(denoised_counts)

# # %%
# ## compare DCA clustering
# # clustering between DCA and peaks
# # no need for harmony integration as the sections come from the same sample
# DefaultAssay(signac_selected) <- "dca"
# signac_selected <- signac_selected %>%
#   RunTFIDF() %>%
#   FindTopFeatures(min.cutoff = 'q0') %>%
#   RunSVD()

# # %%
# signac_selected@assays$dca

# # %%
# DepthCor(signac_selected, n = 50)

# # %%
# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "3_cross_mod_selected_", SELECTED_sATAC_SAMPLE,"_compared_brca_denoised_peaks.rds"))

# # %% [markdown]
# # ---
# # # 5. - Gene activity

# # %%
# # signac_selected <- readRDS("output/sATAC_preprocessing/filtered_compared_brca_denoised_peaks.rds")

# # %%
# # calculate gene activity
# DefaultAssay(signac_selected) <- "peaks"
# annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
# seqlevelsStyle(annotations) <- 'UCSC'
# Annotation(signac_selected) <- annotations
# gene.activities <- GeneActivity(signac_selected)

# # %%
# # add the gene activity matrix to the Seurat object as a new assay and normalize it
# #remove PCDH and UGT genes
# gene.activities <- gene.activities[-grep("PCDH", rownames(gene.activities)),]
# gene.activities <- gene.activities[-grep("UGT", rownames(gene.activities)),]
# signac_selected[['RNA']] <- CreateAssayObject(counts = gene.activities)

# # %%
# write.csv(signac_selected@assays$RNA@counts, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "gene_activity_selected_", SELECTED_sATAC_SAMPLE, "_brca.csv"))

# # %% [markdown]
# # ## 5.1 - DCA for gene activity
# # Run DCA on terminal
# # ```sh
# # nohup dca output/sATAC_preprocessing/gene_activity_selected_<SAMPLE_NAME>_brca.csv output/sATAC_preprocessing/dca_gene_activity_selected_<SAMPLE_NAME>_brca --threads 6 --saveweights > nohup_for_<SAMPLE_NAME>_dca_gene_activity.out &
# # ```

# # %%
# signac_selected

# # %%
# # gene activity
# signac_selected[["dca"]] <- NULL
# mtx <- read.table(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "dca_gene_activity_selected_", SELECTED_sATAC_SAMPLE, "_brca/mean.tsv"))
# colnames(mtx) <- gsub("\\.", "-", colnames(mtx))
# DefaultAssay(signac_selected) <- "RNA"
# signac_selected <- subset(signac_selected, cells = colnames(mtx))

# # add denoised matrix to object
# signac_selected[["RNA_dca"]] <- CreateAssayObject(counts = as.matrix(mtx))
# rm(mtx)

# # %%
# saveRDS(signac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "4_cross_mod_selected_", SELECTED_sATAC_SAMPLE, "_brca_denoised_RNA.rds"))


