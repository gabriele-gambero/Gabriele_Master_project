# Environment name: `r_env_corr`
# 0. - Imports and paths
setwd(dir = "/disk2/user/gabgam/the_project/5_integration_and_correlation/")
suppressMessages(suppressWarnings(source("./utils/source_satac.R")))
library(biovizBase)
library(extrafont) # install.packages("extrafont")
# Loading the fonts.
# Fetch DM Sans font file from https://github.com/google/fonts/blob/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf 
# or, better, https://fonts.google.com/specimen/DM+Sans
# Import only DMSans-Medium
# font_import(paths = "../fonts/static/", prompt = FALSE, recursive = FALSE, pattern = "DMSans-Medium.ttf") # select just the Medium one
# loadfonts()  # Load registered fonts

# # verify if the font is available:
# fonts()
# # for font in Base R plots
# par(family = "DM Sans Medium")
# plot(1: 2, main = "Title in Custom Font")
# sATAC sample paths.
PATH_TO_sATAC_DATA_FOLDER <- "../data/spatial_atac/"
PATH_sATAC_PREPROCESSING_OUTPUT <- "output/sATAC_preprocessing/"
PATH_TO_FIGURES <- "figures/sATAC_preprocessing/"

SELECTED_sATAC_SAMPLE <- "220327_C1"

# peaks obtained from https://www.science.org/doi/10.1126/science.aav1898
BRCA_PEAK_CALLS <- paste0(PATH_TO_sATAC_DATA_FOLDER, "github_folder/meta/BRCA_peakCalls.csv")
# Visium sample paths.
PATH_TO_VISIUM_SAMPLE <- "../data/visium_ffpe/output_n_suppl_files/Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5"
PATH_TO_VISIUM_MOD_SPOTFILES <- "../4_clustering_and_classification/output/UNI2-h/satac_C1_v3_allspots_&_visium_FFPE_dcis_idc_10X_img_not_changed_allspots/on_individual_samples/6_clusters/
                        tissue_position_list_with_6_kmeans_clusters_visium_FFPE_dcis_idc_10X_&_img_not_changed_allspots_&_target_is_reference_full_100um_ORIGINAL WSI.csv"
PATH_TO_VISIUM_HIGH_RES_IMAGE <- "../data/visium_ffpe/output_n_suppl_files/spatial/tissue_hires_image.png"
PATH_TO_VISIUM_JSON <- "../data/visium_ffpe/output_n_suppl_files/spatial/scalefactors_json.json"
# ---
# 1. - Loading the previously created object
# load seurat object with denoised gene activity counts
# satac_selected <- readRDS(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "4_combined_brca_denoised_RNA.rds"))
satac_selected <- readRDS(paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "4_cross_mod_selected_", SELECTED_sATAC_SAMPLE, "_brca_denoised_RNA.rds"))

satac_selected
levels(satac_selected@meta.data$`peaks_snn_res.0.5`)
# link peaks to genes
markers <- FindAllMarkers(satac_selected, logfc.threshold = 0.1, only.pos = T, assay = "RNA") %>%
  filter(p_val_adj < 0.05)
DefaultAssay(satac_selected) <- "peaks"
satac_selected <- RegionStats(satac_selected, genome = BSgenome.Hsapiens.UCSC.hg38)
satac_selected <- LinkPeaks(
  object = satac_selected,
  peak.assay = "peaks",
  expression.assay = "RNA_dca",
  genes.use = unique(markers$gene),
  min.distance = 2000
)

linked_peaks <- satac_selected@assays[["peaks"]]@links@elementMetadata %>% as.data.frame()
# write.csv(linked_peaks, "output/sATAC_preprocessing/linked_peaks_temp.csv")
write.csv(linked_peaks, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "linked_peaks_", SELECTED_sATAC_SAMPLE, "_temp.csv"))
# saveRDS(satac_selected, "output/sATAC_preprocessing/linked_peaks_temp.RDS")
saveRDS(satac_selected, paste0(PATH_sATAC_PREPROCESSING_OUTPUT, "linked_peaks_", SELECTED_sATAC_SAMPLE, "_temp.RDS"))