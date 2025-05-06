import argparse
import scanpy as sc
import pandas as pd
from dca.api import dca
import subprocess

def main(input_file, output_file, threads):
    # Load the dataset
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, index_col=0)

    # Convert to AnnData object
    adata = sc.AnnData(df)
    adata.var_names = df.columns  # Set variable (gene) names

    # Filter genes with at least 1 count
    # sc.pp.filter_genes(adata, min_counts=1)

    # Run DCA denoising
    print("Running DCA denoising...")
    dca(adata, threads=threads)

    # Save the denoised data
    print(f"Saving denoised data to {output_file}...")
    adata.write_h5ad(output_file)

    print("DCA processing completed successfully!")
    
    # Save package versions to a .txt file
    with open("requirements_for_dca_env.txt", "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise ATAC-seq data using DCA.")
    parser.add_argument("input_file", type=str, help="Path to input CSV file (raw counts).")
    parser.add_argument("output_file", type=str, help="Path to save the denoised H5AD file.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for DCA.")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.threads)

