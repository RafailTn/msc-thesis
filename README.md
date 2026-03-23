# Machine Learning approach for miRNA target site prediction

## Summary
This is my Master Thesis' repository, where the goal is to train a machine/deep learning model that predicts whether a microRNA 
binds to a target mRNA. The final model achieves an Average Precision score of 83-84% and a ROC-AUC score of 81%. The dataset 
used was obtained from ***Gresova, K. et al. (2025).*** and is based on results from an AGO2-eCLIP experiment from ***Manakov et al. (2022)***. 
Negative samples were generated for this dataset in order to mitigate class imbalance via a clustering based sampling method 
implemented by ***Gresova, K. et al. (2025).*** and the dataset was further corrected for false negative samples based on data 
from Tarbase that contains experimentally validated miRNA-mRNA interactions. Details on the tools, features and models used 
as well as usage instructions can be found below. 

## Installation 
**It is highly recommended that you use the Pixi package manager that 
supports installing packages via either uv or conda. The dependency files can be immediately used by Pixi to install the 
appropriate files and their versions (cd msc-thesis/dependencies :arrow_right: pixi install)**. 
(https://pixi.prefix.dev/latest/installation/)

## Usage Instructions
The model as well as the necessary files to perform feature extraction (i.e a BigWig file for the hg38, genePhastCons470way) are in the repo and will be downloaded with it via git clone.)
In order to run the **Prediction Pipeline** the user needs to provide:
- 2 FASTA files containing the nucleotide sequences of the miRNA and the MRE they wish to make predictions for using the --target_fasta (MRE) and --query_fasta(miRNA) 
- A path for a TSV containing the coordinates of the MREs of interest in a tsv via --coords. This TSV file can also contain already extracted conservation scores, if you have them precomputed, in one of its columns in a list format and the extraction will work as intended.
- A bigiwig file path for the conservation scores, **IF PhastCons470way is not to be used and there are no conservation scores in the TSV**, via --conservation_path. 
- Optionally define the threshold (0-1, via --threshold) for separating the samples in Positives (interacting) and Negatives (not interacting).

## Citations
- Gresova, K., Sammut, S., Tzimotoudis, D., Klimentova, E., Cechak, D., & Alexiou, P. (2025). miRBench datasets (Version v6) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14734014 bioRxiv 2022.02.13.480296; doi: https://doi.org/10.1101/2022.02.13.480296
- Manakov, S. A., Shishkin, A. A., Yee, B. A., Shen, K. A., Cox, D. C., Park, S. S., Foster, H. M., Chapman, K. B., Yeo, G. W., & Van Nostrand, E. L. (2022). Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. bioRxiv. https://doi.org/10.1101/2022.02.13.480296

