# Machine Learning approach for miRNA target site prediction
## Summary
<p>This is my Master Thesis' repository, where the goal is to train a machine/deep learning model that predicts whether a microRNA binds to a target mRNA. The final model achieves an Average Precision score of 84-85% and a ROC-AUC score of 81%. The dataset used was obtained from ***Gresova, K. et al. (2025).*** and is based on results from an AGO2-eCLIP experiment from ***Manakov et al. (2022)***. Negative samples were generated for this dataset in order to mitigate class imbalance via a clustering based sampling method implemented by ***Gresova, K. et al. (2025).*** and the dataset was further corrected for false negative samples based on data from Tarbase that contains experimentally validated miRNA-mRNA interactions. Details on the tools, features and models used as well as usage instructions can be found below. **It is highly recommended that you use the Pixi package manager that supports installing packages via either uv or conda. The dependency files can be immediately used by Pixi to install the appropriate files and their versions (cd msc-thesis/dependencies -> pixi install)**. (https://pixi.prefix.dev/latest/installation/)
</p>

## Usage Instructions


## Citations
- Gresova, K., Sammut, S., Tzimotoudis, D., Klimentova, E., Cechak, D., & Alexiou, P. (2025). miRBench datasets (Version v6) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14734014 bioRxiv 2022.02.13.480296; doi: https://doi.org/10.1101/2022.02.13.480296
- Manakov, S. A., Shishkin, A. A., Yee, B. A., Shen, K. A., Cox, D. C., Park, S. S., Foster, H. M., Chapman, K. B., Yeo, G. W., & Van Nostrand, E. L. (2022). Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. bioRxiv. https://doi.org/10.1101/2022.02.13.480296

