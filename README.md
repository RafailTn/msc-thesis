# Machine Learning approach for miRNA target site prediction

## Summary
This is my Master Thesis' repository, where the goal is to train a machine/deep learning model that predicts whether a microRNA 
binds to a target mRNA. The primary model is a 2D CNN (`MiRBindCNN`) that encodes RNA complementarity geometry directly as a Watson-Crick pairing matrix and achieves an Average Precision score of 83-84% and a ROC-AUC score of 81%. The dataset 
used was obtained from ***Gresova, K. et al. (2025).*** and is based on results from an AGO2-eCLIP experiment from ***Manakov et al. (2022)***. 
Negative samples were generated for this dataset in order to mitigate class imbalance via a clustering based sampling method 
implemented by ***Gresova, K. et al. (2025).*** and the dataset was further corrected for false negative samples based on data 
from Tarbase that contains experimentally validated miRNA-mRNA interactions. Details on the tools, features and models used 
as well as usage instructions can be found below. 

## Installation 
**It is highly recommended that you use the Pixi package manager that 
supports installing packages via either uv or conda. The dependency files can be immediately used by Pixi to install the 
appropriate files and their versions (cd msc-thesis/dependencies :arrow_right: pixi install :arrow_right: pixi shell)**. 
(https://pixi.prefix.dev/latest/installation/)

Note: IntaRNA is no longer included in the pixi environment. It is only required for the legacy AutoGluon pipeline (`src/predict_target.py`) and must be installed separately if that path is used.

## Usage Instructions (CNN — primary model)
The CNN model operates directly on sequence TSVs (v7 format) and does not require IntaRNA or FASTA inputs.

**Train with k-fold cross-validation:**
```bash
bash cnn/run_train.sh kfold
```

**Score new data:**
```bash
dependencies/.pixi/envs/default/bin/python cnn/cnn_branches_mirbind.py predict \
    --checkpoint checkpoints/cnn_mirbind_fold1.pt \
    --input data/AGO2_eCLIP_Manakov2022_test_v7.tsv \
    --output results/predictions.tsv \
    --mre-col gene --mirna-col noncodingRNA
```

**Explain a prediction (which MRE positions mattered, per sample):**
```bash
dependencies/.pixi/envs/default/bin/python cnn/cnn_branches_mirbind.py explain \
    --checkpoint checkpoints/cnn_mirbind_fold1.pt \
    --input data/AGO2_eCLIP_Manakov2022_test_v7.tsv \
    --output results/explain.tsv \
    --mre-col gene --mirna-col noncodingRNA
```

The input TSV must contain `gene` (MRE sequence), `noncodingRNA` (miRNA sequence), `chr`, `strand`, `start`, `end`, and `label` columns. All `run_train.sh` defaults are overridable via environment variables (e.g. `FOLDS=10 EPOCHS=60 DEVICE=cpu`).

## Usage Instructions (AutoGluon — legacy)
The legacy AutoGluon pipeline accepts FASTA inputs and requires IntaRNA (install separately). To run predictions:
- Provide 2 FASTA files: `-target_fasta` (MRE sequences) and `-query_fasta` (miRNA sequences)
- Provide a TSV with MRE genomic coordinates via `-conservation_tsv`
- Optionally provide an alternative BigWig file via `-bigwig` if not using hg38 phastCons470way
- Optionally set a decision threshold via `-threshold` (0–1)

## TODO:
- Add a way to automatically retrieve the coordinates of the target sequences and add them to the TSV if needed
- Find and add thresholds that optimize common needs (such as prioritising TPR).

## Citations
- Gresova, K., Sammut, S., Tzimotoudis, D., Klimentova, E., Cechak, D., & Alexiou, P. (2025). miRBench datasets (Version v6) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14734014 bioRxiv 2022.02.13.480296; doi: https://doi.org/10.1101/2022.02.13.480296
- Manakov, S. A., Shishkin, A. A., Yee, B. A., Shen, K. A., Cox, D. C., Park, S. S., Foster, H. M., Chapman, K. B., Yeo, G. W., & Van Nostrand, E. L. (2022). Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. bioRxiv. https://doi.org/10.1101/2022.02.13.480296
- Klimentova, E., Klimes, F., Alexiou, P., & Cechak, D. (2022). miRBind: A Deep Learning Method for miRNA Binding Classification. Genes, 13(12), 2323. https://doi.org/10.3390/genes13122323
- Erickson, Nick, et al. "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data." arXiv preprint arXiv:2003.06505 (2020).
- Seshadri, Ram (2020). GitHub - AutoViML/featurewiz: Use advanced feature engineering strategies and select the best features from your data set fast with a single line of code. source code: https://github.com/AutoViML/featurewiz
- IntaRNA 2.0: enhanced and customizable prediction of RNA-RNA interactions Martin Mann, Patrick R. Wright, and Rolf Backofen, Nucleic Acids Research, 45 (W1), W435–W439, 2017, DOI:10.1093/nar/gkx279.
- Sethupathy, P., Corda, B., & Hatzigeorgiou, A. G. (2006). TarBase: A comprehensive database of experimentally supported animal microRNA targets. RNA (New York, N.Y.), 12(2), 192–197. https://doi.org/10.1061/rna.2239606
