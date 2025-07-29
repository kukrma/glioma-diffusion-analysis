# Analysis of Peritumoral and Periedematous Diffusion Properties in Brain Gliomas Using the UCSF-PDGM Dataset
This GitHub repository contains the code and derived data for the diploma thesis *Analysis of White Matter Diffusion Properties in the Context of Selected Brain Tumors* produced at the **Department of Computer Science and Engineering, Faculty of Applied Sciences, University of West Bohemia**. The thesis was successfully defended on 16th June 2025.

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## How to Use
To use this code, you first need to install Python on your computer and optionally some IDE in which you can comfortably interact with the scripts. Specifically, I have used **Python version 3.11.4** and the following libraries:
| LIBRARY      | VERSION     |
| ------------ | ----------- |
| antspyx      | 0.4.2       |
| cliffs-delta | 1.0.0       |
| dcor         | 0.6         |
| dipy         | 1.9.0       |
| matplotlib   | 3.7.2       |
| nibabel      | 5.2.1       |
| numpy        | 1.25.2      |
| optuna       | 4.2.1       |
| os           | built-in    |
| pandas       | 2.1.0       |
| scikit-image | 0.22.0      |
| scikit-learn | 1.3.0       |
| scipy        | 1.11.2      |
| seaborn      | 0.13.0      |
| time         | built-in    |
| tqdm         | 4.66.1      |
| umap-learn   | 0.5.7       |

With everything prepared, the code should be ready to use. Given that the data-specific nature of the analysis, the code is not fully generalized, but tailored specifically to the publicly available **UCSF-PDGM dataset**, which can be downloaded from the following link:

[<img src="https://img.shields.io/badge/TCIA-10.7937/tcia.bdgf--8v37-%23e41154">](https://doi.org/10.7937/tcia.bdgf-8v37)

For a reader who is interested only in the analytical stage of the workflow, the derived preprocessed diffusion properties are provided via `peritumoral.csv` and `periedematous.csv` files (see below for more details).

The source code files contain many comments with descriptions of individual steps and parameters used by the defined classes and their methods. Applying the analysis to other brain diffusion MRI datasets should be possible, but will need additional adjustments based on the analyzed data.


## Description of Files
The repsitory contains the following source codes:
1) `analysis.py` – implements the analysis of diffusion properties, i.e. EDA, manifold learning, cluster analysis, and post-clustering analysis;
2) `preprocessing.py` – implements the image processing, i.e. DWI registration, ROI generation, and computation of diffusion properties via DTI and CSD;
3) `utils.py` – implements the classes and methods used by other scripts;
4) `visualization.py` – implements various visualizations that did not fit well elsewhere, i.e. plots of original data, visualizations of DTI and CSD slices, and visualizations of structuring elements;

The file `utils.py` is called by all other scripts and organizes the functionality into three classes treated as toolkits:
1) `PreprocessingToolkit` – contains methods related to image processing and data preprocessing:
   | METHOD                  | DESCRIPTION                                                                           | 
   | ----------------------- | ------------------------------------------------------------------------------------- |
   | `registration_4Dto3D()` | Used to perform the registration of DWI data to the patient-specific space.           |
   | `generate_ROI()`        | Used to generate ROIs around tumors and edemas.                                       |
   | `model_CSD()`           | Used to compute CSD and selected derived characteristics.                             |
   | `model_DTI()`           | Used to compute DTI and selected derived characteristics.                             |
   | `change_labels()`       | Used to change labels inside columns of a dataframe (they can be too long for plots). |
   
2) `VisualizationToolkit` – contains methods for visualizations:
   | METHOD             | DESCRIPTION                                                                                                              | 
   | ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
   | `__init__()`       | Contains shared color palettes.                                                                                          |
   | `explore_slices()` | Used for simple 3D or 4D visualizations to inspect the spatial data from the Python script without the need to open FSL. |
   | `barplot()`        | Used to create a simple barplot (only counts per categories).                                                            |
   | `histogram()`      | Used to create a combined histogram with KDE.                                                                            |
   | `heatmap()`        | Used to create a heatmap of a correlation matrix.                                                                        |
   | `violin()`         | Used to create a violinplot with individual points.                                                                      |
   | `scatter()`        | Used to create a scatterplot.                                                                                            |
   | `dti_ellipsoids()` | Used to visualize DTI ellipsoids.                                                                                        |
   | `csd_glyphs()()`   | Used to visualize CSD glyphs.                                                                                            |
   | `fodf_sphere()`    | Used to visualize a single fODF.                                                                                         |
   | `structuring_el()` | Used to visualize the structuring element.                                                                               |

4) `AnalysisToolkit` – contains metods for the statistical analysis and ML algorithms.
   | METHOD                    | DESCRIPTION                                                                        | 
   | ------------------------- | ---------------------------------------------------------------------------------- |
   | `correlation()`           | Used to compute linear or non-linear correlation and show it using a heatmap.      |
   | `umap_manifold()`         | Used to compute UMAP on a range of data (UMAP parameters can be optimized).        |
   | `gmm_clustering()`        | Used to compute GMM on a range of data (number of clusters can be optimized).      |
   | `relation_quantitative()` | Used to explore the relationship between a quantitative and qualitative attribute. |
   | `relation_qualitative()`  | Used to explore the relationship between two qualitative attributes.               |

## Original Directory Tree
The provided source code contains paths to various files, and so it is deemed necessary to provide the project directory tree for easier replication or modification:
```
DPthesis/
├── data/
│   ├── preprocessed/
│   │   ├── csd/
│   │   │   ├── periedematous/
│   │   │   │   ├── UCSF-PDGM-0004_CSD_periedematous.npy
│   │   │   │   ├── UCSF-PDGM-0005_CSD_periedematous.npy
│   │   │   │   └── ...
│   │   │   └── peritumoral/
│   │   │       ├── UCSF-PDGM-0004_CSD_peritumoral.npy
│   │   │       ├── UCSF-PDGM-0005_CSD_peritumoral.npy
│   │   │       └── ...
│   │   ├── dti/
│   │   │   ├── periedematous/
│   │   │   │   ├── UCSF-PDGM-0004_DTI_periedematous.npy
│   │   │   │   ├── UCSF-PDGM-0005_DTI_periedematous.npy
│   │   │   │   └── ...
│   │   │   └── peritumoral/
│   │   │       ├── UCSF-PDGM-0004_DTI_peritumoral.npy
│   │   │       ├── UCSF-PDGM-0005_DTI_peritumoral.npy
│   │   │       └── ...
│   │   ├── dwi/
│   │   │   ├── UCSF-PDGM-0004_DWI.nii.gz
│   │   │   ├── UCSF-PDGM-0005_DWI.nii.gz
│   │   │   └── ...
│   │   ├── roi/
│   │   │   ├── periedematous/
│   │   │   │   ├── UCSF-PDGM-0004_ROI_periedematous.nii.gz
│   │   │   │   ├── UCSF-PDGM-0005_ROI_periedematous.nii.gz
│   │   │   │   └── ...
│   │   │   └── peritumoral/
│   │   │       ├── UCSF-PDGM-0004_ROI_peritumoral.nii.gz
│   │   │       ├── UCSF-PDGM-0005_ROI_peritumoral.nii.gz
│   │   │       └── ...
│   │   ├── periedematous.csv
│   │   └── peritumoral.csv
│   └── UCSF-PDGM/
│       ├── PKG-UCSF-PDGM-v3-20230111/
│       │   └── UCSF-PDGM-v3/
│       │       ├── UCSF-PDGM-0004_nifti/
│       │       │   ├── UCSF-PDGM-0004_ADC.nii.gz
│       │       │   ├── UCSF-PDGM-0004_ASL.nii.gz
│       │       │   ├── UCSF-PDGM-0004_brain_parenchyma_segmentation.nii.gz
│       │       │   ├── UCSF-PDGM-0004_brain_segmentation.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DTI_eddy_FA.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DTI_eddy_L1.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DTI_eddy_L2.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DTI_eddy_L3.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DTI_eddy_MD.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DTI_eddy_noreg.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DWI.nii.gz
│       │       │   ├── UCSF-PDGM-0004_DWI_bias.nii.gz
│       │       │   ├── UCSF-PDGM-0004_FLAIR.nii.gz
│       │       │   ├── UCSF-PDGM-0004_FLAIR_bias.nii.gz
│       │       │   ├── UCSF-PDGM-0004_SWI.nii.gz
│       │       │   ├── UCSF-PDGM-0004_SWI_bias.nii.gz
│       │       │   ├── UCSF-PDGM-0004_T1.nii.gz
│       │       │   ├── UCSF-PDGM-0004_T1_bias.nii.gz
│       │       │   ├── UCSF-PDGM-0004_T1c.nii.gz
│       │       │   ├── UCSF-PDGM-0004_T1c_bias.nii.gz
│       │       │   ├── UCSF-PDGM-0004_T2.nii.gz
│       │       │   ├── UCSF-PDGM-0004_T2_bias.nii.gz
│       │       │   └── UCSF-PDGM-0004_tumor_segmentation.nii.gz
│       │       ├── UCSF-PDGM-0005_nifti/...
│       │       └── ...
│       ├── UCSF-PDGM_DTI.bval
│       ├── UCSF-PDGM_DTI.bvec
│       └── UCSF-PDGM_metadata_v2.csv
├── img/...
├── analysis.py
├── preprocess.py
├── utils.py
└── visualization.py
```

## Preprocessed Data
In addition to the source code, the derived diffusion properties in peritumoral and periedematous ROIs are provided. Although the UCSF-PDGM dataset is publicly available and therefore the entire workflow should be reproducible, the image processing stage (i.e. the chapter in the thesis named *4. Image Processing*) can take few tens of hours to complete (depending on the hardware capabilities of the reader), and so readers interested only in the analytical stage (i.e. the chapter in the thesis named *5. Analysis of Diffusion Properties*) would be hindered by the need to first perform the image processing stage. Therefore, the computed diffusion properties are provided, as they present a major milestone in the analysis.

Both the `peritumoral.csv` and `periedematous.csv` files contain the same clinical markers and the diffusion properties, only for different regions (i.e. peritumoral or periedematous ROIs) based on their name. Specifically, the attributes are:
| ATTRIBUTE    | TYPE    | ORIGIN    | DESCRIPTION                                                                                                                       |
| ------------ | ------- | --------- | --------------------------------------------------------------------------------------------------------------------------------- |
| *ID*         | ordinal | UCSF-PDGM | unique identification of the subject                                                                                              |
| *Sex*        | nominal | UCSF-PDGM | sex of the subject                                                                                                                |
| *Age*        | integer | UCSF-PDGM | age in years at time of imaging                                                                                                   |
| *Grade*      | ordinal | UCSF-PDGM | grade based on the WHO CNS5                                                                                                       |
| *Type*       | nominal | UCSF-PDGM | final pathologic diagnosis based on the WHO CNS5                                                                                  |
| *MGMTstatus* | nominal | UCSF-PDGM | status of the MGMT biomarker                                                                                                      |
| *MGMTindex*  | integer | UCSF-PDGM | index developed at UCSF indicating the number of promoter methylation sites                                                       |
| *1p/19q*     | nominal | UCSF-PDGM | status of of 1p and 19q genes, assayed by fluorescent in-situ hybridization                                                       |
| *IDH*        | nominal | UCSF-PDGM | IDH subtype characterized with a capture-based targeted next-generation DNA sequencing panel                                      |
| *AliveDead*  | binary  | UCSF-PDGM | survival at last clinical follow up                                                                                               |
| *OS*         | integer | UCSF-PDGM | OS in days from initial diagnosis to last clinical follow up                                                                      |
| *EoR*        | nominal | UCSF-PDGM | extent of resection determined by review of operative reports and immediate postoperative imaging                                 |
| *Biopsy*     | binary  | UCSF-PDGM | if burr hole biopsy was performed                                                                                                 |
| *Ratio*      | float   | CSD       | ratio between smallest versus largest eigenvalue of the response function (identical for both periedematous and peritumoral data) |
| *NE*         | float   | CSD       | normalized entropy                                                                                                                |
| *GFAmed*     | float   | CSD       | median of GFA in ROI                                                                                                              |
| *GFAiqr*     | float   | CSD       | IQR of GFA in ROI                                                                                                                 |
| *MAGmed*     | float   | CSD       | median of MAG in ROI                                                                                                              |
| *MAGiqr*     | float   | CSD       | IQR of MAG in ROI                                                                                                                 |
| *FAmed*      | float   | DTI       | median of FA in ROI                                                                                                               |
| *FAiqr*      | float   | DTI       | IQR of FA in ROI                                                                                                                  |
| *MDmed*      | float   | DTI       | median of MD in ROI                                                                                                               |
| *MDiqr*      | float   | DTI       | IQR of MD in ROI                                                                                                                  |
| *RDmed*      | float   | DTI       | median of RD in ROI                                                                                                               |
| *RDiqr*      | float   | DTI       | IQR of RD in ROI                                                                                                                  |
| *ADmed*      | float   | DTI       | median of AD in ROI                                                                                                               |
| *ADiqr*      | float   | DTI       | IQR of AD in ROI                                                                                                                  |

The detailed description including formulas and explained abbreviations is provided in the thesis.
