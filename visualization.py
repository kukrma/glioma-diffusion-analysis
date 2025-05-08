# ==================================================================================================== #
# --- DIPLOMA THESIS (visualization.py) -------------------------------------------------------------- #
# ==================================================================================================== #
# title:         Analysis of White Matter Diffusion Properties in the Context of Selected Brain Tumors #
# author:        Bc. Martin KUKRÁL                                                                     #
# supervision:   doc. Ing. Roman MOUČEK, Ph.D.                                                         #
#                doc. MUDr. Irena HOLEČKOVÁ, Ph.D. (consultant)                                        #
# academic year: 2024/2025                                                                             #
# last updated:  2025-05-07                                                                            #
# ==================================================================================================== #
# Python version      3.11.4
import numpy as np  # 1.25.2
import pandas as pd # 2.1.0
import utils        # script
# initialize custom toolkits:
preprocessing = utils.PreprocessingToolkit()
visualization = utils.VisualizationToolkit()





# 1) PLOTS OF ORIGINAL DATA ---------------------------------------------------
# load original data:
repeated = ("UCSF-PDGM-315", "UCSF-PDGM-278", "UCSF-PDGM-175", "UCSF-PDGM-138", "UCSF-PDGM-181", "UCSF-PDGM-289")
data = pd.read_csv("./data/UCSF-PDGM/UCSF-PDGM-metadata_v2.csv").drop(columns=["BraTS21 ID", "BraTS21 Segmentation Cohort", "BraTS21 MGMT Cohort"])
data = data[~data["ID"].isin(repeated)] # no repeating subjects
# print number of NaN for each column:
print(np.sum(data.isna(), axis=0), "\n")
# modify the classification labels:
data = preprocessing.change_labels(data, "Final pathologic diagnosis (WHO 2021)", ["A-IDHmut", "A-IDHwt", "G-IDHwt", "O-IDHmut"])
# modify values so that they can be better plotted:
data["WHO CNS Grade"] = data["WHO CNS Grade"].astype(str)
data["MGMT index"] = data["MGMT index"].astype("string")
data["MGMT index"] = data["MGMT index"].str.replace(r"\.0$", "", regex=True)
data["1p/19q"] = data["1p/19q"].replace("Co-deletion", "co-deletion")
data["1-dead 0-alive"] = data["1-dead 0-alive"].astype("string")
data["1-dead 0-alive"] = data["1-dead 0-alive"].replace(["1", "0"], ["dead", "alive"])
# fill-in empty values:
data["MGMT status"] = data["MGMT status"].replace(np.nan, "NaN")
data["MGMT index"] = data["MGMT index"].replace(np.nan, "NaN")
data["1p/19q"] = data["1p/19q"].replace(np.nan, "NaN")
data["EOR"] = data["EOR"].replace(np.nan, "NaN")
# generate the plots:
visualization.barplot(data, "Sex", "Sex", "number of instances", "", visualization.sex_palette, "orig_sex")
visualization.histogram(data, "Age at MRI", "Sex", "age at MRI", "probability of occurence in data", "", visualization.sex_palette, "orig_age")
visualization.histogram(data, "Age at MRI", None, "age at MRI", "probability of occurence in data", "", "slategray", "orig_age2")
visualization.histogram(data, "Age at MRI", "WHO CNS Grade", "age at MRI", "probability of occurence in data", "", visualization.grade_palette, "orig_age3")
visualization.histogram(data, "Age at MRI", "Final pathologic diagnosis (WHO 2021)", "age at MRI", "probability of occurence in data", "", visualization.type_palette, "orig_age4")
visualization.barplot(data, "WHO CNS Grade", "WHO CNS Grade", "number of instances", "", visualization.grade_palette, "orig_grade")
visualization.barplot(data, "Final pathologic diagnosis (WHO 2021)", "Final pathologic diagnosis (WHO 2021)", "number of instances", "", visualization.type_palette, "orig_type")
visualization.barplot(data, "MGMT status", "MGMT status", "number of instances", "", ["slategray"]*3+["sandybrown"], "orig_mgmtstatus")
visualization.barplot(data, "MGMT index", "MGMT index", "number of instances", "", ["slategray"]+["sandybrown"]+["slategray"]*17, "orig_mgmtindex", ["NaN"]+list(map(str, range(0, 18))))
visualization.barplot(data, "1p/19q", "1p/19q", "number of instances", "", ["sandybrown"]+["slategray"]*3, "orig_1p19q")
visualization.barplot(data, "IDH", "number of instances", "IDH", "", ["slategray"]*9, "orig_idh", horizontal=True)
visualization.barplot(data, "1-dead 0-alive", "survival at the last clinical follow up", "number of instances", "", visualization.alivedead_palette, "orig_alivedead")
visualization.histogram(data, "OS", "Sex", "OS", "probability of occurence in data", "", visualization.sex_palette, "orig_os")
visualization.histogram(data, "OS", None, "OS", "probability of occurence in data", "", "slategray", "orig_os2")
visualization.histogram(data, "OS", "WHO CNS Grade", "OS", "probability of occurence in data", "", visualization.grade_palette, "orig_os3")
visualization.histogram(data, "OS", "Final pathologic diagnosis (WHO 2021)", "OS", "probability of occurence in data", "", visualization.type_palette, "orig_os4")
visualization.histogram(data, "OS", "1-dead 0-alive", "OS", "probability of occurence in data", "", visualization.alivedead_palette, "orig_os5")
visualization.barplot(data, "EOR", "EOR", "number of instances", "", 3*["slategray"]+["sandybrown"], "orig_eor", ["NaN", "biopsy", "GTR", "STR"])
visualization.barplot(data, "Biopsy prior to imaging", "biopsy prior to imaging", "number of instances", "", 2*["slategray"], "orig_biopsy")

# 2) DTI/CSD VISUALIZATION ----------------------------------------------------
# prepare paths:
dwi_path = "data/preprocessed/dwi/UCSF-PDGM-0004_DWI.nii.gz"
bval_path = "data/UCSF-PDGM/UCSF-PDGM_DTI.bval"
bvec_path = "data/UCSF-PDGM/UCSF-PDGM_DTI.bvec"
roi_path_peritumoral = "data/preprocessed/roi/peritumoral/UCSF-PDGM-0004_ROI_peritumoral.nii.gz"
roi_path_periedemal = "data/preprocessed/roi/periedemal/UCSF-PDGM-0004_ROI_periedemal.nii.gz"
img_path = "data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_T1_bias.nii.gz"
tumor_path = "data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0004_nifti/UCSF-PDGM-0004_tumor_segmentation.nii.gz"
# DTI visualization (peritumoral):
dti_fit, _ = preprocessing.model_DTI(dwi_path, bval_path, bvec_path, roi_path_peritumoral)
visualization.dti_ellipsoids(dti_fit, 100, img_path, tumor_path)
# DTI visualization (periedemal):
dti_fit, _ = preprocessing.model_DTI(dwi_path, bval_path, bvec_path, roi_path_periedemal)
visualization.dti_ellipsoids(dti_fit, 100, img_path, tumor_path)
# CSD visualization (peritumoral):
csd_fit, _ = preprocessing.model_CSD(dwi_path, bval_path, bvec_path, roi_path_peritumoral)
visualization.csd_glyphs(csd_fit, 100, img_path, tumor_path)
visualization.fodf_sphere(csd_fit, roi_path_peritumoral, 10000)
# CSD visualization (periedemal):
csd_fit, _ = preprocessing.model_CSD(dwi_path, bval_path, bvec_path, roi_path_periedemal)
visualization.csd_glyphs(csd_fit, 100, img_path, tumor_path)

# 3) STRUCTURING ELEMENT ------------------------------------------------------
visualization.structuring_el(5, "sphere")
visualization.structuring_el(2, "cube")