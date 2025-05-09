# ==================================================================================================== #
# --- DIPLOMA THESIS (preprocess.py) ----------------------------------------------------------------- #
# ==================================================================================================== #
# title:         Analysis of White Matter Diffusion Properties in the Context of Selected Brain Tumors #
# author:        Bc. Martin KUKRÁL                                                                     #
# supervision:   doc. Ing. Roman MOUČEK, Ph.D.                                                         #
#                doc. MUDr. Irena HOLEČKOVÁ, Ph.D. (consultant)                                        #
# academic year: 2024/2025                                                                             #
# last updated:  2025-05-09                                                                            #
# ==================================================================================================== #
# Python version      3.11.4
import numpy as np  # 1.25.2
import pandas as pd # 2.1.0
import utils        # script
import time         # built-in
import os           # built-in
# initialize custom toolkit:
preprocessing = utils.PreprocessingToolkit()





# 1) PREPARE IDs --------------------------------------------------------------
repeated = ("UCSF-PDGM-315", "UCSF-PDGM-278", "UCSF-PDGM-175", "UCSF-PDGM-138", "UCSF-PDGM-181", "UCSF-PDGM-289")
data = pd.read_csv("./data/UCSF-PDGM/UCSF-PDGM-metadata_v2.csv").drop(columns=["BraTS21 ID", "BraTS21 Segmentation Cohort", "BraTS21 MGMT Cohort"])
data = data[~data["ID"].isin(repeated)] # data cleared of IDs corresponding to the duplicate subjects (follow-up imaging)

# 2) REGISTRATION -------------------------------------------------------------
start = time.time() # start time to measure the overall time
# loop through all subject IDs:
for subj in data["ID"]:
    id = subj[-3:] # use only the last three numbers (CSV uses three numbers to determine subject, files are named with four numbers)
    # prepare paths using the ID:
    moving_path = f"data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0{id}_nifti/UCSF-PDGM-0{id}_DTI_eddy_noreg.nii.gz"
    target_path = f"data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0{id}_nifti/UCSF-PDGM-0{id}_T2.nii.gz"
    brain_path = f"data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0{id}_nifti/UCSF-PDGM-0{id}_brain_segmentation.nii.gz"
    output_path = f"data/preprocessed/dwi/UCSF-PDGM-0{id}_DWI.nii.gz"
    # do the registration:
    preprocessing.registration_4Dto3D(moving_path, target_path, output_path, brain_path, flips=["S"], type="ElasticSyN")
end = time.time() # end time to measure overall time
print(f">>> TOTAL REGISTRATION TIME: {(end-start)/3600:.3f} h") # around 14.5 hours

# 3) ROI GENERATION -----------------------------------------------------------
start = time.time() # start time to measure the overall time
# loop through all subject IDs:
for subj in data["ID"]:
    id = subj[-3:] # use only the last three numbers (CSV uses three numbers to determine subject, files are named with four numbers)
    # prepare paths using the ID:
    tumor_path = f"data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0{id}_nifti/UCSF-PDGM-0{id}_tumor_segmentation.nii.gz"
    brain_path = f"data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0{id}_nifti/UCSF-PDGM-0{id}_brain_segmentation.nii.gz"
    parenchyma_path = f"data/UCSF-PDGM/PKG-UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/UCSF-PDGM-0{id}_nifti/UCSF-PDGM-0{id}_brain_parenchyma_segmentation.nii.gz"
    output_path_peritumoral = f"data/preprocessed/roi/peritumoral/UCSF-PDGM-0{id}_ROI_peritumoral.nii.gz"
    output_path_periedematous = f"data/preprocessed/roi/periedematous/UCSF-PDGM-0{id}_ROI_periedematous.nii.gz"
    # generate ROIs:
    preprocessing.generate_ROI(tumor_path, brain_path, "peritumoral", output_path_peritumoral)
    preprocessing.generate_ROI(tumor_path, brain_path, "periedematous", output_path_periedematous, parenchyma_path)
end = time.time() # end time to measure overall time
print(f">>> TOTAL ROI GENERATION TIME: {(end-start)/3600:.3f} h") # around 45 minutes

# 4) CSD AND DTI COMPUTATION --------------------------------------------------
# shared paths and variables:
bval_path = "data/UCSF-PDGM/UCSF-PDGM_DTI.bval" # file with b-values
bvec_path = "data/UCSF-PDGM/UCSF-PDGM_DTI.bvec" # file with gradient directions
csd_time = 0                                    # total CSD computation time
dti_time = 0                                    # total DTI computation time
# loop through all subject IDs:
for subj in data["ID"]:
    id = subj[-3:] # use only the last three numbers (CSV uses three numbers to determine subject, files are named with four numbers)
    # prepare paths using ID:
    dwi_path = f"data/preprocessed/dwi/UCSF-PDGM-0{id}_DWI.nii.gz"
    roi_path_peritumoral = f"data/preprocessed/roi/peritumoral/UCSF-PDGM-0{id}_ROI_peritumoral.nii.gz"
    roi_path_periedematous = f"data/preprocessed/roi/periedematous/UCSF-PDGM-0{id}_ROI_periedematous.nii.gz"
    output_path_csd_peritumoral = f"data/preprocessed/csd/peritumoral/UCSF-PDGM-0{id}_CSD_peritumoral.npy"
    output_path_csd_periedematous = f"data/preprocessed/csd/periedematous/UCSF-PDGM-0{id}_CSD_periedematous.npy"
    output_path_dti_peritumoral = f"data/preprocessed/dti/peritumoral/UCSF-PDGM-0{id}_DTI_peritumoral.npy"
    output_path_dti_periedematous = f"data/preprocessed/dti/periedematous/UCSF-PDGM-0{id}_DTI_periedematous.npy"
    # calculate CSD output for given ROIs:
    start = time.time()
    preprocessing.model_CSD(dwi_path, bval_path, bvec_path, roi_path_peritumoral, output_path_csd_peritumoral)
    preprocessing.model_CSD(dwi_path, bval_path, bvec_path, roi_path_periedematous, output_path_csd_periedematous)
    end = time.time()
    csd_time += (end - start)/3600
    # calculate DTI output for given ROIs:
    start = time.time()
    preprocessing.model_DTI(dwi_path, bval_path, bvec_path, roi_path_peritumoral, output_path_dti_peritumoral)
    preprocessing.model_DTI(dwi_path, bval_path, bvec_path, roi_path_periedematous, output_path_dti_periedematous)
    end = time.time()
    dti_time += (end - start)/3600
print(f">>> TOTAL CSD COMPUTATION TIME: {csd_time:.3f} h") # around 10 h
print(f">>> TOTAL DTI COMPUTATION TIME: {dti_time:.3f} h") # around 1.6 h

# 5) COMBINE EVERYTHING INTO A CSV FILE =======================================
# prepare the dataframes:
columns = ["ID", "Sex", "Age", "Grade", "Type", "MGMTstatus", "MGMTindex", "1p/19q", "IDH", "AliveDead", "OS", "EoR", "Biopsy", # from the original data
           "Ratio", "NE", "GFAmed", "GFAiqr", "MAGmed", "MAGiqr",                                                               # from the CSD model
           "FAmed", "FAiqr", "MDmed", "MDiqr", "RDmed", "RDiqr", "ADmed", "ADiqr"]                                              # from the DTI model
df_peritumoral = pd.DataFrame(columns=columns).set_index("ID")
df_periedematous = df_peritumoral.copy()
# correct the original data:
data["1p/19q"] = data["1p/19q"].replace("Co-deletion", "co-deletion")
# loop through all subject IDs:
for i, subj in enumerate(data["ID"]):
    id = subj[-3:] # use only the last three numbers (CSV uses three numbers to determine subject, files are named with four numbers)
    # prepare paths using the ID:
    peritumoral_csd_path = f"data/preprocessed/csd/peritumoral/UCSF-PDGM-0{id}_CSD_peritumoral.npy"
    periedematous_csd_path = f"data/preprocessed/csd/periedematous/UCSF-PDGM-0{id}_CSD_periedematous.npy"
    peritumoral_dti_path = f"data/preprocessed/dti/peritumoral/UCSF-PDGM-0{id}_DTI_peritumoral.npy"
    periedematous_dti_path = f"data/preprocessed/dti/periedematous/UCSF-PDGM-0{id}_DTI_periedematous.npy"
    # check if the file with CSD peritumoral results exits:
    if os.path.exists(peritumoral_csd_path):
        row_peritumoral_csd = np.load(peritumoral_csd_path)
    else: row_peritumoral_csd = [np.nan]*6
    # check if the file with CSD periedematous results exits:
    if os.path.exists(periedematous_csd_path):
        row_periedematous_csd = np.load(periedematous_csd_path)
    else: row_periedematous_csd = [np.nan]*6
    # check if the file with DTI peritumoral results exits:
    if os.path.exists(peritumoral_dti_path):
        row_peritumoral_dti = np.load(peritumoral_dti_path)
    else: row_peritumoral_dti = [np.nan]*8
    # check if the file with DTI periedematous results exits:
    if os.path.exists(periedematous_dti_path):
        row_periedematous_dti = np.load(periedematous_dti_path)
    else: row_periedematous_dti = [np.nan]*8
    # assemble full dataframe rows:
    row_peritumoral = {
        "ID": subj,
        "Sex": data.iloc[i, 1],
        "Age": data.iloc[i, 2],
        "Grade": data.iloc[i, 3],
        "Type": data.iloc[i, 4],
        "MGMTstatus": data.iloc[i, 5],
        "MGMTindex": data.iloc[i, 6],
        "1p/19q": data.iloc[i, 7],
        "IDH": data.iloc[i, 8],
        "AliveDead": data.iloc[i, 9],
        "OS": data.iloc[i, 10],
        "EoR": data.iloc[i, 11],
        "Biopsy": data.iloc[i, 12],
        "Ratio": row_peritumoral_csd[0],
        "NE": row_peritumoral_csd[1],
        "GFAmed": row_peritumoral_csd[2],
        "GFAiqr": row_peritumoral_csd[3],
        "MAGmed": row_peritumoral_csd[4],
        "MAGiqr": row_peritumoral_csd[5],
        "FAmed": row_peritumoral_dti[0],
        "FAiqr": row_peritumoral_dti[1],
        "MDmed": row_peritumoral_dti[2],
        "MDiqr": row_peritumoral_dti[3],
        "RDmed": row_peritumoral_dti[4],
        "RDiqr": row_peritumoral_dti[5],
        "ADmed": row_peritumoral_dti[6],
        "ADiqr": row_peritumoral_dti[7]}
    row_periedematous = {
        "ID": subj,
        "Sex": data.iloc[i, 1],
        "Age": data.iloc[i, 2],
        "Grade": data.iloc[i, 3],
        "Type": data.iloc[i, 4],
        "MGMTstatus": data.iloc[i, 5],
        "MGMTindex": data.iloc[i, 6],
        "1p/19q": data.iloc[i, 7],
        "IDH": data.iloc[i, 8],
        "AliveDead": data.iloc[i, 9],
        "OS": data.iloc[i, 10],
        "EoR": data.iloc[i, 11],
        "Biopsy": data.iloc[i, 12],
        "Ratio": row_periedematous_csd[0],
        "NE": row_periedematous_csd[1],
        "GFAmed": row_periedematous_csd[2],
        "GFAiqr": row_periedematous_csd[3],
        "MAGmed": row_periedematous_csd[4],
        "MAGiqr": row_periedematous_csd[5],
        "FAmed": row_periedematous_dti[0],
        "FAiqr": row_periedematous_dti[1],
        "MDmed": row_periedematous_dti[2],
        "MDiqr": row_periedematous_dti[3],
        "RDmed": row_periedematous_dti[4],
        "RDiqr": row_periedematous_dti[5],
        "ADmed": row_periedematous_dti[6],
        "ADiqr": row_periedematous_dti[7]}
    # append the rows to dataframes:
    df_peritumoral.loc[len(df_peritumoral)] = row_peritumoral
    df_periedematous.loc[len(df_periedematous)] = row_periedematous
# save the dataframes as CSV:
preprocessing.change_labels(df_peritumoral, "Type", ["A-IDHmut", "A-IDHwt", "G-IDHwt", "O-IDHmut"]).to_csv("data/preprocessed/peritumoral.csv")
preprocessing.change_labels(df_periedematous, "Type", ["A-IDHmut", "A-IDHwt", "G-IDHwt", "O-IDHmut"]).to_csv("data/preprocessed/periedematous.csv")