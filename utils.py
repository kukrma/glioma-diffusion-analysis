# ------------------------- #
# Python version   3.11.4   #
# ------------------------- #
# antspyx          0.4.2    #
# cliffs-delta     1.0.0    #
# dcor             0.6      #
# dipy             1.9.0    #
# matplotlib       3.7.2    #
# nibabel          5.2.1    #
# numpy            1.25.2   #
# optuna           4.2.1    #
# os               built-in #
# pandas           2.1.0    #
# scikit-image     0.22.0   #
# scikit-learn     1.3.0    #
# scipy            1.11.2   #
# seaborn          0.13.0   #
# time             built-in #
# tqdm             4.66.1   #
# umap-learn       0.5.7    #
# ------------------------- #

import nibabel as nib
import numpy as np
import pandas as pd
import ants
from tqdm import tqdm
import skimage
import os
import time
import seaborn as sns
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import umap
import dcor
import optuna
from cliffs_delta import cliffs_delta

import scipy.stats as stats
from scipy.spatial.distance import pdist
from scipy.interpolate import RBFInterpolator

from sklearn.manifold import trustworthiness
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.reconst.dti import TensorModel, color_fa
from dipy.reconst.shm import sh_to_sf
from dipy.data import get_sphere
from dipy.viz import actor
from dipy.viz import window





class PreprocessingToolkit():
    '''
    =============================================================================================================
    METHOD                  FUNCTION
    =============================================================================================================
    registration_4Dto3D() | Used to perform the registration of DWI data to the patient-specific space.
    generate_ROI()        | Used to generate ROIs around tumors and edemas.
    model_CSD()           | Used to compute CSD and selected derived characteristics.
    model_DTI()           | Used to compute DTI and selected derived characteristics.
    change_labels()       | Used to change labels inside columns of a dataframe (they can be too long for plots).
    =============================================================================================================
    '''
    def registration_4Dto3D(self, moving_path, target_path, output_path=None, brain_path=None, type="SyN", interpolation="bSpline", flips=None):
        '''
        --------------------------------------------------------------------------------------------------------------------
        PARAMETER       DTYPE           DESCRIPTION
        --------------------------------------------------------------------------------------------------------------------
        moving_path   | string        | path to the NIfTY file which will get registered
        target_path   | string        | path to the NIfTY file used as target for the registration
        output_path   | string / None | path to where the output NIfTY file will be saved
                      |               | (if set to None, the file will not be saved)
        brain_path    | string / None | path to the NIfTY file containing binary brain mask
                      |               | (if set to None, no masking will take place)
        type          | string        | type of registration (see ANTsPy docs)
        interpolation | string        | type of interpolation (see ANTsPy docs)
        flips         | string / None | array of flips executed one-by-one prior to registration can contain only:
                      |               | "S" = along sagittal plane, "F" = along frontal plane, "T" = along transversal plane
                      |               | (if set to None, no flips will be performed)
        --------------------------------------------------------------------------------------------------------------------
        RETURN >> output_nib - NIfTY image of the registered output
               >> transform  - transformations used during registration
        --------------------------------------------------------------------------------------------------------------------
        Used to perform the registration of DWI data to the patient-specific space to get overlap with tumor segmentation.
        '''
        # user info:
        print(">>> REGISTRATION 4D TO 3D")
        print(f"    moving: {os.path.basename(moving_path)}")
        print(f"    target: {os.path.basename(target_path)}")
        start = time.time()
        
        # 1) LOAD IMAGES ------------------------------------------------------
        print("    loading images...")
        moving_img = ants.image_read(moving_path)  # moving image for ANTsPy
        target_nib = nib.load(target_path)         # target image NIfTY
        target_img = ants.from_nibabel(target_nib) # target image for ANTsPy
        
        # 2) FLIP IMAGES (optional) -------------------------------------------
        if flips:
            print(f"    flipping images...")
            for flip in flips:
                if flip == "S":
                    # flip along sagittal plane:
                    moving_img = ants.from_numpy(np.flip(moving_img.numpy(), axis=0))
                if flip == "F":
                    # flip along frontal plane:
                    moving_img = ants.from_numpy(np.flip(moving_img.numpy(), axis=1))
                if flip == "T":
                    # flip along transversal plane:
                    moving_img = ants.from_numpy(np.flip(moving_img.numpy(), axis=2))

        # 3) REGISTRATION OF B0 -----------------------------------------------
        b0 = ants.from_numpy(moving_img[..., 0]) # first volume in DWI sequence
        # perform the registration:
        if brain_path:
            # use brain masking if specified:
            print(f"    masked registration of b0 ({type})...")
            brain_img = ants.image_read(brain_path)
            reg = ants.registration(fixed=target_img, moving=b0, type_of_transform=type, mask=brain_img)
        else:
            # with no brain mask:
            print(f"    registration of b0 ({type})...")
            reg = ants.registration(fixed=target_img, moving=b0, type_of_transform=type)
        # store the transformations:
        transform = reg["fwdtransforms"] # transforms
        
        # 4) APPLY THE REGISTRATION -------------------------------------------
        output = [] # will be filled with registered volumes
        # loop through all volumes:
        for i in tqdm(range(moving_img.shape[3]), bar_format="    transforming the volumes ({n_fmt}/{total_fmt})..."):
            # apply transform and store:
            volume = ants.apply_transforms(fixed=reg["warpedmovout"], moving=ants.from_numpy(moving_img[..., i]), transformlist=transform, interpolator=interpolation)
            output.append(volume.numpy())
        
        # 5) BRAIN MASK (optional) --------------------------------------------
        if brain_path:
            brain_img = brain_img.numpy() # binary brain mask image
            # apply the mask to all volumes:
            output = [volume*brain_img for volume in tqdm(output, bar_format="    masking brain ({n_fmt}/{total_fmt})...")]

        # 6) FINALIZE ---------------------------------------------------------
        # make NIfTY image:
        output_nib = nib.Nifti1Image(np.stack(output, axis=-1), target_nib.affine, target_nib.header, target_nib.extra)
        # optionally save the NIfTY file:
        if output_path:
            print("    saving output...")
            nib.save(output_nib, output_path)
        # print time:
        end = time.time()
        print(f"    time: {end-start:.3f} s")
        # return the NIfTY image and transformation:
        return output_nib, transform
    

    def generate_ROI(self, tumor_path, brain_path, type, output_path=None, parenchyma_path=None, radius=5, closing_size=2):
        '''
        ---------------------------------------------------------------------------------------------------
        PARAMETER         DTYPE           DESCRIPTION
        ---------------------------------------------------------------------------------------------------
        tumor_path      | string        | path to the NIfTY file containing tumor segmentation
        brain_path      | string        | path to the NIfTY file containing binary brain mask
        type            | string        | what ROI is to be generated, options:
                        |               | "peritumoral" = around tumor mass, "periedematous" = around edema
        output_path     | string / None | path to where the output NIfTY file with ROI will be saved
                        |               | (if set to None, the file will not be saved)
        parenchyma_path | string / None | path to the NIfTY file containing brain parenchyma
                        |               | (if set to None, parenchyma is not masked)
        radius          | int           | thickness (i.e. distance from tumor/edema) of the ROI
        closing_size    | int           | size of the cube used for parenchyma closing
        ---------------------------------------------------------------------------------------------------
        RETURN >> roi_nib - NIfTY image with ROI
        ---------------------------------------------------------------------------------------------------
        Used to generate ROIs around tumors and edemas.
        '''
        # user info:
        print(f">>> GENERATE ROI ({type})")
        print(f"    tumor: {os.path.basename(tumor_path)}")
        start = time.time()

        # 1) LOAD IMAGES ------------------------------------------------------
        print("    loading images...")
        tumor_nib = nib.load(tumor_path)              # tumor segmentation NIfTY
        brain_mask = nib.load(brain_path).get_fdata() # binary brain mask
        # preprocess the mask based on "type" parameter:
        if type == "peritumoral":
            # mask for peritumoral ROI:
            tumor_mask = np.where(np.isin(tumor_nib.get_fdata(), [1, 4]), 1, 0)
        elif type == "periedematous":
            # mask for periedematous ROI:
            tumor_mask = np.where(tumor_nib.get_fdata() == 0, 0, 1)
        
        # 2) MAKE THE ROI -----------------------------------------------------
        # dilatation:
        print("    performing dilation...")
        sphere = skimage.morphology.ball(radius)                          # structuring element
        mask_dil = skimage.morphology.binary_dilation(tumor_mask, sphere) # dilated mask
        # clipping:
        print("    clipping to ROI inside brain...")
        roi = mask_dil - tumor_mask                            # only ROI
        roi = np.logical_and(roi, brain_mask).astype(np.uint8) # only ROI inside brain
        # if generating peritumoral ROI:
        if type == "peritumoral":
            roi = np.logical_and(roi, np.where(tumor_nib.get_fdata() == 2, 1, 0)).astype(np.uint8) # only ROI inside brain and edema
        # incorporate parenchyma if specified:
        if parenchyma_path:
            print("    clipping to brain parenchyma...")
            parenchyma_mask = nib.load(parenchyma_path).get_fdata()                # binary mask of parenchyma
            parenchyma_roi = np.logical_and(roi, parenchyma_mask).astype(np.uint8) # clip parenchyma to ROI
            # closing to fill gaps in parenchyma:
            parenchyma_roi_closed = skimage.morphology.binary_closing(parenchyma_roi, skimage.morphology.cube(closing_size))
            roi = np.logical_and(roi, parenchyma_roi_closed).astype(np.uint8)      # clip ROI to closed parenchyma

        # 3) FINALIZE ---------------------------------------------------------
        # make NIfTY image:
        roi_nib = nib.Nifti1Image(roi, tumor_nib.affine, tumor_nib.header, tumor_nib.extra)
        # optionally save the NIfTY file:
        if output_path:
            print("    saving output...")
            nib.save(roi_nib, output_path)
        # print time:
        end = time.time()
        print(f"    time: {end-start:.3f} s")
        # return the NIfTY image:
        return roi_nib
        

    def model_CSD(self, dwi_path, bval_path, bvec_path, roi_path=None, output_path=None, fa_thresh=0.7):
        '''
        --------------------------------------------------------------------------------------
        PARAMETER     DTYPE           DESCRIPTION
        --------------------------------------------------------------------------------------
        dwi_path    | string        | path to the NIfTY file containing DWI
        bval_path   | string        | path to the file containing b-values
        bvec_path   | string        | path to the file containing gradient directions
        roi_path    | string / None | path to the NIfTY file containing ROI
                    |               | (if set to None, calculate CSD for the entire DWI)
        output_path | string / None | path (no file extension) to where the data will be saved
                    |               | (if set to None, the data will not be saved)
        fa_thresh   | float         | FA threshold when estimating response function
                    |               | (must be between 0 and 1)
        --------------------------------------------------------------------------------------
        RETURN >> csd_fit - fitted CSD model on the DWI data
                            (defaults to None if some problem)
               >> data    - array of computed diffusion properties (+ ratio)
                            (defaults to None if some problem)
        --------------------------------------------------------------------------------------
        Used to compute CSD and selected derived characteristics.
        '''
        # user info:
        print(f">>> COMPUTE CSD")
        print(f"    DWI: {os.path.basename(dwi_path)}")
        if roi_path: print(f"    ROI: {os.path.basename(roi_path)}")
        start = time.time()

        # 1) PREPARE DATA AND GRADIENT TABLE ----------------------------------
        dwi_nib = nib.load(dwi_path)  # DWI data as NIfTY image
        dwi_img = dwi_nib.get_fdata() # DWI data
        # prepare ROI if given:
        if roi_path:
            roi = nib.load(roi_path).get_fdata() # ROI
            unique = np.unique(roi)
            # test if ROI is not empty:
            if len(unique) == 1 and unique[0] == 0:
                print("    empty ROI")
                return None, None
        # build the gradient table:
        bval, bvec = read_bvals_bvecs(bval_path, bvec_path) # info about gradients
        gtab = gradient_table(bval, bvec)                   # gradient table

        # 2) CALCULATE CSD ----------------------------------------------------
        # try to compute the CSD (some brains might not have regions with FA > 0.7):
        try:
            # estimate the response function:
            response, ratio = auto_response_ssst(gtab, dwi_img, fa_thr=fa_thresh)
            print(f"    ratio: {ratio}")
            # fit the model using response function (only in ROI if given):
            sphere = get_sphere("symmetric362")
            csd_model = ConstrainedSphericalDeconvModel(gtab, response, convergence=100, reg_sphere=sphere, sh_order_max=8)
            csd_fit = csd_model.fit(dwi_img, mask=roi) if roi_path else csd_model.fit(dwi_img)

            # 3) CSD DIFFUSION PROPERTIES --------------------------------
            print(f"    computing diffusion properties...")
            roi_bool = roi.astype(np.bool_)                                                    # boolean mask to use when computing
            fodf_sphere = sh_to_sf(csd_fit.shm_coeff[roi_bool], sphere=sphere, sh_order_max=8) # sampled fODFs using sphere
            fodf_sphere[fodf_sphere < 0] = 0                                                   # remove negative values if any
            # normalized entropy (NE):
            fodf_flat = fodf_sphere[fodf_sphere > 0]
            hist, _ = np.histogram(fodf_flat, bins=128, density=True)
            ne = stats.entropy(hist + 1e-16) / np.log(128)
            # generalized fractional anisotropy (GFA):
            gfa = csd_fit.gfa[roi_bool]
            gfa_med = np.median(gfa)
            gfa_iqr = stats.iqr(gfa)
            # fODF magnitude:
            mag = np.sum(fodf_sphere, axis=1)
            mag_med = np.median(mag)
            mag_iqr = stats.iqr(mag)
            # print CSD diffusion properties:
            print(f"    NE:     {ne}")
            print(f"    GFAmed: {gfa_med}")
            print(f"    GFAiqr: {gfa_iqr}")
            print(f"    MAGmed: {mag_med}")
            print(f"    MAGiqr: {mag_iqr}")

            # 4) FINALIZE -----------------------------------------------------
            data = np.array([ratio, ne, gfa_med, gfa_iqr, mag_med, mag_iqr])
            # optionally save the CSD results:
            if output_path:
                # prepare and save the data:
                print("    saving outputs...")
                np.save(output_path, data)
            # print time:
            end = time.time()
            print(f"    time: {end-start:.3f} s")
            # return the CSD modelling output:
            return csd_fit, data
        # if some problem, inform the user and return empty values
        except Exception as e:
            print(e)
            return None, None
        

    def model_DTI(self, dwi_path, bval_path, bvec_path, roi_path=None, output_path=None):
        '''
        --------------------------------------------------------------------------------------
        PARAMETER     DTYPE           DESCRIPTION
        --------------------------------------------------------------------------------------
        dwi_path    | string        | path to the NIfTY file containing DWI
        bval_path   | string        | path to the file containing b-values
        bvec_path   | string        | path to the file containing gradient directions
        roi_path    | string / None | path to the NIfTY file containing ROI
                    |               | (if set to None, calculate CSD for the entire DWI)
        output_path | string / None | path (no file extension) to where the data will be saved
                    |               | (if set to None, the data will not be saved)
        --------------------------------------------------------------------------------------
        RETURN >> dti_fit - fitted DTI model on the DWI data
                            (defaults to None if some problem)
               >> data    - array of computed diffusion properties
                            (defaults to None if some problem)
        -------------------------------------------------------------------------------------
        Used to compute DTI and selected derived characteristics.
        '''
        # user info:
        print(f">>> COMPUTE DTI")
        print(f"    DWI: {os.path.basename(dwi_path)}")
        if roi_path: print(f"    ROI: {os.path.basename(roi_path)}")
        start = time.time()

        # 1) PREPARE DATA AND GRADIENT TABLE ----------------------------------
        dwi_nib = nib.load(dwi_path)  # DWI data as NIfTY image
        dwi_img = dwi_nib.get_fdata() # DWI data
        # prepare ROI if given:
        if roi_path:
            roi = nib.load(roi_path).get_fdata() # ROI
            unique = np.unique(roi)
            # test if ROI is not empty:
            if len(unique) == 1 and unique[0] == 0:
                print("    empty ROI")
                return None, None
        # build the gradient table:
        bval, bvec = read_bvals_bvecs(bval_path, bvec_path) # info about gradients
        gtab = gradient_table(bval, bvec)                   # gradient table

        # 2) CALCULATE DTI ----------------------------------------------------
        dti_model = TensorModel(gtab)
        dti_fit = dti_model.fit(dwi_img, mask=roi) if roi_path else dti_model.fit(dwi_img)

        # 3) DTI DIFFUSION PROPERTIES --------------------------------
        print(f"    computing diffusion properties...")
        roi_bool = roi.astype(np.bool_) # boolean mask to use when computing
        # fractional anisotropy (FA):
        fa = dti_fit.fa[roi_bool]
        fa_med = np.median(fa)
        fa_iqr = stats.iqr(fa)
        # mean diffusivity (MD):
        md = dti_fit.md[roi_bool]
        md_med = np.median(md)
        md_iqr = stats.iqr(md)
        # radial diffusivity (RD):
        rd = dti_fit.rd[roi_bool]
        rd_med = np.median(rd)
        rd_iqr = stats.iqr(rd)
        # axial diffusivity (AD):
        ad = dti_fit.ad[roi_bool]
        ad_med = np.median(ad)
        ad_iqr = stats.iqr(ad)
        # print DTI diffusion properties:
        print(f"    FAmed: {fa_med}")
        print(f"    FAiqr: {fa_iqr}")
        print(f"    MDmed: {md_med}")
        print(f"    MDiqr: {md_iqr}")
        print(f"    RDmed: {rd_med}")
        print(f"    RDiqr: {rd_iqr}")
        print(f"    ADmed: {ad_med}")
        print(f"    ADiqr: {ad_iqr}")

        # 4) FINALIZE -----------------------------------------------------
        data = np.array([fa_med, fa_iqr, md_med, md_iqr, rd_med, rd_iqr, ad_med, ad_iqr])
        # optionally save the DTI results:
        if output_path:
            # prepare and save the data:
            print("    saving outputs...")
            np.save(output_path, data)
        # print time:
        end = time.time()
        print(f"    time: {end-start:.3f} s")
        # return the CSD modelling output:
        return dti_fit, data
    

    def change_labels(self, df, column, colnames_new):
        '''
        -------------------------------------------------------------------------------------
        PARAMETER      DTYPE          DESCRIPTION
        -------------------------------------------------------------------------------------
        df           | pd.DataFrame | dataframe in which the labels will be changed
        column       | string       | name of the column in which the labels will be changed
        colnames_new | array-like   | array of new names
        -------------------------------------------------------------------------------------
        RETURN >> df - dataframe with modified labels
        -------------------------------------------------------------------------------------
        Used to change labels inside columns of a dataframe (they can be too long for plots).
        '''
        print(">>> CHANGE LABELS")
        print(f"    column:     {column}")
        print(f"    new values: {colnames_new}")
        unique = np.unique(df[column])            # array of unique labels (alphabetical)
        mapping = dict(zip(unique, colnames_new)) # mapping dictionary
        df[column] = df[column].map(mapping)      # change labels
        return df





class VisualizationToolkit():
    '''
    ======================================================================================
    METHOD             FUNCTION
    ======================================================================================
    __init__()       | Contains shared color palettes.
    explore_slices() | Used for simple 3D or 4D visualizations to inspect the spatial data
                     | from the Python script without the need to open FSL.
    barplot()        | Used to create a simple barplot (only counts per categories).
    histogram()      | Used to create a combined histogram with KDE.
    heatmap()        | Used to create a heatmap of a correlation matrix.
    violin()         | Used to create a violinplot with individual points.
    scatter()        | Used to create a scatterplot.
    dti_ellipsoids() | Used to visualize DTI ellipsoids.
    csd_glyphs()     | Used to visualize CSD glyphs.
    fodf_sphere()    | Used to visualize a single fODF.
    structuring_el() | Used to visualize the structuring element.
    ======================================================================================
    '''
    def __init__(self):
        self.sex_palette = ["cornflowerblue", "indianred"]                          # colors based on sex
        self.grade_palette = ["#910000", "#D55454", "#F0A4A4"]                      # colors based on grade
        self.type_palette = ["tomato", "dodgerblue", "yellowgreen", "mediumpurple"] # colors based on type
        self.alivedead_palette = ["palevioletred", "mediumseagreen"]                # colors based on survival


    def explore_slices(self, data):
        '''
        ---------------------------------------------------------------------------------------------
        PARAMETER   DTYPE      DESCRIPTION
        ---------------------------------------------------------------------------------------------
        data      | np.array | 3D or 4D data to be visualized
        ---------------------------------------------------------------------------------------------
        RETURN xxx
        ---------------------------------------------------------------------------------------------
        Used for simple 3D or 4D visualizations to inspect the spatial data from the Python scripts
        without the need to open FSL.
        '''
        print(">>> EXPLORE SLICES")
        # 1) PREPARE SUBPLOTS -----------------------------------------------------
        plt.style.use("dark_background") # easier to read
        # define subplot mosaic:
        fig, axs = plt.subplot_mosaic([["F", "T"],
                                       ["F", "T"],
                                       ["F", "T"],
                                       ["S", "T"],
                                       ["S", "T"],
                                       ["S", "sliders"],])
        # remove x and y ticks:
        for ax in axs.values():
            ax.set_xticks([])
            ax.set_yticks([])
        # frontal subplot:
        axs["F"].set_xlabel("sinister ⟷ dexter", color="deepskyblue")
        axs["F"].set_ylabel("inferior ⟷ superior", color="deepskyblue")
        axs["F"].text(5, 12, "F", fontsize=16, color="deepskyblue")
        for spine in axs["F"].spines.values():
            spine.set_edgecolor("deepskyblue")
        # saggital subplot:
        axs["S"].set_xlabel("anterior ⟷ posterior", color="lightcoral")
        axs["S"].set_ylabel("inferior ⟷ superior", color="lightcoral")
        axs["S"].text(5, 12, "S", fontsize=16, color="lightcoral")
        for spine in axs["S"].spines.values():
            spine.set_edgecolor("lightcoral")
        # transverse subplot:
        axs["T"].set_xlabel("sinister ⟷ dexter", color="palegreen")
        axs["T"].set_ylabel("posterior ⟷ anterior", color="palegreen")
        axs["T"].text(5, 10, "T", fontsize=16, color="palegreen")
        for spine in axs["T"].spines.values():
            spine.set_edgecolor("palegreen")
        # sliders subplots:
        axs["sliders"].axis("off")

        # 2) PLOTTING ---------------------------------------------------------
        # for 3D data:
        if len(data.shape) == 3:
            # pick cmap (normal vs. segmentation):
            cmap = ListedColormap(["black", "red", "yellow", "orange"]) if len(np.unique(data)) == 4 else "gray"
            # initial indices:
            idxF = data.shape[0]//2
            idxS = data.shape[1]//2
            idxT = data.shape[2]//2
            # plots:
            imgF = axs["F"].imshow(np.flip(np.rot90(data[:, idxF, :], k=1), axis=1), cmap=cmap, aspect="equal")
            imgS = axs["S"].imshow(np.rot90(data[idxS, :, :], k=1), cmap=cmap, aspect="equal")
            imgT = axs["T"].imshow(np.rot90(data[:, :, idxT], k=3), cmap=cmap, aspect="equal")
            # sliders:
            slidF_ax = fig.add_axes([0.55, 0.14, 0.315, 0.02])
            slidS_ax = fig.add_axes([0.55, 0.11, 0.315, 0.02])
            slidT_ax = fig.add_axes([0.55, 0.08, 0.315, 0.02])
            slidF = Slider(slidF_ax, "F", 0, data.shape[0]-1, valinit=idxF, valfmt="%d", facecolor="lightskyblue")
            slidS = Slider(slidS_ax, "S", 0, data.shape[1]-1, valinit=idxS, valfmt="%d", facecolor="lightcoral")
            slidT = Slider(slidT_ax, "T", 0, data.shape[2]-1, valinit=idxT, valfmt="%d", facecolor="palegreen")
            # slider updates:
            def updateF(val):
                idxF_ = slidF.val
                imgF.set_data(np.flip(np.rot90(data[:, int(idxF_), :], k=1), axis=1))
            def updateS(val):
                idxS_ = slidS.val
                imgS.set_data(np.rot90(data[int(idxS_), :, :], k=1))
            def updateT(val):
                idxT_ = slidT.val
                imgT.set_data(np.rot90(data[:, :, int(idxT_)], k=3))
            slidF.on_changed(updateF)
            slidS.on_changed(updateS)
            slidT.on_changed(updateT)
        # for 4D data:
        elif len(data.shape) == 4:
            # initial indices:
            idxF = data.shape[0]//2
            idxS = data.shape[1]//2
            idxT = data.shape[2]//2
            # plots:
            imgF = axs["F"].imshow(np.flip(np.rot90(data[:, idxF, :, 0], k=1), axis=1), cmap="gray", aspect="equal")
            imgS = axs["S"].imshow(np.rot90(data[idxS, :, :, 0], k=1), cmap="gray", aspect="equal")
            imgT = axs["T"].imshow(np.rot90(data[:, :, idxT, 0], k=3), cmap="gray", aspect="equal")
            # make sliders:
            slidF_ax = fig.add_axes([0.55, 0.14, 0.315, 0.02])
            slidS_ax = fig.add_axes([0.55, 0.11, 0.315, 0.02])
            slidT_ax = fig.add_axes([0.55, 0.08, 0.315, 0.02])
            slidV_ax = fig.add_axes([0.55, 0.05, 0.315, 0.02])
            slidF = Slider(slidF_ax, "F", 0, data.shape[0]-1, valinit=idxF, valfmt="%d", facecolor="lightskyblue")
            slidS = Slider(slidS_ax, "S", 0, data.shape[1]-1, valinit=idxS, valfmt="%d", facecolor="lightcoral")
            slidT = Slider(slidT_ax, "T", 0, data.shape[2]-1, valinit=idxT, valfmt="%d", facecolor="palegreen")
            slidV = Slider(slidV_ax, "V", 0, data.shape[3]-1, valinit=0, valfmt="%d", facecolor="wheat")
            # slider updates:
            def updateF(val):
                idxF_ = slidF.val
                idxV_ = slidV.val
                imgF.set_data(np.flip(np.rot90(data[:, int(idxF_), :, int(idxV_)], k=1), axis=1))
            def updateS(val):
                idxS_ = slidS.val
                idxV_ = slidV.val
                imgS.set_data(np.rot90(data[int(idxS_), :, :, int(idxV_)], k=1))
            def updateT(val):
                idxT_ = slidT.val
                idxV_ = slidV.val
                imgT.set_data(np.rot90(data[:, :, int(idxT_), int(idxV_)], k=3))
            def updateV(val):
                idxF_ = slidF.val
                idxS_ = slidS.val
                idxT_ = slidT.val
                idxV_ = slidV.val
                imgF.set_data(np.flip(np.rot90(data[:, int(idxF_), :, int(idxV_)], k=1), axis=1))
                imgS.set_data(np.rot90(data[int(idxS_), :, :, int(idxV_)], k=1))
                imgT.set_data(np.rot90(data[:, :, int(idxT_), int(idxV_)], k=3))
            slidF.on_changed(updateF)
            slidS.on_changed(updateS)
            slidT.on_changed(updateT)
            slidV.on_changed(updateV)

        # 3) FINALIZE PLOT ----------------------------------------------------
        plt.tight_layout(w_pad=-8)
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()


    def barplot(self, df, column, xlabel="", ylabel="", title="", palette="Set1", saveas=None, order=None, horizontal=False):
        '''
        -----------------------------------------------------------------------------------
        PARAMETER    DTYPE            DESCRIPTION
        -----------------------------------------------------------------------------------
        df         | pd.DataFrame   | target dataframe
        column     | string         | name of column to be plotted
        xlabel     | string         | label of the x-axis
        ylabel     | string         | label of the y-axis
        title      | string         | title of the plot
        palette    | string / array | palette name (see seaborn docs) or an array of colors
        saveas     | string / None  | name of the file to be saved as
                   |                | (if None, the plot is shown instead of saved)
        order      | array / None   | order of bars in the plot
                   |                | (if None, the categories are sorted and used)
        horizontal | bool           | if True, plot the bars horizontally
        -----------------------------------------------------------------------------------
        RETURN xxx
        -----------------------------------------------------------------------------------
        Used to create a simple barplot (only counts per categories).
        '''
        print(f">>> BARPLOT ({column})")
        unique = df[column].unique()                    # unique values
        counts = df[column].value_counts().loc[unique]  # counts
        order = order if order else sorted(unique)      # sorted for bar ordering
        # make the plot:
        if horizontal:
            ax = sns.barplot(x=counts, y=unique, hue=unique, palette=palette, legend=False, order=order)
            # add count labels on top of bars
            for bar, count in zip(ax.patches, counts):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, str(count), ha="left", va="center")
        else:
            ax = sns.barplot(x=unique, y=counts, hue=unique, palette=palette, legend=False, order=order)
            # add count labels on top of bars
            for bar, count in zip(ax.patches, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        # print info about number of classes:
        print(df[column].value_counts(dropna=False), "\n")
        # save if name was given:
        if saveas:
            plt.savefig(f"./img/{saveas}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
            print(f"    plot saved as {saveas}.pdf")
        # otherwise just show the plot:
        else:
            plt.show()


    def histogram(self, df, column, hue=None, xlabel="", ylabel="", title="", palette="Set1", saveas=None):
        '''
        -----------------------------------------------------------------------------------
        PARAMETER    DTYPE            DESCRIPTION
        -----------------------------------------------------------------------------------
        df         | pd.DataFrame   | target dataframe
        column     | string         | name of column to be plotted
        hue        | string / None  | name of column to be used as categories
                   |                | (if None, the data is plot as a single category)
        xlabel     | string         | label of the x-axis
        ylabel     | string         | label of the y-axis
        title      | string         | title of the plot
        palette    | string / array | palette name (see seaborn docs) or an array of colors
        saveas     | string / None  | name of the file to be saved as
                   |                | (if None, the plot is shown instead of saved)
        -----------------------------------------------------------------------------------
        RETURN xxx
        -----------------------------------------------------------------------------------
        Used to create a combined histogram with KDE.
        '''
        print(f">>> HISTOGRAM ({column})")
        # make the plot:
        if hue:
            sns.histplot(data=df, x=column, hue=hue, edgecolor="white", palette=palette, alpha=0.2, kde=True, stat="probability", common_norm=False, line_kws={"linewidth": 2})
        else:
            sns.histplot(data=df, x=column, edgecolor="white", color=palette, alpha=0.2, kde=True, stat="probability", common_norm=False, line_kws={"linewidth": 2})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        # print info:
        vals = df.dropna(subset=column)[column]
        print(f"    mean:   {np.mean(vals)}")
        print(f"    std:    {np.std(vals)}")
        print(f"    median: {np.median(vals)}")
        print(f"    iqr:    {stats.iqr(vals)}")
        print(f"    min:    {np.min(vals)}")
        print(f"    max:    {np.max(vals)}")
        # save if name was given:
        if saveas:
            plt.savefig(f"./img/{saveas}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
            print(f"    plot saved as {saveas}.pdf")
        # otherwise just show the plot:
        else:
            plt.show()


    def heatmap(self, corr_matrix, colnanmes, saveas=None):
        '''
        ---------------------------------------------------------------------------
        PARAMETER     DTYPE           DESCRIPTION
        ---------------------------------------------------------------------------
        corr_matrix | array         | correlation matrix
        colnanmes   | array         | names of the columns
        saveas      | string / None | name of the file to be saved as
                    |               | (if None, the plot is shown instead of saved)
        ---------------------------------------------------------------------------
        RETURN xxx
        ---------------------------------------------------------------------------
        Used to create a heatmap of a correlation matrix.
        '''
        print(">>> HEATMAP")
        # determine color scheme:
        if np.min(corr_matrix) < 0:
            # for correlations in range from -1 to 1:
            cmap = "RdYlBu"
            vmin = -1
            vmax = 1
            center = 0
        else:
            # for correlations in range from 0 to 1:
            cmap = "Blues"
            vmin = 0
            vmax = 1
            center = None
        # heatmap itself:
        sns.heatmap(corr_matrix, cmap=cmap, annot=False, center=center, square=True, vmin=vmin, vmax=vmax, mask=np.triu(corr_matrix, k=1), linewidths=0.5, linecolor="white", xticklabels=colnanmes, yticklabels=colnanmes)
        # save if name was given:
        if saveas:
            plt.savefig(f"./img/{saveas}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
            print(f"    plot saved as {saveas}.pdf")
        # otherwise just show the plot:
        else:
            plt.show()


    def violin(self, df, x, y, hue=None, palette="Set1", saveas=None):
        '''
        -----------------------------------------------------------------------------------
        PARAMETER    DTYPE           DESCRIPTION
        -----------------------------------------------------------------------------------
        df        | pd.DataFrame   | target dataframe
        x         | string         | name of column to used as main categories
        y         | string         | name of column to be used as values
        hue       | string         | name of column to used as secondary categories
                  |                | (if None, there is one violin per main category)
        palette   | string / array | palette name (see seaborn docs) or an array of colors
        saveas    | string / None  | name of the file to be saved as
                  |                | (if None, the plot is shown instead of saved)
        -----------------------------------------------------------------------------------
        RETURN xxx
        -----------------------------------------------------------------------------------
        Used to create a violinplot with individual points.
        '''
        print(">>> VIOLINPLOT")
        # make the plot:
        sns.stripplot(data=df, x=x, y=y, hue=hue, dodge=True, marker=".", palette=palette, jitter=0.25, alpha=0.3, zorder=1, legend=False)
        sns.violinplot(data=df, x=x, y=y, hue=hue, split=False, density_norm="width", palette=palette, fill=False, gap=0.05, zorder=2)
        # save if name was given:
        if saveas:
            plt.savefig(f"./img/{saveas}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
            print(f"    plot saved as {saveas}.pdf")
        # otherwise just show the plot:
        else:
            plt.show()


    def scatter(self, df, x, y, hue=None, palette="Set1", saveas=None):
        '''
        -----------------------------------------------------------------------------------
        PARAMETER    DTYPE           DESCRIPTION
        -----------------------------------------------------------------------------------
        df        | pd.DataFrame   | target dataframe
        x         | string         | name of column to be used on x-axis
        y         | string         | name of column to be used on y-axis
        hue       | string         | name of column to used as categories for colors
                  |                | (if None, all points will have the same color)
        palette   | string / array | palette name (see seaborn docs) or an array of colors
        saveas    | string / None  | name of the file to be saved as
                  |                | (if None, the plot is shown instead of saved)
        -----------------------------------------------------------------------------------
        RETURN xxx
        -----------------------------------------------------------------------------------
        Used to create a scatterplot.
        '''
        print(">>> SCATTERPLOT")
        # make plot:
        legend = True if hue else False
        hue = hue if hue else ["instance"]*len(df[x])
        sns.scatterplot(df, x=x, y=y, hue=hue, palette=palette, alpha=0.5, legend=legend)
        if legend:
            plt.legend(loc="upper right")
        # save if name was given:
        if saveas:
            plt.savefig(f"./img/{saveas}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
            print(f"    plot saved as {saveas}.pdf")
        # otherwise just show the plot:
        else:
            plt.show()


    def dti_ellipsoids(self, dti_fit, idx, img_path=None, tumor_path=None):
        '''
        ----------------------------------------------------------
        PARAMETER    DTYPE            DESCRIPTION
        ----------------------------------------------------------
        dti_fit    | dipy.TensorFit | fitted DTI model
        idx        | int            | index of the axial segment 
        img_path   | string         | path to the background image
        tumor_path | string         | path to the segmented tumor
        ----------------------------------------------------------
        RETURN xxx
        ----------------------------------------------------------
        Used to visualize DTI ellipsoids.
        '''
        print(">>> DTI ELLIPSOIDS VISUALIZATION")
        # prepare coloring based on direction and FA:
        RGB = color_fa(np.clip(dti_fit.fa, 0, 1), dti_fit.evecs)
        cfa = RGB[:, :, idx:idx+1]
        cfa /= cfa.max()
        # make the 3D scene:
        scene = window.Scene()
        # add background image if given:
        if img_path:
            img = nib.load(img_path).get_fdata()[:, :, idx:idx+1]
            img_slice = actor.slicer(img, opacity=0.5)
            scene.add(img_slice)
        # add tumor segmentation if given:
        if tumor_path:
            tumor = nib.load(tumor_path).get_fdata()[:, :, idx:idx+1]
            tumor_slice = actor.slicer(tumor, opacity=1)
            scene.add(tumor_slice)
        # add the DTI ellipsoids:
        sphere = get_sphere("symmetric362")
        tensors = actor.tensor_slicer(dti_fit.evals[:, :, idx:idx+1], dti_fit.evecs[:, :, idx:idx+1], scalar_colors=cfa, sphere=sphere, scale=0.35)
        scene.add(tensors)
        # finish and show the scene:
        window.show(scene)
    

    def csd_glyphs(self, csd_fit, idx, img_path=None, tumor_path=None):
        '''
        -----------------------------------------------------------
        PARAMETER    DTYPE             DESCRIPTION
        -----------------------------------------------------------
        csd_fit    | dipy.SphHarmFit | fitted CSD model
        idx        | int             | index of the axial segment 
        img_path   | string          | path to the background image
        tumor_path | string          | path to the segmented tumor
        -----------------------------------------------------------
        RETURN xxx
        -----------------------------------------------------------
        Used to visualize CSD glyphs.
        '''
        print(">>> CSD GLYPHS VISUALIZATION")
        # make the 3D scene:
        scene = window.Scene()
        # add background image if given:
        if img_path:
            img = nib.load(img_path).get_fdata()[:, :, idx:idx+1]
            img_slice = actor.slicer(img, opacity=0.5)
            scene.add(img_slice)
        # add tumor segmentation if given:
        if tumor_path:
            tumor = nib.load(tumor_path).get_fdata()[:, :, idx:idx+1]
            tumor_slice = actor.slicer(tumor, opacity=1)
            scene.add(tumor_slice)
        # add the CSD glyphs:
        sphere = get_sphere("symmetric362")
        odf = sh_to_sf(csd_fit.shm_coeff[:, :, idx:idx+1], sphere, sh_order_max=8)
        glyphs = actor.odf_slicer(odf, sphere=sphere, scale=0.75)
        scene.add(glyphs)
        # finish and show the scene:
        window.show(scene)


    def fodf_sphere(self, csd_fit, roi_path, number):
        '''
        --------------------------------------------------------
        PARAMETER   DTYPE             DESCRIPTION
        --------------------------------------------------------
        csd_fit   | dipy.SphHarmFit | fitted CSD model
        roi_path  | string          | path to the ROI NIfTY file 
        number    | int             | fODF number
        --------------------------------------------------------
        RETURN xxx
        --------------------------------------------------------
        Used to visualize a single fODF.
        '''
        print(f">>> fODF VISUALIZATION ({number})")
        # prepare everything:
        roi = nib.load(roi_path).get_fdata().astype(np.bool_)                         # ROI to get only relevant fODFs
        sphere = get_sphere("symmetric362")                                           # sphere to sample
        fodf_sphere = sh_to_sf(csd_fit.shm_coeff[roi], sphere=sphere, sh_order_max=8) # fODFs
        fodf_sphere[fodf_sphere < 0] = 0                                              # fODFs with corrected 0
        values = fodf_sphere[number]                                                  # one specific fODF (i.e. one voxel)
        # create spherical grid:
        grid_size = 50
        theta_grid, phi_grid = np.meshgrid(np.linspace(0, np.pi, grid_size), np.linspace(0, 2*np.pi, grid_size))
        # convert to Cartesian coordinates:
        x_grid = np.sin(theta_grid) * np.cos(phi_grid)
        y_grid = np.sin(theta_grid) * np.sin(phi_grid)
        z_grid = np.cos(theta_grid)
        # flatten coordinates:
        grid_points = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T
        data_points = np.vstack((sphere.x, sphere.y, sphere.z)).T
        # interpolate:
        rbf = RBFInterpolator(data_points, values, kernel="linear")
        interpolated_values = rbf(grid_points).reshape(grid_size, grid_size)
        # surface with interpolated values:
        ax = plt.axes(projection="3d")
        ax.plot_surface(x_grid, y_grid, z_grid, facecolors=plt.cm.YlOrRd(interpolated_values), rstride=1, cstride=1, alpha=0.9)
        # plot the fODF surface:
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
        # plot only the sampled fODF points:
        ax = plt.axes(projection="3d")
        ax.scatter3D(sphere.x, sphere.y, sphere.z, c=values, cmap="YlOrRd")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()


    def structuring_el(self, size, type):
        '''
        ------------------------------------------
        PARAMETER   DTYPE     DESCRIPTION
        ------------------------------------------
        size      | integer | size of the object
        type      | string  | "sphere" or "cube"
        ------------------------------------------
        RETURN xxx
        ------------------------------------------
        Used to visualize the structuring element.
        '''
        # if the structuring element is a sphere:
        if type == "sphere":
            # make the sphere:
            sphere = skimage.morphology.ball(size).astype(np.bool_)
            # prepare colors of the sphere:
            colors = np.empty(sphere.shape, dtype=object)
            colors[sphere] = "silver"              # voxels
            colors[size, size, size] = "red" # anchor
            # create the 3D visualization:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(sphere, facecolors=colors, edgecolor="gray")
            plt.axis("equal")
            plt.show()
            # create the slice in 3D:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(sphere[:, :, :size+1], facecolors=colors[:, :, :size+1], edgecolor="gray")
            plt.axis("equal")
            plt.show()
        # if the structuring element is a cube:
        elif type == "cube":
            # make the cube:
            sphere = skimage.morphology.cube(size).astype(np.bool_)
            # prepare colors of the sphere:
            colors = np.empty(sphere.shape, dtype=object)
            colors[sphere] = "silver" # voxels
            # even cube:
            if size % 2 == 0:
                colors[0, 0, 1] = "red" # anchor
            # odd cube:
            else:
                colors[size, size, size] = "red" # anchor
            # create the 3D visualization:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(sphere, facecolors=colors, edgecolor="gray")
            plt.axis("equal")
            plt.show()





class AnalysisToolkit():
    '''
    ============================================================================================================
    METHOD                    FUNCTION
    ============================================================================================================
    correlation()           | Used to compute linear or non-linear correlation and show it using a heatmap.
    umap_manifold()         | Used to compute UMAP on a range of data (UMAP parameters can be optimized).
    gmm_clustering()        | Used to compute GMM on a range of data (number of clusters can be optimized).
    relation_quantitative() | Used to explore the relationship between a quantitative and qualitative attribute.
    relation_qualitative()  | Used to explore the relationship between two qualitative attributes.
    ============================================================================================================
    '''
    def correlation(self, df, col_indices, nlin=True, saveas=None):
        '''
        --------------------------------------------------------------------------------------
        PARAMETER     DTYPE           DESCRIPTION
        --------------------------------------------------------------------------------------
        df          | pd.DataFrame  | target dataframe
        col_indices | array         | array of indices from which correlation will be computed
        nlin        | bool          | True: use distance correlation, False: use Spearman
        saveas      | string / None | name of the file to be saved as
                    |               | (if None, the plot is shown instead of saved)
        --------------------------------------------------------------------------------------
        RETURN xxx
        --------------------------------------------------------------------------------------
        Used to compute linear or non-linear correlation and show it using a heatmap.
        '''
        print(">>> CORRELATION ANALYSIS")
        data = df.iloc[:, col_indices].dropna() # drop NaN from data
        vals = data.values                      # get the values
        # variant using distance correlation:
        if nlin:
            print("    using distance correlation...")
            features = len(col_indices)
            corr_matrix = np.zeros((features, features))
            # fill the correlation matrix:
            for i in range(features):
                for j in range(features):
                    corr_matrix[i, j] = dcor.distance_correlation(vals[:, i], vals[:, j])
        # variant using Spearman's correlation coefficient:
        else:
            print("    using Spearman correlation coefficient...")
            corr_matrix = data.corr(method="spearman").values
        # visualize the heatmap:
        viz = VisualizationToolkit()
        viz.heatmap(corr_matrix, data.columns, saveas)


    def umap_manifold(self, df, col_indices, n_trials=100, params=None, plots_name=None):
        '''
        -----------------------------------------------------------------------------------------------------
        PARAMETER     DTYPE           DESCRIPTION
        -----------------------------------------------------------------------------------------------------
        df          | pd.DataFrame  | target dataframe
        col_indices | array         | array of indices to which UMAP will be applied
        n_trials    | int           | number of generations when optimizing UMAP
        params      | array / None  | contains parameters for UMAP if already known, which skips optimization
                    |               | (if None, UMAP is optimized using optuna)
        plots_name  | string / None | text used in the names of files, implies that plots will be made
                    |               | (if None, no plots are made)
        -----------------------------------------------------------------------------------------------------
        RETURN >> df - input dataframe with appended 2D embeddings from UMAP
        -----------------------------------------------------------------------------------------------------
        Used to compute UMAP on a range of data (UMAP parameters can be optimized).
        '''
        print(">>> UMAP MANIFOLD")
        # drop rows with at least one empty diffusion property:
        df = df.dropna(subset=df.columns[col_indices], how="any")
        # standardize the diffusion properties:
        print("    scaling data...")
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.iloc[:, col_indices].values)
        # if no parameters are given, start the parameter optimization:
        if not params:
            print("    optimizing parameters using optuna...")
            start = time.time() # start time
            # objective function for the optimization:
            def objective(trial):
                # optimized variables:
                metric = trial.suggest_categorical("metric", ["euclidean", "braycurtis", "cosine", "correlation"])
                n_neighbors = trial.suggest_int("n_neigh", 2, 100)
                min_dist = trial.suggest_float("min_dist", 0.1, 0.99)
                # use UMAP for a possible solution:
                reducer = umap.UMAP(random_state=0, n_epochs=1000, n_jobs=1, metric=metric, n_neighbors=n_neighbors, min_dist=min_dist)
                embedding = reducer.fit_transform(data_scaled)
                # compute trustworthiness:
                trust = trustworthiness(data_scaled, embedding, n_neighbors=n_neighbors, metric=metric)
                # compute stress:
                d_high = pdist(data_scaled, metric=metric)
                d_low = pdist(embedding, metric=metric)
                stress = np.sqrt(np.sum((d_high - d_low) ** 2) / np.sum(d_high**2))
                # return results of a possible solution:
                return 1-trust, stress
            # create an optuna study and optimize it:
            study = optuna.create_study(directions=["minimize", "minimize"], study_name="UMAP.study", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(objective, n_trials=n_trials)
            # retrieve the best trial:
            best_trial = min(study.best_trials, key=lambda trial: sum(trial.values))
            print(">>> OPTIMIZATION RESULTS")
            print(f"    best parameters: {best_trial.params}")
            print(f"    1-T: {best_trial.values[0]}")
            print(f"    S:   {best_trial.values[1]}")
            # retrieve the best parameters:
            params = [best_trial.params["metric"], best_trial.params["n_neigh"], best_trial.params["min_dist"]]
            end = time.time() # end time
            print(f">>> OPTIMIZATION TIME: {end - start}")
            # save plots if desired:
            if plots_name:
                print("    making and saving the plots...")
                # retrieve results of both criterion:
                history = np.array([[t.values[0], t.values[1]] for t in study.trials])
                # optimization history:
                history_sum = np.sum(history, axis=1)
                best = np.argmin(history_sum)
                plt.plot(history_sum, color="mediumseagreen", zorder=1, label="optimization criterion")
                plt.scatter(best, history_sum[best], color="orangered", zorder=2, label=f"best trial ({best})")
                plt.xlabel("trials")
                plt.ylabel("(1 - trustworthiness) + stress")
                plt.legend(loc="upper right")
                plt.savefig(f"./img/optuna_history_{plots_name}.pdf", format="pdf", bbox_inches="tight")
                plt.close()
                # pareto front:
                plt.scatter(history[:, 0], history[:, 1], color="mediumseagreen", label="trials", alpha=0.5)
                plt.scatter(history[best, 0], history[best, 1], color="orangered", label=f"best trial ({best})")
                plt.xlabel("1 - trustworthiness")
                plt.ylabel("stress")
                plt.xlim(right=0.15)
                plt.ylim(top=1)
                plt.legend(loc="upper right")
                plt.savefig(f"./img/optuna_pareto_{plots_name}.pdf", format="pdf", bbox_inches="tight")
                plt.close()
        # compute UMAP using given or computed parameters:
        print("    computing final UMAP...")
        reducer = umap.UMAP(random_state=0, n_epochs=1000, n_jobs=1, metric=params[0], n_neighbors=params[1], min_dist=params[2])
        embedding = reducer.fit_transform(data_scaled)
        # append embeddings to the dataframe:
        df["Embedding1"] = embedding[:, 0]
        df["Embedding2"] = embedding[:, 1]
        # return extended dataframe:
        return df
    

    def gmm_clustering(self, df, col_indices, n_clusters, plots_name=None, metric="davies-bouldin"):
        '''
        ------------------------------------------------------------------------------------------------------
        PARAMETER     DTYPE           DESCRIPTION
        ------------------------------------------------------------------------------------------------------
        df          | pd.DataFrame  | target dataframe
        col_indices | array         | array of indices containing UMAP embeddings
        n_clusters  | int / list    | number of clusters to be found, or a range to look for optimal number
                    |               | (int - directly compute GMM for given number of clusters)
                    |               | (list - range of [low, high] determining the range of possible clusters)
        plots_name  | string / None | text used in the names of files, implies that plots will be made
                    |               | (if None, no plots are made)
        metric      | string        | metric used to determine the optimal number of clusters
                    |               | ("davies-bouldin", "calinski-harabasz", "silhouette")
        ------------------------------------------------------------------------------------------------------
        RETURN >> df - input dataframe with appended clustering classes
        ------------------------------------------------------------------------------------------------------
        Used to compute GMM on a range of data (number of clusters can be optimized).
        '''
        print(">>> GMM CLUSTERING")
        # take only the UMAp embeddings:
        data = df.values[:, col_indices]
        # if we search for the optimal number of clusters:
        if isinstance(n_clusters, list):
            # prepare arrays:
            rnge = range(n_clusters[0], n_clusters[1]+1) # tested number of clusters
            crit = []                                    # cluster quality metric score
            lls = []                                     # log-likelihood
            # determine which metric will be used:
            if metric == "davies-bouldin":
                func = davies_bouldin_score
                picker = np.argmin
            elif metric == "calinski-harabasz":
                func = calinski_harabasz_score
                picker = np.argmax
            elif metric == "silhouette":
                func = silhouette_score
                picker = np.argmax
            # test the given range to find optimal number of clusters using the selected metric:
            for n in tqdm(rnge, bar_format="    testing the range of clusters ({n_fmt}/{total_fmt})..."):
                # define the GMM and apply it to data:
                gmm = GaussianMixture(n_components=n, random_state=0, n_init=10)
                gmm.fit(data)
                pred = gmm.predict(data)
                # append metrics to arrays:
                crit.append(func(data, pred))
                lls.append(gmm.score(data))
            # retrieve the optimal number of clusters:
            idx = picker(crit)
            best = rnge[idx]
            print(f"    {best} number of clusters optimal")
            print(f"    ({metric} = {crit[idx]})")
            n_clusters = best
            # generate plots if desired:
            if plots_name:
                # log-likelihood plot:
                plt.plot(rnge, lls, "o-", color="forestgreen", label="score")
                plt.xlabel("number of clusters")
                plt.ylabel("log-likelihood")
                plt.legend()
                plt.savefig(f"./img/gmm_loglikelihoods_{plots_name}.pdf", format="pdf", bbox_inches="tight")
                plt.close()
                # cluster quality metric plot:
                plt.plot(rnge, crit, "o-", color="mediumpurple", zorder=1, label="index")
                plt.scatter(n_clusters, crit[idx], color="orangered", zorder=2, label="best number of clusters")
                plt.xlabel("number of clusters")
                plt.ylabel(metric)
                plt.legend()
                plt.savefig(f"./img/gmm_{plots_name}.pdf", format="pdf", bbox_inches="tight")
                plt.close()
        # use the best number of clusters to compute GMM and append the clustering classes to the dataframe:
        print("    computing final GMM...")
        gmm = GaussianMixture(n_components=n_clusters, random_state=0, n_init=10)
        gmm.fit(data)
        clusters = gmm.predict(data)
        df["Cluster"] = clusters + 1
        # if desired, plot the GMM components:
        if plots_name:
            # prepare the grid to compute values:
            min = np.min(data, axis=0)
            max = np.max(data, axis=0)
            x_grid, y_grid = np.meshgrid(np.linspace(min[0]-0.5, max[0]+0.5, 500), np.linspace(min[1]-0.5, max[1]+0.5, 500))
            grid = np.array([x_grid.ravel(), y_grid.ravel()]).T
            # add data points:
            plt.scatter(data[:, 0], data[:, 1], color="silver", marker=".", zorder=1, alpha=0.5)
            # plot individual GMM components:
            compo = 1
            for mean, cov, col in zip(gmm.means_, gmm.covariances_, sns.color_palette("tab10", n_clusters)):
                pdf = stats.multivariate_normal.pdf(grid, mean=mean, cov=cov).reshape(x_grid.shape)  # No weight needed
                cmap = LinearSegmentedColormap.from_list("", [col + (0.2,), col + (1,)], 256)
                plt.scatter(mean[0], mean[1], color=col, zorder=3, label=compo)
                plt.contour(x_grid, y_grid, pdf, levels=10, cmap=cmap, zorder=2)  # Unique color for each component
                compo += 1
            # save the plot:
            plt.xlabel("Embedding1")
            plt.ylabel("Embedding2")
            plt.legend(loc="upper right")
            plt.savefig(f"./img/gmm_components_{plots_name}.pdf", format="pdf", bbox_inches="tight")
            plt.close()
        # return extended dataframe:
        return df
    

    def relation_quantitative(self, df, atr_quantitative, atr_qualitative):
        '''
        ----------------------------------------------------------------------------------
        PARAMETER          DTYPE          DESCRIPTION
        ----------------------------------------------------------------------------------
        df               | pd.DataFrame | target dataframe
        atr_quantitative | string       | name of the quantitative attribute
        atr_qualitative  | string       | name of the qualitative attribute
        ----------------------------------------------------------------------------------
        RETURN xxx
        ----------------------------------------------------------------------------------
        Used to explore the relationship between a quantitative and qualitative attribute.
        '''
        print(">>> POST-CLUSTERING ANALYSIS (QUANTITATIVE)")
        print(f"    {atr_quantitative} <-> {atr_qualitative}")
        # prepare the data:
        df = df.dropna(subset=[atr_quantitative, atr_qualitative], how="any")
        unique = np.unique(df[atr_qualitative])
        cluster1 = df[df[atr_qualitative]==unique[0]][atr_quantitative]
        cluster2 = df[df[atr_qualitative]==unique[1]][atr_quantitative]
        # Kolmogorov-Smirnov test:
        stat, p_value = stats.ks_2samp(cluster1, cluster2)
        print("    Kolmogorov-Smirnov test")
        print(f"    test stat: {stat}")
        print(f"    p-value:   {p_value}\n")
        # Mann-Whitney U test:
        stat, p_value = stats.mannwhitneyu(cluster1, cluster2)
        print("    Mann-Whitney U test")
        print(f"    test stat: {stat}")
        print(f"    p-value:   {p_value}\n")
        # Levene test:
        stat, p_value = stats.levene(cluster1, cluster2, center="mean")
        print("    Levene test")
        print(f"    test stat: {stat}")
        print(f"    p-value:   {p_value}\n")
        # Brown-Forsythe test:
        stat, p_value = stats.levene(cluster1, cluster2, center="median")
        print("    Brown-Forsythe test")
        print(f"    test stat: {stat}")
        print(f"    p-value:   {p_value}\n")
        # Cliff's delta:
        delta, _ = cliffs_delta(cluster1, cluster2)
        print(f"    Cliff's delta: {delta}")

    
    def relation_qualitative(self, df, atr_qualitative1, atr_qualitative2):
        '''
        --------------------------------------------------------------------------
        PARAMETER          DTYPE          DESCRIPTION
        --------------------------------------------------------------------------
        df               | pd.DataFrame | target dataframe
        atr_qualitative1 | string       | name of the first qualitative attribute
        atr_qualitative2 | string       | name of the second qualitative attribute
        --------------------------------------------------------------------------
        RETURN xxx
        --------------------------------------------------------------------------
        Used to explore the relationship between two qualitative attributes.
        '''
        print(">>> POST-CLUSTERING ANALYSIS (QUALITATIVE)")
        print(f"    {atr_qualitative1} <-> {atr_qualitative2}")
        # remove NaN if any:
        df = df.dropna(subset=[atr_qualitative1, atr_qualitative2], how="any")
        # create a contingency table:
        ctab = pd.crosstab(df[atr_qualitative1], df[atr_qualitative2])
        print(ctab)
        # Chi-squared test:
        chi2, p_value, _, _ = stats.chi2_contingency(ctab)
        print("    Chi-squared test")
        print(f"    test stat: {chi2}")
        print(f"    p-value:   {p_value}\n")
        # Fisher's exact test (if the contingency table is 2x2):
        if ctab.shape == (2, 2):
            odds_ratio, p_value = stats.fisher_exact(ctab)
            print("    Fisher's exact test")
            print(f"    odds-ratio: {odds_ratio}")
            print(f"    p-value:    {p_value}\n")
        else:
            # print that the Fisher's test was not performed:
            print("    Contingency table is not 2x2 => Fisher's exact test not performed\n")
        # Cramér's V:
        cramers_v = np.sqrt(chi2 / (ctab.sum().sum() * (min(ctab.shape) - 1)))
        print(f"    Cramér's V: {cramers_v}")