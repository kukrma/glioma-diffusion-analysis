# ==================================================================================================== #
# --- DIPLOMA THESIS (analysis.py) ------------------------------------------------------------------- #
# ==================================================================================================== #
# title:         Analysis of White Matter Diffusion Properties in the Context of Selected Brain Tumors #
# author:        Bc. Martin KUKRÁL                                                                     #
# supervision:   doc. Ing. Roman MOUČEK, Ph.D.                                                         #
#                doc. MUDr. Irena HOLEČKOVÁ, Ph.D. (consultant)                                        #
# academic year: 2024/2025                                                                             #
# last updated:  2025-05-07                                                                            #
# ==================================================================================================== #
# Python version      3.11.4
import pandas as pd # 2.1.0
import utils        # script
# initialize custom toolkits:
visualization = utils.VisualizationToolkit()
analysis = utils.AnalysisToolkit()





# 1) LOAD THE DATA ------------------------------------------------------------
df_peritumoral = pd.read_csv("data/preprocessed/peritumoral.csv") # peritumoral region
df_periedemal = pd.read_csv("data/preprocessed/periedemal.csv")   # periedemal region
indices = range(14, 27)                                           # indices of diffusion data

# 2) EXPLORATORY ANALYSIS -----------------------------------------------------
# plot the ratios:
visualization.histogram(df_peritumoral, "Ratio", None, "ratios", "probability of occurence in data", "", "slategray", "ratios")
# explore the computed diffusion properties:
for atr in df_peritumoral.columns[14:27]:
    visualization.violin(df_peritumoral, "Type", atr, "Sex", ["cornflowerblue", "indianred"], f"atr_{atr}_peritumoral")
    visualization.violin(df_periedemal, "Type", atr, "Sex", ["cornflowerblue", "indianred"], f"atr_{atr}_periedemal")
# correlation analysis:
analysis.correlation(df_peritumoral, indices, False, "corr_spearman_peritumoral") # Spearman's rank correlation coefficient
analysis.correlation(df_peritumoral, indices, True, "corr_dcorr_peritumoral")     # distance correlation
analysis.correlation(df_periedemal, indices, False, "corr_spearman_periedemal")   # Spearman's rank correlation coefficient
analysis.correlation(df_periedemal, indices, True, "corr_dcorr_periedemal")       # distance correlation

# 3) MANIFOLD LEARNING --------------------------------------------------------
# extend the datasets with 2D UMAP embeddings (each takes around 1.5 minutes):
params_peritumoral = ["euclidean", 6, 0.185664993078593] # optimal parameters from a previous run (peritumoral UMAP)
params_periedemal = ["euclidean", 7, 0.13583387128117366]  # optimal parameters from a previous run (periedemal UMAP)
df_peritumoral = analysis.umap_manifold(df_peritumoral, indices, n_trials=100, plots_name="peritumoral", params=params_peritumoral) # remove "params" to repeat optimization
df_periedemal = analysis.umap_manifold(df_periedemal, indices, n_trials=100, plots_name="periedemal", params=params_periedemal)     # remove "params" to repeat optimization
# better readability:
df_peritumoral["AliveDead"] = df_peritumoral["AliveDead"].astype("string")
df_peritumoral["AliveDead"] = df_peritumoral["AliveDead"].replace(["1", "0"], ["dead", "alive"])
df_periedemal["AliveDead"] = df_periedemal["AliveDead"].astype("string")
df_periedemal["AliveDead"] = df_periedemal["AliveDead"].replace(["1", "0"], ["dead", "alive"])
# scatterplots using various classes:
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", None, ["slategray"], "umap_none_peritumoral")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", None, ["slategray"], "umap_none_periedemal")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Sex", visualization.sex_palette, "umap_sex_peritumoral")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "Sex", visualization.sex_palette, "umap_sex_periedemal")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Type", visualization.type_palette, "umap_type_peritumoral")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "Type", visualization.type_palette, "umap_type_periedemal")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Grade", visualization.grade_palette[::-1], "umap_grade_peritumoral")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "Grade", visualization.grade_palette[::-1], "umap_grade_periedemal")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "AliveDead", visualization.alivedead_palette, "umap_alivedead_peritumoral")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "AliveDead", visualization.alivedead_palette, "umap_alivedead_periedemal")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "MGMTstatus", "Set1", "umap_mgmtstatus_peritumoral")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "MGMTstatus", "Set1", "umap_mgmtstatus_periedemal")

# 4) CLUSTER ANALYSIS ---------------------------------------------------------
embedding_indices = range(27, 29) # range of UMAP output
# using Davies-Bouldin index:
df_peritumoral = analysis.gmm_clustering(df_peritumoral, embedding_indices, [2, 10], "peritumoral_dbi", "davies-bouldin")
df_periedemal = analysis.gmm_clustering(df_periedemal, embedding_indices, [2, 10], "periedemal_dbi", "davies-bouldin")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_peritumoral_dbi")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_periedemal_dbi")
# using Calinski-Harabasz index:
df_peritumoral = analysis.gmm_clustering(df_peritumoral, embedding_indices, [2, 10], "peritumoral_chi", "calinski-harabasz")
df_periedemal = analysis.gmm_clustering(df_periedemal, embedding_indices, [2, 10], "periedemal_chi", "calinski-harabasz")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_peritumoral_chi")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_periedemal_chi")
# using Silhouette score:
df_peritumoral = analysis.gmm_clustering(df_peritumoral, embedding_indices, [2, 10], "peritumoral_sil", "silhouette")
df_periedemal = analysis.gmm_clustering(df_periedemal, embedding_indices, [2, 10], "periedemal_sil", "silhouette")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_peritumoral_sil")
visualization.scatter(df_periedemal, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_periedemal_sil")

# 5) POST-CLUSTERING ANALYSIS -------------------------------------------------
# quantitative analysis:
analysis.relation_quantitative(df_peritumoral, "Age", "Cluster")
analysis.relation_quantitative(df_peritumoral, "MGMTindex", "Cluster")
analysis.relation_quantitative(df_peritumoral, "OS", "Cluster")
# qualitative analysis:
analysis.relation_qualitative(df_peritumoral, "AliveDead", "Cluster")
analysis.relation_qualitative(df_peritumoral, "Sex", "Cluster")