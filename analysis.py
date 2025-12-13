# Python version      3.11.4
import pandas as pd # 2.1.0
import utils        # script
# initialize custom toolkits:
visualization = utils.VisualizationToolkit()
analysis = utils.AnalysisToolkit()




# 1) LOAD THE DATA ------------------------------------------------------------
df_peritumoral = pd.read_csv("data/preprocessed/peritumoral.csv")     # peritumoral region
df_periedematous = pd.read_csv("data/preprocessed/periedematous.csv") # periedematous region
indices = range(14, 27)                                               # indices of diffusion data
# print number of samples:
print("samples with all peritumoral diffusion properties:   ", len(df_peritumoral.dropna(subset="GFAmed")))
print("samples with all periedematous diffusion properties: ", len(df_periedematous.dropna(subset="GFAmed")))

# 2) EXPLORATORY ANALYSIS -----------------------------------------------------
# plot the ratios:
visualization.histogram(df_peritumoral, "Ratio", None, "ratios", "probability of occurence in data", "", "slategray", "ratios")
# explore the computed diffusion properties:
for atr in df_peritumoral.columns[14:27]:
    visualization.violin(df_peritumoral, "Type", atr, "Sex", ["cornflowerblue", "indianred"], f"atr_{atr}_peritumoral")
    visualization.violin(df_periedematous, "Type", atr, "Sex", ["cornflowerblue", "indianred"], f"atr_{atr}_periedematous")
# correlation analysis:
analysis.correlation(df_peritumoral, indices, False, "corr_spearman_peritumoral")     # Spearman's rank correlation coefficient
analysis.correlation(df_peritumoral, indices, True, "corr_dcorr_peritumoral")         # distance correlation
analysis.correlation(df_periedematous, indices, False, "corr_spearman_periedematous") # Spearman's rank correlation coefficient
analysis.correlation(df_periedematous, indices, True, "corr_dcorr_periedematous")     # distance correlation

# 3) MANIFOLD LEARNING --------------------------------------------------------
# extend the datasets with 2D UMAP embeddings (each takes around 1.5 minutes):
params_peritumoral = ["euclidean", 9, 0.2543302859135638]   # optimal parameters from a previous run (peritumoral UMAP)
params_periedematous = ["euclidean", 6, 0.2810460850731149] # optimal parameters from a previous run (periedematous UMAP)
df_peritumoral = analysis.umap_manifold(df_peritumoral, indices, n_trials=100, plots_name="peritumoral", params=params_peritumoral)         # remove "params" to repeat optimization
df_periedematous = analysis.umap_manifold(df_periedematous, indices, n_trials=100, plots_name="periedematous", params=params_periedematous) # remove "params" to repeat optimization
# better readability:
df_peritumoral["AliveDead"] = df_peritumoral["AliveDead"].astype("string")
df_peritumoral["AliveDead"] = df_peritumoral["AliveDead"].replace(["1", "0"], ["dead", "alive"])
df_periedematous["AliveDead"] = df_periedematous["AliveDead"].astype("string")
df_periedematous["AliveDead"] = df_periedematous["AliveDead"].replace(["1", "0"], ["dead", "alive"])
# scatterplots using various classes:
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", None, ["slategray"], "umap_none_peritumoral")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", None, ["slategray"], "umap_none_periedematous")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Sex", visualization.sex_palette, "umap_sex_peritumoral")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "Sex", visualization.sex_palette, "umap_sex_periedematous")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Type", visualization.type_palette, "umap_type_peritumoral")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "Type", visualization.type_palette, "umap_type_periedematous")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Grade", visualization.grade_palette[::-1], "umap_grade_peritumoral")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "Grade", visualization.grade_palette[::-1], "umap_grade_periedematous")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "AliveDead", visualization.alivedead_palette, "umap_alivedead_peritumoral")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "AliveDead", visualization.alivedead_palette, "umap_alivedead_periedematous")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "MGMTstatus", "Set1", "umap_mgmtstatus_peritumoral")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "MGMTstatus", "Set1", "umap_mgmtstatus_periedematous")

# 4) CLUSTER ANALYSIS ---------------------------------------------------------
embedding_indices = range(27, 29) # range of UMAP output
# using Davies-Bouldin index:
df_peritumoral = analysis.gmm_clustering(df_peritumoral, embedding_indices, [2, 10], "peritumoral_dbi", "davies-bouldin")
df_periedematous = analysis.gmm_clustering(df_periedematous, embedding_indices, [2, 10], "periedematous_dbi", "davies-bouldin")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_peritumoral_dbi")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_periedematous_dbi")
# using Calinski-Harabasz index:
df_peritumoral = analysis.gmm_clustering(df_peritumoral, embedding_indices, [2, 10], "peritumoral_chi", "calinski-harabasz")
df_periedematous = analysis.gmm_clustering(df_periedematous, embedding_indices, [2, 10], "periedematous_chi", "calinski-harabasz")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_peritumoral_chi")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_periedematous_chi")
# using Silhouette score:
df_peritumoral = analysis.gmm_clustering(df_peritumoral, embedding_indices, [2, 10], "peritumoral_sil", "silhouette")
df_periedematous = analysis.gmm_clustering(df_periedematous, embedding_indices, [2, 10], "periedematous_sil", "silhouette")
visualization.scatter(df_peritumoral, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_peritumoral_sil")
visualization.scatter(df_periedematous, "Embedding1", "Embedding2", "Cluster", "tab10", "gmm_clustered_periedematous_sil")

# 5) POST-CLUSTERING ANALYSIS -------------------------------------------------
# quantitative analysis:
analysis.relation_quantitative(df_peritumoral, "Age", "Cluster")
analysis.relation_quantitative(df_peritumoral, "MGMTindex", "Cluster")
analysis.relation_quantitative(df_peritumoral, "OS", "Cluster")
# qualitative analysis:
analysis.relation_qualitative(df_peritumoral, "AliveDead", "Cluster")
analysis.relation_qualitative(df_peritumoral, "Sex", "Cluster")