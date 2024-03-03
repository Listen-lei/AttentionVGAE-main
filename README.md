# AttentionVGAE
___
![AVGN-last](https://github.com/Listen-lei/AttentionVGAE-main/assets/57699859/63c3e5d7-d21d-4f9c-b372-c9c5bb63024a)

___
## Abstract
```
The latest breakthroughs in spatially resolved transcriptomics (SRT) technology offer comprehensive opportunities to delve into gene expression patterns within the tissue microenvironment. However, the precise identification of spatial domains within tissues remains challenging. In this study,
we introduce AttentionVGAE (AVGN), which integrates slice images, spatial information, and raw gene expression while calibrating low-quality gene expression. By combining Variational Graph Autoencoder (VGAE) with Multi-Head Attention Blocks (MHA blocks), AVGN captures spatial relationships in
tissue gene expression, adaptively focusing on key features and alleviating the need for prior knowledge of cluster numbers, thereby achieving superior clustering performance. Particularly, AVGN attempts to balance the model's attention focus on local and global structures by utilizing MHA blocks,
an aspect that current graph neural networks have not extensively addressed. Benchmark testing demonstrates its significant efficacy in elucidating tissue anatomy and interpreting tumor heterogeneity, indicating its potential in advancing spatial transcriptomics research and understanding complex  biological phenomena.
```
## Requirement
```
python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3
```

## Run
```
First install the environment
```python
from run_analysis import RunAnalysis
import scanpy as sc
data_path = "/data/AttentionVGAE-main/data/DLPFC" 
data_name = '151673' 
save_path = "../result"
n_domains = 7

handle = RunAnalysis(
    save_path=save_path,
    use_gpu=True
)

adata = handle._get_adata(platform="Visium", data_path=data_path, data_name=data_name)

adata = handle._get_image_crop(adata, data_name=data_name)

adata = handle._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

graph_dict = handle._get_graph(adata.obsm["spatial"], distType="KDTree")

data = handle._data_process(adata, pca_n_comps=128 )
emb = handle._fit(
    data=data,
    graph_dict=graph_dict,
    Conv_type='GCNConv'
    )

adata.obsm["emb"] = emb

adata = handle._get_cluster_data(adata, n_domains=n_domains, priori=True)

sc.pl.spatial(adata, color='refine spatial domain', frameon=False, spot_size=150, img_key='hires')


```
  
## Acknowledgement
  Thanks to the input of the relevant researchers, especially the excellent code developers in the field, to provide a precedent for all the research of the later, best wishes.
