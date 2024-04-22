# AttentionVGAE
___
![AVGN-last](https://github.com/Listen-lei/AttentionVGAE-main/assets/57699859/63c3e5d7-d21d-4f9c-b372-c9c5bb63024a)

___

## Requirements
```
python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3
```

## Dataset
```
(1) Human DLPFCs within the spatialLIBD at http://spatial.libd.org/spatialLIBD.  
(2) Adult mouse brain (fresh-frozen), Adult Human Glioblastoma Multiforme, and Infiltrating ductal carcinoma datasets at https://support.10xgenomics.com/spatial-gene-expression/datasets.  
(3) Mouse embryo data, Dorsal_midbrain_celand, and Adult mouse hemi-brain data at https://db.cngb.org/stomics/mosta.  
(4) Hippocampus dataset at https://portals.broadinstitute.org/single_cell/study/slide-seq-study.  
```

## Run

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

sc.pl.spatial(adata, color='refine spatial domain',  spot_size=150)
```

## Statement
```
I am glad that our research has been noticed, and if you feel that our research can help you in your work, please cite our article:  
https://doi.org/10.1093/bib/bbae173
```
