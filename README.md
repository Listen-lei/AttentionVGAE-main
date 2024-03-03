# AttentionVGAE
___
![AVGN-last](https://github.com/Listen-lei/AttentionVGAE-main/assets/57699859/63c3e5d7-d21d-4f9c-b372-c9c5bb63024a)

___
## Abstract
"""The latest breakthroughs in spatially resolved transcriptomics (SRT) technology offer comprehensive opportunities to delve into gene expression patterns within the tissue microenvironment. However, the precise identification of spatial domains within tissues remains challenging. In this study, we introduce AttentionVGAE (AVGN), which integrates slice images, spatial information, and raw gene expression while calibrating low-quality gene expression. By combining Variational Graph Autoencoder (VGAE) with Multi-Head Attention Blocks (MHA blocks), AVGN captures spatial relationships in tissue gene expression, adaptively focusing on key features and alleviating the need for prior knowledge of cluster numbers, thereby achieving superior clustering performance. Particularly, AVGN attempts to balance the model's attention focus on local and global structures by utilizing MHA blocks, an aspect that current graph neural networks have not extensively addressed. Benchmark testing demonstrates its significant efficacy in elucidating tissue anatomy and interpreting tumor heterogeneity, indicating its potential in advancing spatial transcriptomics research and understanding complex biological phenomena."""

## Requirement
  python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3  

## Acknowledgement
  Thanks to the input of the relevant researchers, especially the excellent code developers in the field, to provide a precedent for all the research of the later, best wishes.
