# AttentionVGAE
___
![AVGN_V3](https://github.com/Listen-lei/AttentionVGAE/assets/57699859/a3b851ea-ab30-4cfc-aab2-438afd7cfd9c)
___
## Abstract
Although recent breakthroughs in spatially resolved transcriptomics technology have provided a comprehensive opportunity to delve into gene expression patterns within the tissue microenvironment, the results of prior related research have been unsatisfactory. In this

study, we introduce AttentionVGAE and aim to address the challenge of capturing spatial domain information more precisely, thereby propelling further development in spatial transcriptomics research. Our primary objective is to enhance the understanding of tissue 

structure, function, and gene expression patterns in the spatial context by introducing a multi-head attention mechanism to the Variational Graph Autoencoder. To achieve this goal, AttentionVGAE employs a graph convolutional neural network as a baseline, aggregating 

histological images, gene information, and spatial location data from samples. This enables AttentionVGAE to genuinely perceive relationships between each point and its neighboring points in spatial space. It is noteworthy that AttentionVGAE exhibits robust adaptive 

capability when dealing with tasks involving no prior information on cluster quantity. Tests conducted on various platforms and datasets indicate that AttentionVGAE outperforms existing state-of-the-art methods. In summary, AttentionVGAE achieves a high level of 

performance in spatial domain detection tasks, and the multi-head attention-guided Variational Graph Autoencoder injects new vitality into the further advancement of spatial domain detection research. 

## Requirement
python == 3.9.* 

torch == 1.13.0 

scanpy == 1.9.2 

anndata == 0.8.0 

numpy == 1.22.3 

## Acknowledgement
Thanks to the input of the relevant researchers, especially the excellent code developers in the field, to provide a precedent for all the research of the later, best wishes.