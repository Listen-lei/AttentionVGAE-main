import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
from _compat import Literal
import matplotlib.pyplot as plt
from typing import List

_QUALITY = Literal["fulres", "hires", "lowres"]

def read_10X_Visium(
    path,
    genome=None,
    count_file='filtered_feature_bc_matrix.h5',
    library_id=None,
    load_images=True,
    quality='hires',
    image_path=None
):
    adata = sc.read_visium(
        path,
        genome=genome,
        count_file=count_file,
        library_id=library_id,
        load_images=load_images,
    )
    adata.var_names_make_unique()

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale

    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality

    return adata


def read_SlideSeq(
    path,
    data_name,
    library_id=None,
    scale=None,
    quality="hires",
    spot_diameter_fullres=50,
    background_color="white",
) -> AnnData:
    count = pd.read_csv(os.path.join(path, f"{data_name}.count"))
    meta = pd.read_csv(os.path.join(path, f"{data_name}.idx"))

    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)
    adata.var["ENSEMBL"] = count["ENSEMBL"].values
    adata.obs["index"] = meta["index"].values

    if scale is None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    max_size = int(np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()]) + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))

    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {
        library_id: {
            "images": {quality: imgarr},
            "use_quality": quality,
            "scalefactors": {
                "tissue_" + quality + "_scalef": scale,
                "spot_diameter_fullres": spot_diameter_fullres,
            },
        }
    }

    adata.obsm["spatial"] = meta[["x", "y"]].values

    return adata


def read_stereoSeq(
    path,
    bin_size=100,
    is_sparse=True,
    library_id=None,
    scale=None,
    quality="hires",
    spot_diameter_fullres=1,
    background_color="white",
) -> AnnData:
    from scipy import sparse

    count = pd.read_csv(os.path.join(path, "count.txt"), sep='\t', comment='#', header=0)
    count.dropna(inplace=True)

    if "MIDCounts" in count.columns:
        count.rename(columns={"MIDCounts": "UMICount"}, inplace=True)

    count['x1'] = (count['x'] / bin_size).astype(np.int32)
    count['y1'] = (count['y'] / bin_size).astype(np.int32)
    count['pos'] = count['x1'].astype(str) + "-" + count['y1'].astype(str)

    bin_data = count.groupby(['pos', 'geneID'])['UMICount'].sum()
    cells = set(x[0] for x in bin_data.index)
    genes = set(x[1] for x in bin_data.index)
    cellsdic = dict(zip(cells, range(0, len(cells))))
    genesdic = dict(zip(genes, range(0, len(genes))))
    rows = [cellsdic[x[0]] for x in bin_data.index]
    cols = [genesdic[x[1]] for x in bin_data.index]

    exp_matrix = sparse.csr_matrix((bin_data.values, (rows, cols))) if is_sparse else \
                 sparse.csr_matrix((bin_data.values, (rows, cols))).toarray()

    obs = pd.DataFrame(index=cells)
    var = pd.DataFrame(index=genes)

    adata = AnnData(X=exp_matrix, obs=obs, var=var)
    pos = np.array(list(adata.obs.index.str.split('-', expand=True)), dtype=np.int)
    adata.obsm['spatial'] = pos

    if scale is None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor

    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

    max_size = int(np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()]) + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))

    imgarr = np.array(image)

    if library_id is None:
        library_id = "StereoSeq"

    adata.uns["spatial"] = {
        library_id: {
            "images": {quality: imgarr},
            "use_quality": quality,
            "scalefactors": {
                "tissue_" + quality + "_scalef": scale,
                "spot_diameter_fullres": spot_diameter_fullres,
            },
        }
    }

    return adata


def refine(
    sample_id: List,
    pred: List,
    dis: np.ndarray,
    shape: str = "hexagon"
) -> List:
    refined_pred = []
    pred_df = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)

    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        raise ValueError("Shape not recognized. Use 'hexagon' for Visium data or 'square' for ST data.")

    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred_df.loc[nbs.index, "pred"]
        self_pred = pred_df.loc[index, "pred"]
        v_c = nbs_pred.value_counts()

        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)

    return refined_pred
