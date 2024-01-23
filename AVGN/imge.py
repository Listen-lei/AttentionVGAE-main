import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import random

class ImageFeature:
    def __init__(
        self,
        adata,
        pca_components=50,
        cnn_type='MaxVit',
        verbose=False,
        seeds=88,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = adata
        self.pca_components = pca_components
        self.verbose = verbose
        self.seeds = seeds
        self.cnn_type = cnn_type

    def load_cnn_model(self):
        if self.cnn_type == 'ResNet50':
            cnn_pretrained_model = models.resnet50(pretrained=True)
        elif self.cnn_type == 'ResNet152':
            cnn_pretrained_model = models.resnet152(pretrained=True)
        elif self.cnn_type == 'Vgg19':
            cnn_pretrained_model = models.vgg19(pretrained=True)
        elif self.cnn_type == 'Vgg16':
            cnn_pretrained_model = models.vgg16(pretrained=True)
        elif self.cnn_type == 'DenseNet121':
            cnn_pretrained_model = models.densenet121(pretrained=True)
        elif self.cnn_type == 'Inception_v3':
            cnn_pretrained_model = models.inception_v3(pretrained=True)
        elif self.cnn_type == 'MaxVit':
            cnn_pretrained_model = models.maxvit_t(weights=True)
        elif self.cnn_type == 'vit_b_16':
            cnn_pretrained_model = models.vit_b_16(weights='DEFAULT')
        elif self.cnn_type == 'swin_t':
            cnn_pretrained_model = models.swin_t(weights='DEFAULT')
        else:
            raise ValueError(f"{self.cnn_type} is not a valid type.")

        cnn_pretrained_model.to(self.device)
        return cnn_pretrained_model

    def extract_image_feat(self):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAutocontrast(),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.)),
            transforms.RandomInvert(),
            transforms.RandomAdjustSharpness(random.uniform(0, 1)),
            transforms.RandomSolarize(random.uniform(0, 1)),
            transforms.RandomAffine(45, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=(-0.3, 0.3, -0.3, 0.3)),
            transforms.RandomErasing()
        ]

        img_to_tensor = transforms.Compose(transform_list)

        feat_df = pd.DataFrame()
        model = self.load_cnn_model()
        model.eval()

        if "slices_path" not in self.adata.obs.keys():
            raise ValueError("Please run the function image_crop first")

        with tqdm(
            total=len(self.adata),
            desc="Extract image feature",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for spot, slice_path in self.adata.obs['slices_path'].items():
                spot_slice = Image.open(slice_path)
                spot_slice = spot_slice.resize((224, 224))
                spot_slice = np.asarray(spot_slice, dtype="int32")
                spot_slice = spot_slice.astype(np.float32)
                tensor = img_to_tensor(spot_slice)
                tensor = tensor.resize_(1, 3, 224, 224)
                tensor = tensor.to(self.device)
                result = model(Variable(tensor))
                result_npy = result.data.cpu().numpy().ravel()
                feat_df[spot] = result_npy
                feat_df = feat_df.copy()
                pbar.update(1)

        self.adata.obsm["image_feat"] = feat_df.transpose().to_numpy()

        if self.verbose:
            print("The image feature is added to adata.obsm['image_feat']!")

        pca = PCA(n_components=self.pca_components, random_state=self.seeds)
        pca.fit(feat_df.transpose().to_numpy())
        self.adata.obsm["image_feat_pca"] = pca.transform(feat_df.transpose().to_numpy())

        if self.verbose:
            print("The pca result of image feature is added to adata.obsm['image_feat_pca']!")

        return self.adata


def image_crop(
    adata,
    save_path,
    library_id=None,
    crop_size=50,
    target_size=224,
    verbose=False,
):
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]

    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(
        total=len(adata),
        desc="Tiling image",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]"
    ) as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )
            tile.thumbnail((target_size, target_size), Image.LANCZOS)
            tile.resize((target_size, target_size))
            tile_name = f"{imagecol}-{imagerow}-{crop_size}"
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))

            if verbose:
                print(f"Generate tile at location ({imagecol}, {imagerow})")

            tile.save(out_tile, "PNG")
            pbar.update(1)

    adata.obs["slices_path"] = tile_names

    if verbose:
        print("The slice path of image feature is added to adata.obs['slices_path']!")

    return adata
