from diffusers import UNet2DModel, UNet2DConditionModel
from datasets import load_dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import torch
import os
import numpy as np


def get_cifar_model(sample_size, n_channels=3) -> UNet2DModel:
    # Model matching DDPM paper
    model = UNet2DModel(
        sample_size=sample_size,  # the target image resolution
        in_channels=n_channels,  # the number of input channels, 3 for RGB images
        out_channels=n_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 256),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D"
            ),
        attention_head_dim=None,
    )

    return model.cuda()


def get_mscoco_model(sample_size, n_channels=4) -> UNet2DConditionModel:
    sample_size = sample_size // 8
    model = UNet2DConditionModel(
        sample_size=sample_size,  # the target image resolution
        in_channels=n_channels,  # the number of input channels, 3 for RGB images
        out_channels=n_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 256, 256),  # the number of output channels for each UNet block
        cross_attention_dim=1024,  # NOTE: 1024 for V2,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D"
            ),
        attention_head_dim=8,
    )

    return model.cuda()


def get_cifar_loader(batch_size, split="train"):
    dataset = load_dataset("cifar10", split=split)

    preprocess = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # apply image transofmrations on the fly during training
    def transform(examples) -> dict[str, list]:
        images = examples["img"]

        images = [preprocess(image.convert("RGB")) for image in images]

        return {"images": images}

    dataset.set_transform(transform)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False)

    return dataloader


def get_cifar_ffcv_loader(batch_size, split="train",
                          distributed=False,
                          num_workers=5,
                          indices=None,):
    # ffcv imports
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import Convert, ToTensor, ToDevice, Squeeze, \
        RandomHorizontalFlip, ToTorchImage
    from ffcv.fields.basics import IntDecoder
    paths = {
        'train': '/mnt/cfs/datasets/ffcv_datasets/cifar10/cifar10_train.beton',
        'test': '/mnt/cfs/datasets/ffcv_datasets/cifar10/cifar10_test.beton',
        }

    path = paths[split]

    MEAN = np.array([0.5]) * 255
    STD = np.array([0.5]) * 255
    res = 32  # CIFAR-10 has 32x32 images

    decoder = CenterCropRGBImageDecoder((res, res), ratio=1)
    image_pipeline = [decoder]

    use_flip = (split == 'train')
    if use_flip:
        image_pipeline.append(RandomHorizontalFlip())

    image_pipeline += [
        ToTensor(),
        ToDevice(torch.device('cuda'), non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16),
        transforms.Normalize(MEAN, STD),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        Convert(torch.int64),
        ToDevice(torch.device('cuda'), non_blocking=True)
        ]

    loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    indices=indices,
                    os_cache=True,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)

    return loader

    pass


def center_crop(image):

    width, height = image.size   # Get dimensions
    new_width = new_height = min(width, height)

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    return image


class COCODataset:
    def __init__(self, path='/mnt/xfs/datasets/coco2017', split='train'):
        dataType = f'{split}2017'
        annFile = os.path.join(path, 'annotations', f"captions_{dataType}.json")
        self.imgdir = os.path.join(path, 'images', dataType)
        self.coco = COCO(annFile)
        self.img_ids = list(self.coco.imgs.keys())
        self.captions = self.coco.imgToAnns
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, idx):

        # get image
        i = self.img_ids[idx]
        img_dict = self.coco.loadImgs([i])[0]
        path = os.path.join(self.imgdir, img_dict['file_name'])

        image = Image.open(path).convert('RGB')
        im = center_crop(image).resize((128, 128))

        # get captions
        captions = [x['caption'] for x in self.captions[i]]

        return self.preprocess(im), captions

    def __len__(self):
        return len(self.img_ids)


def get_mscoco_loader(batch_size, split='train'):
    ds = COCODataset(split=split)

    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             # collate_fn=lambda x: x)
                                             collate_fn=lambda x: tuple(zip(*x)))

    return dataloader
