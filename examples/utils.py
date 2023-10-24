from diffusers import UNet2DModel, UNet2DConditionModel
from datasets import load_dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from typing import Union, Optional, List, Callable, Dict, Any
import torch
import os


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


def center_crop(image):

    width, height = image.size   # Get dimensions
    new_width = new_height = min(width, height)

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

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


@torch.no_grad()
def run_mscoco_pipe(
    self,
    prompt: Union[str, List[str]] = None,
    start_from = -1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    verbose=True
):
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    x_0_hats = []
    inputs = [latents.detach().clone().cpu()]

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    if verbose:
        f = tqdm
    else:
        f = lambda x: x

    for i in tqdm(range(len(timesteps))):

        t = timesteps[i]

        if start_from > 0 and t > start_from:
            continue


        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        out = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
        latents = out.prev_sample

        x_0_hat = out.pred_original_sample
        x_0_hats.append(x_0_hat.detach().clone().cpu())

        inputs.append(latents.detach().clone().cpu())


    image = self.decode_latents(latents)

    # 9. Run safety checker
    image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

    # 10. Convert to PIL
    image = self.numpy_to_pil(image)

    return image, x_0_hats[::-1], inputs[::-1]
