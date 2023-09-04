"""
TODO: Add docstring.
"""
from trak.modelout_functions import AbstractModelOutput
from typing import Iterable
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch


class DiffusionModelOutputFeaturizing(AbstractModelOutput):
    def __init__(self, conditional=False, latent=False, vqvae=False) -> None:
        """
        Model output function for diffusion models.
        """
        super().__init__()
        from diffusers import DDPMScheduler

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.conditional = conditional
        self.latent = latent
        self.vqvae = vqvae

        if self.latent:

            if self.vqvae:

                from diffusers import VQModel

                self.vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
                self.vae.to("cuda", dtype=torch.float16)
                self.vae.requires_grad_(False)

            else:

                from diffusers import AutoencoderKL

                model_id = 'stabilityai/stable-diffusion-2'
                self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision="fp16")
                self.vae.to("cuda", dtype=torch.float16)
                self.vae.requires_grad_(False)

    def get_output(self,
                   model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   image: Tensor,
                   label: Tensor,
                   timestep: Tensor):

        clean_image = image.unsqueeze(0)

        if self.latent:
            if self.vqvae:
                latent = self.vae.encode(clean_image).latents
            else:
                latent = self.vae.encode(clean_image).latent_dist.sample()

            latent = latent * self.vae.config.scaling_factor
        else:
            latent = clean_image

        noise = torch.randn(latent.shape).to(latent.device)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timestep)

        if self.conditional:
            i = np.random.randint(len(label))
            kwargs = {'encoder_hidden_states': label[i].unsqueeze(0), 'return_dict': False}
        else:
            kwargs = {'return_dict': False}

        noise_pred = torch.func.functional_call(model,
                                                (weights, buffers),
                                                args=(noisy_latent, timestep),
                                                kwargs=kwargs)[0]
        return F.mse_loss(noise_pred, noise)

    def get_out_to_loss_grad(self, model, weights, buffers, batch):
        latents, _, __ = batch
        return torch.ones(latents.shape[0]).to(latents.device).unsqueeze(-1)


TASK_TO_MODELOUT = {
    'diffusion_featurizing': DiffusionModelOutputFeaturizing,
}


class DiffusionModelOutputScoringLikelihoodNoise(AbstractModelOutput):
    def __init__(self, conditional=False, ddim=False, mask=None, delta_t=0) -> None:
        """
        For scoring, returns the MSE between the U-net output and the noise
        added to the on the x_0s (predicted original samples) along the
        trajectory of the synthesized image.
        """
        super().__init__()
        from diffusers import DDPMScheduler

        self.ddim = ddim

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.conditional = conditional

        self.mask = mask
        self.delta_t = delta_t

    def get_output(self,
                   model,
                   weights: Iterable[Tensor],
                   buffers: Iterable[Tensor],
                   tstep,
                   noise,
                   x_0_hats: Tensor = None,  # shape [batch_size, 1000, 3, 32, 32]
                   label: Tensor = None,  # shape [batch_size]
                   ):

        noise = noise.unsqueeze(0)

        x0_hat = x_0_hats[tstep].cuda().unsqueeze(0)

        tstep = tstep + self.delta_t
        if self.ddim:
            tstep = tstep * 20

        latent = x0_hat

        noisy_latent = self.noise_scheduler.add_noise(latent, noise, tstep)

        if self.conditional:
            i = np.random.randint(len(label))
            kwargs = {'encoder_hidden_states': label[i].unsqueeze(0), 'return_dict': False}
        else:
            kwargs = {'return_dict': False}

        noise_pred = torch.func.functional_call(model,
                                                (weights, buffers),
                                                args=(noisy_latent, tstep.cuda()),
                                                kwargs=kwargs)[0]

        if self.mask is None:

            return F.mse_loss(noise_pred, noise)

        else:
            return F.mse_loss(noise_pred*self.mask, noise*self.mask)

    def get_out_to_loss_grad(self, model, weights, buffers, batch):
        latents, _, __ = batch
        return torch.ones(latents.shape[0]).to(latents.device).unsqueeze(-1)
