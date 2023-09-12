"""
TODO: Add docstring.
"""
from trak.modelout_functions import AbstractModelOutput
from typing import Iterable
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch


class DiffusionModelOutput(AbstractModelOutput):
    def __init__(self, conditional=False, latent=False) -> None:
        """
        Model output function for diffusion models.
        """
        super().__init__()
        from diffusers import DDPMScheduler

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.conditional = conditional
        self.latent = latent

        if self.latent:
            from diffusers import AutoencoderKL

            model_id = 'stabilityai/stable-diffusion-2'
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision="fp16")
            self.vae.to("cuda", dtype=torch.float16)
            self.vae.requires_grad_(False)

        self._are_we_featurizing = False

    def get_output(self, *args, **kwargs) -> Tensor:
        if self._are_we_featurizing:
            return self._get_output_featurizing(*args, **kwargs)
        else:
            return self._get_output_scoring(*args, **kwargs)

# class DiffusionModelOutputFeaturizing(AbstractModelOutput):
    def _get_output_featurizing(self,
                                model,
                                weights: Iterable[Tensor],
                                buffers: Iterable[Tensor],
                                image: Tensor,
                                label: Tensor,
                                timestep: Tensor):

        clean_image = image.unsqueeze(0)

        if self.latent:
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

# class DiffusionModelOutputScoringLikelihoodNoise(AbstractModelOutput):
    def _get_output_scoring(self,
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

        # TODO: either get rid of this, or make it less hardcoded
        # if self.ddim:
        #     tstep = tstep * 20

        latent = x0_hat

        noisy_latent = self.noise_scheduler.add_noise(latent, noise, tstep)

        if self.conditional:
            # i = np.random.randint(len(label))
            # kwargs = {'encoder_hidden_states': label[i].unsqueeze(0), 'return_dict': False}
            kwargs = {'encoder_hidden_states': label, 'return_dict': False}
        else:
            kwargs = {'return_dict': False}

        noise_pred = torch.func.functional_call(model,
                                                (weights, buffers),
                                                args=(noisy_latent, tstep.cuda()),
                                                kwargs=kwargs)[0]

        return F.mse_loss(noise_pred, noise)

    def get_out_to_loss_grad(self, model, weights, buffers, batch):
        latents, _, __ = batch
        return torch.ones(latents.shape[0]).to(latents.device).unsqueeze(-1)
