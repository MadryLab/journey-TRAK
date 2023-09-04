from typing import Iterable
from torch import Tensor
from trak.gradient_computers import AbstractGradientComputer
from trak.modelout_functions import AbstractModelOutput
from .utils import _accumulate_vectorize, get_num_params
import torch


class DiffusionGradientComputer(AbstractGradientComputer):
    def __init__(self,
                 model: torch.nn.Module,
                 task: AbstractModelOutput,
                 grad_dim: int) -> None:
        super().__init__(model, task, grad_dim)
        self.model = model
        self.num_params = get_num_params(self.model)
        self.load_model_params(model)
        self._are_we_featurizing = False

    def load_model_params(self, model) -> None:
        """ Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self,
                                batch: Iterable[Tensor],
                                ) -> Tensor:
        """ Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.
        """
        if self._are_we_featurizing:
            return self._compute_per_sample_grad_featurizing(batch)
        else:
            return self._compute_per_sample_grad_scoring(batch)

    def _compute_per_sample_grad_featurizing(self,
                                             batch: Iterable[Tensor],
                                             ) -> Tensor:
        """
        Args:
            batch (Iterable[Tensor]):
                contains [data, labels, timesteps]
                each of those arrays should have the same first dimension (=batch size)

        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.

        """
        images, labels, timesteps = batch
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(self.modelout_fn.get_output, has_aux=False, argnums=1)

        batch_size = batch[0].shape[0]
        grads = torch.zeros(size=(batch_size, self.num_params),
                            dtype=batch[0].dtype,
                            device=batch[0].device)

        for i_tstep in range(timesteps.shape[1]):
            tsteps = timesteps[:, i_tstep:i_tstep + 1]
            # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
            _accumulate_vectorize(g=torch.func.vmap(grads_loss,
                                                    in_dims=(None, None, None, 0, 0, 0),
                                                    randomness='different')(self.model,
                                                                            self.func_weights,
                                                                            self.func_buffers,
                                                                            images,
                                                                            labels,
                                                                            tsteps),
                                  arr=grads)
        return grads

    def _compute_per_sample_grad_scoring(self,
                                         batch: Iterable[Tensor],
                                         ) -> Tensor:
        """
        Args:
            batch (Iterable[Tensor]):
            [images, labels, timesteps]
            each of the three should have the same first dimension (batch size)
            for CIFAR:
            images.shape = [batch size, 1000, 3, 32, 32]
            labels.shape = [batch size, 1]
            timesteps.shape = [batch size, num_timesteps, 1] or [batch size, num_timesteps, 2]
            2 should be passed if using trajectory noise (rather than fresh
            noise); in that case, it will use the first one to index into the
            x_0_hats hat and the second one to index into x_ts and  pass to the
            U-net.


        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(self.modelout_fn.get_output, has_aux=False, argnums=1)

        batch_size = batch[0].shape[0]
        grads = torch.zeros(size=(batch_size, self.num_params),
                            dtype=batch[0].dtype,
                            device='cuda')

        images, labels, tstep, n_iters = batch

        for i in range(n_iters):
            noise = torch.randn(images[:, 0].shape, device='cuda')
            # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
            # batch_size x num_timesteps x dim, ordered x_0 to x_T
            _accumulate_vectorize(g=torch.func.vmap(grads_loss,
                                                    in_dims=(None, None, None, None, *([0] * 3)),
                                                    randomness='different')(self.model,
                                                                            self.func_weights,
                                                                            self.func_buffers,
                                                                            tstep,
                                                                            noise,
                                                                            images,
                                                                            labels
                                                                            ),
                                  arr=grads)
        return grads

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output

        .. math::

            \\partial \\ell / \\partial \\text{(model output)}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`FunctionalGradientComputer:.get_output`,
        i.e. something like:

        .. code-block:: python

            grad_out_to_loss = grad(self.model_out_to_loss_grad, ...)
            grads = vmap(grad_out_to_loss, ...)
            ...

        Args:
            batch (Iterable[Tensor]):
                batch of data

        """
        return self.modelout_fn.get_out_to_loss_grad(self.model,
                                                     self.func_weights,
                                                     self.func_buffers,
                                                     batch)
