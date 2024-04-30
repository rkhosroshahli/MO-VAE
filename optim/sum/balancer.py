import torch

from .. import basic_balancer
from .. import balancers


@balancers.register("linear")
class LinearScalarization(basic_balancer.BasicBalancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def step(self, losses,
             shared_params,
             task_specific_params,
             shared_representation=None,
             last_shared_layer_params=None
             ):

        if self.compute_stats:
            G = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=False)
            self.compute_metrics(G)
        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)
        self.set_loss_weights({task_id: 1.0 for _, task_id in enumerate(losses)})
