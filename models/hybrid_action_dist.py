# -----------------------------------------------------------
# Hybrid action distribution for FlagFrenzy action space
#
# Author : Sanjna Ravichandar
# Created: April 2025
# -----------------------------------------------------------

import torch
import torch.nn.functional as F

from ray.rllib.models.torch.torch_action_dist import (
    TorchMultiActionDistribution,
    TorchCategorical,
    TorchDiagGaussian,
)
import logging


class HybridActionDistribution(TorchMultiActionDistribution):
    """Hybrid action distribution: Discrete (Categorical) + Continuous (Gaussian), with
    special masking logic for params[1] (target entity) when action_type == 3 (engage)."""

    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space):
        super().__init__(
            inputs,
            model,
            child_distributions=[TorchCategorical, TorchDiagGaussian],
            input_lens=[
                action_space.spaces["action_type"].n,
                action_space.spaces["params"].shape[0] * 2
            ],
            action_space=action_space,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.model = model
        self.max_entities = 100 # TODO: Don't hardcode!

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        discrete_size = action_space.spaces["action_type"].n
        param_size = action_space.spaces["params"].shape[0]
        return discrete_size + (param_size * 2)

    def sample(self):
        action_type = self.flat_child_distributions[0].sample()
        gaussian_dist = self.flat_child_distributions[1]
        params_sampled = gaussian_dist.sample()  # shape: (batch, 10)

        engage_mask = getattr(self.model, "last_obs", {}).get("valid_engage_mask", None)
        entity_id_list_batch = getattr(self.model, "last_obs", {}).get("entity_id_list", None)

        if engage_mask is not None and entity_id_list_batch is not None:
            if engage_mask.dim() == 2:
                engage_mask = engage_mask.unsqueeze(0)

            batch_size = action_type.shape[0]
            engage_mask = engage_mask.view(batch_size, self.max_entities, self.max_entities)

            for i in range(batch_size):
                if action_type[i].item() == 3:
                    # print("Action 3 selected!")
                    # Decode source & target floats
                    src_float = torch.clamp(params_sampled[i, 0], 0.0, 1.0).item()
                    tgt_float = torch.clamp(params_sampled[i, 1], 0.0, 1.0).item()

                    entity_id_list = entity_id_list_batch[i]
                    entity_count = len(entity_id_list)

                    src_idx = int(src_float * (entity_count - 1))
                    tgt_idx = int(tgt_float * (entity_count - 1))

                    src_idx = min(max(src_idx, 0), entity_count - 1)
                    tgt_idx = min(max(tgt_idx, 0), entity_count - 1)

                    src_eid = entity_id_list[src_idx]
                    tgt_eid = entity_id_list[tgt_idx]

                    # Validity check
                    is_valid = engage_mask[i, src_idx, tgt_idx].item()

                    if is_valid == 0.0:
                        action_type[i] = 0  # force NO-OP
                        # self.logger.info(f"[MASKED OUT] Invalid engage: src_idx={src_idx} (ID={src_eid}) tgt_idx={tgt_idx} (ID={tgt_eid}):  NO-OP")
                    else:
                        # self.logger.info(f"[AFTER MASKING] Valid engage: src_idx={src_idx} (ID={src_eid}) tgt_idx={tgt_idx} (ID={tgt_eid})")
                        pass

        self.last_sample = {
            "action_type": action_type,
            "params": params_sampled
        }
        return self.last_sample


    def deterministic_sample(self):
        return self.sample()

    def logp(self, actions):
        # RLlib flattens actions: actions = (batch, 11)
        action_type = actions[:, 0].long()   # (batch,)
        params = actions[:, 1:]             # (batch, 10)

        logp_discrete = self.flat_child_distributions[0].logp(action_type)
        logp_continuous = self.flat_child_distributions[1].logp(params)

        return logp_discrete + logp_continuous

    def entropy(self):
        return sum(dist.entropy() for dist in self.flat_child_distributions)

    def kl(self, other):
        return sum(
            d.kl(o) for d, o in zip(
                self.flat_child_distributions,
                other.flat_child_distributions
            )
        )
