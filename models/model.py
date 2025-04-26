# -----------------------------------------------------------
# Custom Model for Flag Frenzy Env
#
# Author : Sanjna Ravichandar
# Created: April 2025
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()

class FlagFrenzyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.last_obs = None

        # TODO: Don't hardcode!
        self.max_entities = 100
        self.entity_feat_dim = 26
        self.mission_dim = 7

        self.action_dim_param = 10

        # Entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Flatten(),  # (max_entities * entity_feat_dim,)
            nn.Linear(self.max_entities * self.entity_feat_dim, 256),
            nn.ReLU()
        )

        # Radar visibility encoder
        self.visibility_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.max_entities * 2, 64),
            nn.ReLU()
        )

        # Mission status encoder
        self.mission_encoder = nn.Sequential(
            nn.Linear(self.mission_dim, 64),
            nn.ReLU()
        )

        # Controllable entities encoder
        self.controllable_encoder = nn.Sequential(
            nn.Linear(self.max_entities, 64),
            nn.ReLU()
        )

        # Combine and output
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 64 + 64 + 64, 256),
            nn.ReLU()
        )

        # Discrete action type head
        self.action_type_head = nn.Linear(256, 4)

        # Continuous param vector head
        self.param_head_mu = nn.Linear(256, self.action_dim_param)
        self.param_head_log_std = nn.Linear(256, self.action_dim_param)

        self._value_branch = nn.Linear(256, 1)
        self._last_value = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        self.last_obs = obs

        self.entity_id_list = input_dict["obs"]["entity_id_list"]

        entities = obs["entities"]
        mission = obs["mission"]
        visibility = obs["visibility"]
        controllable_entities= obs.get("controllable_entities", None)
        engage_mask = obs.get("valid_engage_mask", None)

        entity_encoded = self.entity_encoder(entities)

        legacy = visibility["legacy"].float()   # (batch, 100)
        dynasty = visibility["dynasty"].float() # (batch, 100)
        vis_concat = torch.cat([legacy, dynasty], dim=-1)   # (batch, 200)
        vis_encoded = self.visibility_encoder(vis_concat)   # (batch, 64)

        mission_encoded = self.mission_encoder(mission)

        controllable_encoded = self.controllable_encoder(controllable_entities)  # (B, 64)

        # Combine encoders
        x = torch.cat([controllable_encoded, entity_encoded, vis_encoded, mission_encoded], dim=-1)   # (batch, 256 + 64 + 64 + 64 = 384)
        features = self.combined_fc(x)  # (batch, 256)
        self._features = features

        # Compute logits + params
        logits = self.action_type_head(features)

        mu = self.param_head_mu(features)
        log_std = self.param_head_log_std(features).clamp(-5, 2)
        std = torch.exp(log_std)

        params = torch.cat([mu, std], dim=1)  # (batch, 20)

        # Mask out "engage target" (action_type == 3) if no valid targets are available
        if engage_mask is not None:
            if engage_mask.dim() == 1:
                engage_mask = engage_mask.unsqueeze(0)  # Add batch dim

            engage_mask = engage_mask.view(engage_mask.shape[0], -1)
            has_valid_targets = (engage_mask > 0.5).any(dim=1)

            mask_value = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
            logits[:, 3] = torch.where(has_valid_targets, logits[:, 3], mask_value)


        # Concatenate: 4 logits, 10 params
        return torch.cat([logits, params], dim=1), state

    @override(ModelV2)
    def value_function(self):
        return self._value_branch(self._features).squeeze(1)
