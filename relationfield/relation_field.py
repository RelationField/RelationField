# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0


import sys
import numpy as np
import torch

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import NeRFEncoding


try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


class RelationField(Field):
    def __init__(
        self,
        grid_layers,
        grid_sizes,
        grid_resolutions,
        num_hidden_clip_layers,
        shared_encoding: bool = False,
        relation_semantics: bool = False,
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ):
        super().__init__()
        assert len(grid_layers) == len(grid_sizes) and len(grid_resolutions) == len(grid_layers)
        self.spatial_distortion = spatial_distortion
        if shared_encoding:
            self.encs = self.encs = torch.nn.ModuleList(
                [
                    RelationField._get_encoding_nerf()
                ]	
            )
        else:
            self.encs = torch.nn.ModuleList(
                [
                    RelationField._get_encoding(
                        grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i], indim=3, hash_size=grid_sizes[i]
                    )
                    for i in range(len(grid_layers))
                ]
            )

        if relation_semantics:
            n_input_dims = 768*2 + 96*4 # 2* 2 hashgrid encodings + 2* 1 openseg feature vector
        else:
            n_input_dims = 96*4 # 2* 2 hashgrid encodings
        self.relation_net = tcnn.Network(
            n_input_dims=n_input_dims, 
            n_output_dims=512, 
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": num_hidden_clip_layers,
            },
        )

        

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc
    
    @staticmethod
    def _get_encoding_nerf():
        position_encoding = NeRFEncoding(
            in_dim=6, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        return position_encoding
        

    def get_outputs(self, ray_samples: RaySamples) -> dict:
        return {}