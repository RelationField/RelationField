# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from GARField
#   (https://github.com/chungmin99/garfield
# Copyright (c) 2014 GARField authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
from torch import nn, Tensor
from jaxtyping import Float


class MeanRenderer(nn.Module):
    """Calculate average of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: Float[Tensor, "bs num_samples num_classes"],
        weights: Float[Tensor, "bs num_samples 1"],
    ) -> Float[Tensor, "bs num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        return output


class FeatureRenderer(nn.Module):
    """Render feature embeddings along  a ray, where features are unit norm"""

    @classmethod
    def forward(
        cls,
        embeds: Float[Tensor, "bs num_samples num_classes"],
        weights: Float[Tensor, "bs num_samples 1"],
    ) -> Float[Tensor, "bs num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        output = output / torch.linalg.norm(output, dim=-1, keepdim=True)
        return output

