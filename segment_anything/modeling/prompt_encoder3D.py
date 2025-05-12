# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class PromptEncoder3D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        pos_dim : int = 3,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pos_dim = pos_dim
        # todo: 2d pos for prompt
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 3, pos_dim=pos_dim)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (image_embedding_size[0], image_embedding_size[1], image_embedding_size[2])
        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans // 4),
            activation(),
            nn.Conv3d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm3d(mask_in_chans),
            activation(),
            nn.Conv3d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  # 1xXxYxZ

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1], self.image_embedding_size[2]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Supports 2D or 3D encoding based on pos_dim.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None, pos_dim: int = 3) -> None:
        """
        Args:
            num_pos_feats (int): Number of random frequencies per dimension.
                                 The output feature dimension will be 3 * num_pos_feats.
            scale (Optional[float]): Scaling factor for the random frequencies.
            pos_dim (int): Dimensionality of the positional encoding (2 for 2D, 3 for 3D).
                           Input grid/coords are assumed to be 3D, but PE is applied
                           only to the first pos_dim coordinates.
        """
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0

        if pos_dim not in [2, 3]:
            raise ValueError(f"pos_dim must be 2 or 3, but got {pos_dim}")
        self.pos_dim = pos_dim
        # print(f"create {pos_dim}D pe")
        # The Gaussian matrix shape depends on the dimensionality of the position encoding
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((self.pos_dim, num_pos_feats)),
        )
        
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points.
        Input coords are assumed to be normalized to [0, 1]^self.pos_dim
        and have shape d_1 x ... x d_n x self.pos_dim or B x N x self.pos_dim.
        Outputs shape d_1 x ... x d_n x (3*num_pos_feats) or B x N x (3*num_pos_feats).
        """
        # assuming coords are in [0, 1]^pos_dim
        coords = 2 * coords - 1  # Normalize to [-1, 1]

        # Apply random projection
        # Input shape: ... x self.pos_dim
        # Matrix shape: self.pos_dim x num_pos_feats
        # Output shape: ... x num_pos_feats
        coords = coords @ self.positional_encoding_gaussian_matrix

        # Apply sin and cos
        coords = 2 * np.pi * coords
        
        # outputs shape ... x (3 * num_pos_feats)
        # Note: The original code concatenates sin, cos, and sin again.
        # This results in 3*num_pos_feats features per point.
        return torch.cat([torch.sin(coords), torch.cos(coords), torch.sin(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a grid of the specified size.
        The output PE will have shape (3*num_pos_feats) x X x Y x Z.
        If pos_dim is 2, the PE will vary only across X and Y dimensions.
        """
        # size is (X, Y, Z)
        x_size, y_size, z_size = size
        device: Any = self.positional_encoding_gaussian_matrix.device

        # Create grid coordinates for all 3 dimensions, normalized to [0, 1]
        # We use cumsum on a tensor of ones, then subtract 0.5 for pixel centers,
        # and finally normalize by the dimension size.
        x_embed = (torch.linspace(0.5, x_size - 0.5, x_size, device=device) / x_size).unsqueeze(0).unsqueeze(0).repeat(z_size, y_size, 1).permute(2, 1, 0)
        y_embed = (torch.linspace(0.5, y_size - 0.5, y_size, device=device) / y_size).unsqueeze(0).unsqueeze(2).repeat(z_size, 1, x_size).permute(2, 1, 0)
        z_embed = (torch.linspace(0.5, z_size - 0.5, z_size, device=device) / z_size).unsqueeze(1).unsqueeze(2).repeat(1, y_size, x_size).permute(2, 1, 0)

        # Stack the coordinates. Shape: X x Y x Z x 3
        coords_3d = torch.stack([x_embed, y_embed, z_embed], dim=-1)

        # Select the first pos_dim coordinates for encoding
        # If pos_dim is 2, we take X and Y (columns 0 and 1)
        # If pos_dim is 3, we take X, Y, and Z (columns 0, 1, and 2)
        coords_to_encode = coords_3d[:, :, :, :self.pos_dim] # Shape: X x Y x Z x self.pos_dim

        # Apply positional encoding
        pe = self._pe_encoding(coords_to_encode) # Shape: X x Y x Z x (3*num_pos_feats)

        # Permute to get the channel dimension first
        # Shape: (3*num_pos_feats) x X x Y x Z
        return pe.permute(3, 0, 1, 2)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0,1].
        Input coords_input shape: B x N x 3 (or ... x 3)
        Output shape: B x N x (3*num_pos_feats) (or ... x (3*num_pos_feats))
        If pos_dim is 2, only the first two coordinates (X, Y) are used for encoding.
        """
        # Make a copy to avoid modifying the input tensor
        coords = coords_input.clone() # Shape: B x N x 3

        # Normalize coordinates based on image_size
        # Assuming image_size is (Width, Height, Depth) -> (X, Y, Z)
        coords[:, :, 0] = coords[:, :, 0] / image_size[0] # Normalize X
        coords[:, :, 1] = coords[:, :, 1] / image_size[1] # Normalize Y
        # We normalize Z even if pos_dim is 2, but we won't use it for encoding
        coords[:, :, 2] = coords[:, :, 2] / image_size[2] # Normalize Z

        # Select the first pos_dim coordinates for encoding
        # If pos_dim is 2, we take X and Y (columns 0 and 1)
        # If pos_dim is 3, we take X, Y, and Z (columns 0, 1, and 2)
        coords_to_encode = coords[:, :, :self.pos_dim].to(torch.float) # Shape: B x N x self.pos_dim

        # Apply positional encoding
        return self._pe_encoding(coords_to_encode) # Shape: B x N x (3*num_pos_feats)

if __name__ == '__main__':
    # Example with 3D encoding (default)
    pe_layer_3d = PositionEmbeddingRandom(num_pos_feats=64, pos_dim=3)
    grid_size_3d = (32, 32, 16) # X, Y, Z
    pe_3d = pe_layer_3d(grid_size_3d)
    print(f"3D PE grid shape: {pe_3d.shape}") # Expected: (3*64) x 32 x 32 x 16

    coords_3d_input = torch.tensor([[[10.5, 5.2, 8.1], [25.1, 15.9, 1.3]]], dtype=torch.float32) # Batch=1, N=2 points
    image_size_3d = (32, 32, 16)
    pe_3d_coords = pe_layer_3d.forward_with_coords(coords_3d_input, image_size_3d)
    print(f"3D PE coords shape: {pe_3d_coords.shape}") # Expected: 1 x 2 x (3*64)

    # Example with 2D encoding
    pe_layer_2d = PositionEmbeddingRandom(num_pos_feats=64, pos_dim=2)
    grid_size_2d = (32, 32, 16) # Still input as 3D size
    pe_2d = pe_layer_2d(grid_size_2d)
    print(f"\n2D PE grid shape (on 3D grid): {pe_2d.shape}") # Expected: (3*64) x 32 x 32 x 16
                                                           # PE varies over X, Y, constant over Z

    coords_2d_input = torch.tensor([[[10.5, 5.2, 8.1], [25.1, 15.9, 1.3]]], dtype=torch.float32) # Batch=1, N=2 points
    image_size_2d = (32, 32, 16)
    pe_2d_coords = pe_layer_2d.forward_with_coords(coords_2d_input, image_size_2d)
    print(f"2D PE coords shape (using first 2 coords): {pe_2d_coords.shape}") # Expected: 1 x 2 x (3*64)
    # Notice that even though the input coordinates have a Z component (8.1 and 1.3),
    # only the X (10.5, 25.1) and Y (5.2, 15.9) are used for the 2D encoding when pos_dim=2.
