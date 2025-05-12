# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT3D, MaskDecoder3D, PromptEncoder3D, Sam3D

def _build_sam3D(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 384
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_sam3D_ori(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    pos_dim=3,
    atten_dim=3,
):
    prompt_embed_dim = 384
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            pos_dim=pos_dim,
            atten_dim=atten_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
            pos_dim=pos_dim,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def build_sam3D_vit_b(checkpoint=None):
    return _build_sam3D(
        # encoder_embed_dim=768,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam3D_vit_b_ori(checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam3D_vit_b_pe2d(checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        pos_dim=2,
    )

def build_sam3D_vit_b_pe2d_att2d(checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        pos_dim=2,
        atten_dim=2,
    )

def build_sam3D_vit_b_att2d(checkpoint=None):
    return _build_sam3D_ori(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        atten_dim=2,
    )

sam_model_registry3D = {
    "default": build_sam3D_vit_b_ori,
    "vit_b_ori": build_sam3D_vit_b_ori,
    "vit_b_pe2d": build_sam3D_vit_b_pe2d,
    "vit_b_att2d": build_sam3D_vit_b_att2d,
    "vit_b_pe2d_att2d": build_sam3D_vit_b_pe2d_att2d,
}


if __name__ == "__main__":
    model = sam_model_registry3D['vit_b_pe2d_att2d']()
    in_tensor = torch.rand(1, 1, 128, 128, 128)
    out_tensor = model.image_encoder(in_tensor)
    print(out_tensor.shape)
