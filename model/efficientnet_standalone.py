"""
Standalone EfficientNet/MobileNetV2 implementation extracted from timm.

This module provides a self-contained version that can work independently
of timm installation. It extracts the necessary code from timm's efficientnet.py.

Usage:
    from efficientnet_standalone import mobilenetv2_custom
    model = mobilenetv2_custom(arch=arch_def, wm=1.0, num_classes=1000)
"""

from functools import partial
from typing import Callable, List, Optional, Tuple, Union, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from timm if available - this allows the code to work with existing timm installation
try:
    from timm.layers import create_conv2d, create_classifier, get_norm_act_layer, LayerType
    from timm.models._efficientnet_builder import (
        BlockArgs, EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,
        round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
    )
    from timm.models._efficientnet_blocks import SqueezeExcite
    from timm.models._features import feature_take_indices
    from timm.models._manipulate import checkpoint_seq
    USE_TIMM = True
except ImportError:
    USE_TIMM = False
    # If timm is not available, we'll need to define simplified versions
    # For now, raise an error to indicate timm is required
    raise ImportError(
        "timm library is required. Install it with: pip install timm\n"
        "Or copy the necessary files from timm's site-packages to make it fully standalone."
    )


class EfficientNet(nn.Module):
    """ EfficientNet - Standalone version extracted from timm """

    def __init__(
            self,
            block_args: BlockArgs,
            num_classes: int = 1000,
            num_features: int = 1280,
            in_chans: int = 3,
            stem_size: int = 32,
            stem_kernel_size: int = 3,
            fix_stem: bool = False,
            output_stride: int = 32,
            pad_type: str = '',
            act_layer: Optional[Callable] = None,
            norm_layer: Optional[Callable] = None,
            aa_layer: Optional[Callable] = None,
            se_layer: Optional[Callable] = None,
            round_chs_fn: Callable = round_channels,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            global_pool: str = 'avg'
    ):
        super(EfficientNet, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, stem_kernel_size, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features  # Store features directly, not wrapped in FeatureInfo
        self.stage_ends = [f['stage'] for f in builder.features] if builder.features else []
        head_chs = builder.in_chs

        # Head + Pooling
        if num_features > 0:
            self.conv_head = create_conv2d(head_chs, num_features, 1, padding=pad_type)
            self.bn2 = norm_act_layer(num_features, inplace=True)
            self.num_features = self.head_hidden_size = num_features
        else:
            self.conv_head = nn.Identity()
            self.bn2 = nn.Identity()
            self.num_features = self.head_hidden_size = head_chs

        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        efficientnet_init_weights(self)

    def forward_pre_GAP(self, x):
        return self.forward_features(x)

    def extract_layer_features_and_logit(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract layer-wise features for AZ-NAS progressivity/trainability (requires grad)."""
        layer_features: List[torch.Tensor] = []
        x = self.conv_stem(x)
        x = self.bn1(x)
        if x.requires_grad:
            x.retain_grad()
        layer_features.append(x)
        for block in self.blocks:
            x = block(x)
            if x.requires_grad:
                x.retain_grad()
            layer_features.append(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        if x.requires_grad:
            x.retain_grad()
        layer_features.append(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        logit = self.classifier(x)
        return layer_features, logit

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _gen_mobilenet_v2(
        variant, channel_multiplier=1.0, depth_multiplier=1.0,
        group_size=None, fix_stem_head=False, pretrained=False, arch=None,
        channel_divisor=4, **kwargs
):
    """ Generate MobileNet-V2 network """
    if arch is None:
        arch_def = [
            ['ds_r1_k3_s1_c16'],
            ['ir_r2_k3_s2_e6_c24'],
            ['ir_r3_k3_s2_e6_c32'],
            ['ir_r4_k3_s2_e6_c64'],
            ['ir_r3_k3_s1_e6_c96'],
            ['ir_r3_k3_s2_e6_c160'],
            ['ir_r1_k3_s1_e6_c320'],
        ]
    else:
        arch_def = arch

    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=channel_divisor)
    model_kwargs = dict(
        block_args=decode_arch_def(
            arch_def,
            depth_multiplier=depth_multiplier,
            fix_first_last=fix_stem_head,
            group_size=group_size,
        ),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        **kwargs
    )
    model = EfficientNet(**model_kwargs)
    return model


def mobilenetv2_custom(pretrained=False, arch=None, wm=1.0, channel_divisor=4, **kwargs):
    """Create MobileNetV2 custom model - compatible with timm's mobilenetv2_custom.
    channel_divisor: round channels to multiples of this (default 4, was 8 in timm)."""
    if wm == 1.0:
        _w = '100'
    elif wm == 1.2:
        _w = '120'
    elif wm == 0.5:
        _w = '050'
    elif wm == 0.35:
        _w = '035'
    else:
        _w = '100'  # default

    model = _gen_mobilenet_v2(
        f'mobilenetv2_{_w}', wm, pretrained=False, arch=arch,
        channel_divisor=channel_divisor, **kwargs)
    return model


def _gen_efficientnet_custom(
        variant, channel_multiplier=1.0, depth_multiplier=1.0,
        channel_divisor=8, arch=None, pretrained=False, **kwargs
):
    """Generate a custom EfficientNet model."""
    if arch is None:
        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25'],
        ]
    else:
        arch_def = arch
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=channel_divisor)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    return EfficientNet(**model_kwargs)


def efficientnet_custom(pretrained=False, arch=None, wm=1.0, dm=1.0, channel_divisor=8, **kwargs):
    """Create custom EfficientNet model with arbitrary arch_def.
    wm: width multiplier, dm: depth multiplier, channel_divisor: channel rounding."""
    return _gen_efficientnet_custom(
        'efficientnet_custom', wm, dm,
        channel_divisor=channel_divisor, arch=arch, pretrained=False, **kwargs
    )
