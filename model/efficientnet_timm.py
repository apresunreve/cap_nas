from efficientnet_standalone import efficientnet_custom


def get_default_cfg():
    """Return default EfficientNet-B0 cfg as list of (expansion, channels, num_blocks, stride, kernel_size, se_ratio) per stage."""
    return [
        (1,  16, 1, 1, 3, 0.25),  # stage 0: ds
        (6,  24, 2, 2, 3, 0.25),  # stage 1
        (6,  40, 2, 2, 5, 0.25),  # stage 2
        (6,  80, 3, 2, 3, 0.25),  # stage 3
        (6, 112, 3, 1, 5, 0.25),  # stage 4
        (6, 192, 4, 2, 5, 0.25),  # stage 5
        (6, 320, 1, 1, 3, 0.25),  # stage 6
    ]


def get_cfg_with_num_blocks_and_expansion(num_delta, expansion, base_cfg=None):
    """
    Apply num_delta uniformly to each IR stage (min 1 block) and set uniform expansion ratio.
    num_delta: e.g. -1 removes 1 block from every IR stage, -2 removes 2.
    expansion: uniform expansion ratio for all IR stages (stage 0 ds keeps its original t=1).
    """
    if base_cfg is None:
        base_cfg = get_default_cfg()
    cfg = list(base_cfg)
    for i in range(len(cfg)):
        t, c, n, s, k, se = cfg[i]
        if i > 0:  # IR stages only; stage 0 (ds) unchanged
            cfg[i] = (expansion, c, max(1, n + num_delta), s, k, se)
    return cfg


def get_default_arch_def():
    return [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]


def get_arch_def_from_cfg(cfg, img_size=224):
    _arch_def = get_default_arch_def()
    for stage_idx, (t, c, n, s, k, se) in enumerate(cfg):
        se_str = f'_se{se}' if se > 0 else ''
        if stage_idx == 0:
            _arch_def[stage_idx] = [f'ds_r{n}_k{k}_s{s}_e{t}_c{c}{se_str}']
        elif img_size == 32 and stage_idx == 1:  # for cifar
            _arch_def[stage_idx] = [f'ir_r{n}_k{k}_s1_e{t}_c{c}{se_str}']
        else:
            _arch_def[stage_idx] = [f'ir_r{n}_k{k}_s{s}_e{t}_c{c}{se_str}']
    return _arch_def


def parse_effnet_cfg_from_str(cfg_str):
    """Parse inline cfg_str '1_16_1_1_3_0.25|6_24_2_2_3_0.25|...' → list of (t,c,n,s,k,se)."""
    blocks = [b.strip() for b in cfg_str.strip().replace('|', ',').split(',') if b.strip()]
    cfg = []
    for b in blocks:
        parts = b.split('_')
        assert len(parts) == 6, f'Expected 6 values (t_c_n_s_k_se), got {parts} from "{b}"'
        t, se = float(parts[0]), float(parts[5])
        c, n, s, k = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        cfg.append((t, c, n, s, k, se))
    return cfg


def efficientnet_timm(cfg=None, num_classes=1000, wm=1.0, dm=1.0, img_size=224, channel_divisor=4):
    """Create a custom EfficientNet model.

    cfg: list of (expansion, channels, num_blocks, stride, kernel_size, se_ratio) per stage.
         If None, defaults to EfficientNet-B0.
    wm: width multiplier
    dm: depth multiplier (scales num_blocks per stage via decode_arch_def)
    channel_divisor: round wm-scaled channels to multiples of this (default 4)
    """
    if cfg is None:
        arch_def = get_default_arch_def()
    else:
        arch_def = get_arch_def_from_cfg(cfg, img_size=img_size)

    model = efficientnet_custom(
        pretrained=False, arch=arch_def, wm=wm, dm=dm,
        num_classes=num_classes, channel_divisor=channel_divisor,
    )
    return model
