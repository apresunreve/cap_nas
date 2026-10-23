# Try to use standalone version first, fallback to timm if not available
try:
    from efficientnet_standalone import mobilenetv2_custom
    USE_STANDALONE = True
except ImportError:
    import timm
    USE_STANDALONE = False

def get_default_cfg():
    """Return default MBv2 cfg as list of (expansion, channels, num_blocks, stride) per stage."""
    return [
        (1, 16, 1, 1),   # stage 0: ds
        (6, 24, 2, 2),   # stage 1
        (6, 32, 3, 2),   # stage 2
        (6, 64, 4, 2),   # stage 3
        (6, 96, 3, 1),   # stage 4
        (6, 160, 3, 2),   # stage 5
        (6, 320, 1, 1),   # stage 6
    ]


def get_cfg_with_num_blocks_and_expansion(num_delta, expansion, base_cfg=None):
    """
    Return cfg with num_delta blocks applied and uniform expansion_ratio.
    num_delta: -1 = remove 1 block, -2 = remove 2 blocks, etc.
    expansion: uniform expansion ratio (float) for all IR stages.
    """
    if base_cfg is None:
        base_cfg = get_default_cfg()
    cfg = list(base_cfg)
    # Apply uniform expansion to all stages (t in (t,c,n,s))
    expansion = round(expansion, 1) if isinstance(expansion, float) else expansion
    for i in range(len(cfg)):
        t, c, n, s = cfg[i]
        cfg[i] = (expansion if i > 0 else t, c, n, s)  # stage 0 (ds) keeps t=1
    if num_delta == 0:
        return cfg
    # Remove blocks from stages with most blocks first
    to_remove = abs(num_delta)
    if num_delta > 0:
        # Add blocks - add to stage with fewest blocks (simplest: add to stage 3)
        for _ in range(num_delta):
            cfg[3] = (cfg[3][0], cfg[3][1], cfg[3][2] + 1, cfg[3][3])
        return cfg
    # num_delta < 0: remove blocks
    stage_order = sorted(range(len(cfg)), key=lambda i: -cfg[i][2])  # most blocks first
    for idx in stage_order:
        if to_remove <= 0:
            break
        t, c, n, s = cfg[idx]
        if n > 1:
            remove_here = min(to_remove, n - 1)
            cfg[idx] = (t, c, n - remove_here, s)
            to_remove -= remove_here
    return cfg


def get_default_arch_def():
    return [
             ['ds_r1_k3_s1_c16'],
             ['ir_r2_k3_s2_e6_c24'],
             ['ir_r3_k3_s2_e6_c32'],
             ['ir_r4_k3_s2_e6_c64'],
             ['ir_r3_k3_s1_e6_c96'],
             ['ir_r3_k3_s2_e6_c160'],
             ['ir_r1_k3_s1_e6_c320'],
         ]

def get_arch_def_from_cfg(cfg, img_size=224):
    _arch_def = get_default_arch_def()
    for stage_idx, (t, c, n, s) in enumerate(cfg):
        if stage_idx == 0:
            _arch_def[stage_idx] = [f'ds_r{n}_k3_s{s}_c{c}']
        elif img_size == 32 and stage_idx == 1: # for cifar
            _arch_def[stage_idx] = [f'ir_r{n}_k3_s1_e{t}_c{c}']
        else:
            _arch_def[stage_idx] = [f'ir_r{n}_k3_s{s}_e{t}_c{c}']
    return _arch_def

def mbv2_timm(cfg=None, num_classes=1000, wm=1.0, img_size=224, channel_divisor=4):
    """channel_divisor: round width_mult-scaled channels to multiples of this (default 4)."""
    if cfg is None:
        cfg = get_default_arch_def()
        arch_def = cfg
    else:
        arch_def = get_arch_def_from_cfg(cfg, img_size=img_size)
    
    if USE_STANDALONE:
        model = mobilenetv2_custom(pretrained=False, arch=arch_def, wm=wm, num_classes=num_classes, channel_divisor=channel_divisor)
    else:
        model = timm.create_model('mobilenetv2_custom', wm=wm, arch=arch_def, pretrained=False, num_classes=num_classes)
    return model

