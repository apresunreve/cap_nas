from typing import Any
import torch
import torch.nn as nn
import math
from itertools import product
import random
import sys
import json
import os
from thop import profile as thop_profile

conv_cfgs = set()
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        # record construct args
        self.building_cfg = [inp, oup, stride, expand_ratio]

        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        # self.add = nn.FloatFunctional()
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualQ(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualQ, self).__init__()

        # record construct args
        self.building_cfg = [inp, oup, stride, expand_ratio]

        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.add = nn.quantized.FloatFunctional()
    def forward(self, x):
        if self.identity:
            return self.add.add(x,  self.conv(x))
        else:
            return self.conv(x)

def get_ib_from_cfg(cfg):
    inp, oup, stride, expand_ratio = cfg
    return InvertedResidual(inp, oup, stride, expand_ratio)

def arr2conv2d(cfg):
    inc, outc, ks, strd, pad, grps = cfg
    return nn.Conv2d(in_channels=inc,
                     out_channels=outc,
                     kernel_size=ks,
                     stride=strd,
                     padding=pad,
                     groups=grps,
                     bias=False)

def conv2d2arr(m):
    return [m.in_channels,
            m.out_channels,
            m.kernel_size[0],
            m.stride[0],
            m.padding[0],
            m.groups]

def conv2d2tvmstr(m, in_shape):
    N, CI, H, W = list[Any](in_shape)
    CO = m.out_channels
    KH = m.kernel_size[0]
    KW = m.kernel_size[1]
    strd = m.stride[0]
    pad = m.padding[0]
    grps = m.groups
    line = f"{N}_{H}_{W}_{CO}_{CI}_{KH}_{KW}_{strd}_{pad}_{grps}"
    return line

def break_down_ib_to_convs(in_shape, layer_cfg, tvm_fmt=False):
    # build an ib
    res = [] # each entry is inp_shape, conv_cfg
    ib = get_ib_from_cfg(layer_cfg)
    inp = torch.randn(in_shape)
    x = inp
    for l in ib.modules():
        # f"{N}_{H}_{W}_{CO}_{CI}_{KH}_{KW}_{strides[0]}_{padding[0]}_{grps}"
        if isinstance(l, nn.Conv2d):
            if tvm_fmt:
                l.register_forward_hook(
                    lambda l, input, output:
                    res.append(conv2d2tvmstr(l, input)))
                    # res.append(conv2d2tvmstr(l, input[0].shape)))
            else:
                l.register_forward_hook(
                    lambda l, input, output:
                    res.append((tuple(list(input[0].shape)), tuple(conv2d2arr(l)))))
    ib(x)
    return res

_default_cfg = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2], # = 2 x mbconv(6,24,2)
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

def expand_wm(chn):
    return [chn//2,
            chn//1.5,
            chn//1.25,
            chn,
            int(chn*1.25),
            int(chn*1.5),
            chn * 2
            ]

def _expand_chn(chn):
    if chn >= 160:
        m = 8
        # m=16
    elif chn >= 80:
        m = 8
    else:
        m = 4

    chns=[chn]
    res = []
    for c in chns:
        res.extend([
            # chn - 2*m,
            chn - m,
            chn,
            chn + m,
            # chn + 2*m
            ])
    return res

def enumerate_arch(base_cfg=_default_cfg, expand_chn=False):
    results = []
    _num_blocks_seq = [1,2,3,4,3,3,1]
    for i, layer_cfg in enumerate(base_cfg):
        layer_choices = []
        expansion, out_planes, num_blocks, stride = layer_cfg
        _expansion_space = [1,2,3,4,5,6]
        # _expansion_space = [1,2,4,6]
        if expand_chn:
            _out_planes_space = _expand_chn(out_planes)
        else:
            _out_planes_space = [out_planes]
        _stride_space = [stride]

        # _nb = _num_blocks_seq[i]
        # for _op in _out_planes_space: # NOTE currently not varying out_planes, but change width_multiplier instead
        #     for _exp in _expansion_space:
        #         for _strd in _stride_space:
        #             layer_choices.append(tuple([_exp, _op, _nb, _strd]))

        _num_blocks_space = [1,2,3,4,5,6]
        for _op in _out_planes_space: # NOTE currently not varying out_planes, but change width_multiplier instead
            for _exp in _expansion_space:
                for _nb in _num_blocks_space:
                    for _strd in _stride_space:
                        layer_choices.append(tuple([_exp, _op, _nb, _strd]))
        results.append(layer_choices)

    return results

class MobileNetV2(nn.Module):
    def __init__(self,  num_classes=1000, width_mult=1., cfg=None, quantize=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        if cfg is None:
            self.cfg = _default_cfg
        else:
            self.cfg = cfg

        self.quantize = quantize
        if self.quantize:
            self.quant = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        self.blk_params = []
        block = InvertedResidual
        for t, c, n, s in self.cfg:
            #output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            output_channel = _make_divisible(c * width_mult, 4) #if width_mult == 0.1 else 8)
            p = 0
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                for pr in layers[-1].parameters():
                    p += pr.numel()
                input_channel = output_channel
            self.blk_params.append(p)

        # ref = sum(self.blk_params[1:]) / 5573088
        # ps = [p/sum(self.blk_params[1:]) * ref  for p in self.blk_params[1:]]

        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.quantize:
            x = self.dequant(x)
        return x

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        stage_features_list = []
        image_size = x.shape[2]
        output = x

        for block_id, the_block in enumerate(self.features):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
                # TODO
            pass
        pass
        output = self.conv(output)
        output = self.avgpool(output)
        output = output.view(output.size(0),-1)
        logit = self.classifier(output)
        return stage_features_list, logit

    def forward_pre_GAP(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def forward_feat(self, x):
        x = self.features(x)
        return x

    def extract_cell_features(self, inputs):
        cell_features = []
        feature = inputs
        if feature.requires_grad:
            feature.retain_grad()

        for i, cell in enumerate(self.features):
            feature = cell(feature)
            if feature.requires_grad:
                feature.retain_grad()
            cell_features.append(feature)

        return cell_features

    def get_input_shapes_and_ib_cfgs(self, input_image_size=224):
        x = torch.randn(1,3,input_image_size, input_image_size)
        # features is conv + ibs
        res = []
        print(len(self.features))
        print([type(l) for l in self.features])
       
        for l in self.features:
            if isinstance(l, InvertedResidual):
                cur_res =  (list(x.shape), l.building_cfg)
                # if len(res)>0 and cur_res == res[-1]:
                #     continue
                res.append(cur_res)
            x = l(x)
        return res

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

_default_cfg_imgnt = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2], # = 2 x mbconv(6,24,2)
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

def round_to_multiple_of_4(n):
    return ((n + 3) // 4) * 4

def get_chn_space():
    # hardcoded 1st and 2nd layer
    chn_space = [ 64, 96, 160, 320]
    _res = [[16], [16, 20, 24], [16, 20, 24, 28, 32]]
    #_res = [[8, 12, 16], [8, 12, 16, 20, 24]]
    for chn in chn_space:
        _res.append(list(range(chn//4,chn+3,  round_to_multiple_of_4(chn//8))))
        # print(list(range(chn//4,chn+1, chn//8)))
    return _res

def get_cfg_from_chn(chn_space, exp_space=None):
    if exp_space is None:
        exp_space = [1, 6, 6, 6, 6, 6]
    cfg = [
            # t, c, n, s
            [1,  chn_space[0], 1, 1],
            [exp_space[0],  chn_space[1], 2, 2], # = 2 x mbconv(6,24,2)
            [exp_space[1],  chn_space[2], 3, 2],
            [exp_space[2],  chn_space[3], 4, 2],
            [exp_space[3],  chn_space[4], 3, 1],
            [exp_space[4],  chn_space[5], 3, 2],
            [exp_space[5],  chn_space[6], 1, 1],
        ]
    return cfg

def enumerate_all_combinations(arrays):
    return [list(comb) for comb in product(*arrays)]

def enumerate_mobilenet_arch():
    res = []
    chn_space = get_chn_space()
    # print(chn_space)
    from monotonic_combinations import find_monotonic_combinations
    res = find_monotonic_combinations(chn_space)
    return res

def conv2d2tvmstr(m, input):
    if isinstance(input, tuple):
        in_shape = input[0].shape
    else:
        in_shape = input.shape

    N, CI, H, W = list(in_shape)
    CO = m.out_channels
    KH = m.kernel_size[0]
    KW = m.kernel_size[1]
    S = m.stride[0]
    P = m.padding[0]
    G = m.groups
    line = f"{N}_{H}_{W}_{CO}_{CI}_{KH}_{KW}_{S}_{P}_{G}"
    return line


def record_conv_hook(layer_name):
  def hook(module, input, output):
    # conv_cfgs[layer_name] = conv2d2tvmstr(module, input)
    tvmstr = conv2d2tvmstr(module, input)
    # print(tvmstr)
    conv_cfgs.add(tvmstr)
  return hook








def get_conv_from_model(model, input_shape=(1,3,224,224)):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(record_conv_hook(n))
    model(torch.randn(input_shape).cuda())
    del model
    return


def get_conv_list_from_model(model, input_shape=(1,3,224,224)):
    conv_cfgs_list = []
    def record_conv_hook_list(layer_name):
        def hook(module, input, output):
            tvmstr = conv2d2tvmstr(module, input)
            conv_cfgs_list.append(tvmstr)
        return hook
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(record_conv_hook_list(n))
    model(torch.randn(input_shape))#.cuda())
    del model
    return conv_cfgs_list


def sample_arch(space):
    space = enumerate_arch(expand_chn=True)[1:]
    _first_stage = (1,16,1,1)

    res = [_first_stage]
    for i in space:
        res.append(random.choice(i))
    return res

def estimate_latency(arch, lut, input_shape=None):
    model = MobileNetV2(cfg=arch)
    if input_shape is not None:
        conv_cfgs_list = get_conv_list_from_model(model, input_shape)
    else:
        conv_cfgs_list = get_conv_list_from_model(model)

    total_latency = 0
    missing_cfgs = []
    for conv_cfg in conv_cfgs_list:
        try:
            total_latency += lut[conv_cfg]
        except:
            missing_cfgs.append(conv_cfg)
            total_latency += 0
    # print(f'1st and last cfgs: {conv_cfgs_list[0]},{conv_cfgs_list[-1]}')
    # print(f'missing cfgs: {missing_cfgs}')
    with open('missing_cfgs.txt', 'a') as f:
        f.writelines([f'{cfg}\n' for cfg in missing_cfgs])
    return total_latency


STAGE_INP_SIZE = [112, 112, 56, 28, 14, 14, 7]

# STAGE_CHNS = [[32], [12,16,20], [16, 20, 24], [20, 24, 28], [60,64,68], [88,96,104], [144,160,176], ]#[304,320,336]]
STAGE_CHNS = [[32], [12,16,20], [16, 20, 24], [20, 24, 28], [60,64,68], [88,96,104], [152,160,168], ]#[304,320,336]]

stage_in_out_chns = [
    [(32), (8, 12, 16, 20, 24)],
    [(8, 12, 16, 20, 24), (16, 20, 24, 28, 32)],
    [(16, 20, 24, 28, 32), (24, 28, 32, 36, 40)],
    [(24, 28, 32, 36, 40), (56,60,64,68,72)],
    [(56,60,64,68,72), (80,88,96,104,112)],
    [(80,88,96,104,112), (128, 144, 160, 176, 192)],
    [(128, 144, 160, 176, 192), (288,304, 320, 336,352)],
]

stage_in_chns = [
    [32],
    [12,16,20],
    [20,24,28],
    [28,32,36],
    [60,64,68],
    [88,96,104],
    [144,160,176],
]


def get_lut(lut_path):
    with open(lut_path, 'rb') as f:
        lut = pickle.load(f)
    if 'orin' in lut_path:
        lut ['1_224_224_32_3_3_3_2_1_1'] = 8.6201e-05
        lut ['1_7_7_1280_304_1_1_1_0_1'] = 0.000174853
        lut['1_7_7_1280_320_1_1_1_0_1']= 0.000148759
        lut['1_7_7_1280_336_1_1_1_0_1']= 0.000129201
    elif 'a57' in lut_path:
        lut ['1_224_224_32_3_3_3_2_1_1'] = 8.6201e-05
        lut ['1_7_7_1280_304_1_1_1_0_1'] = 0.002796688
        lut['1_7_7_1280_320_1_1_1_0_1']= 0.00319692
        lut['1_7_7_1280_336_1_1_1_0_1']= 0.00232884
    with open(lut_path, 'wb') as f:
        pickle.dump(lut, f)
    return lut


def get_latencies(lut_path):
    lut = get_lut(lut_path)
    space = enumerate_arch(expand_chn=True)[1:]
    _first_stage = (1,16,1,1)
    i = 0
    while True:
        arch = sample_arch(space)
        latency = estimate_latency(arch, lut)
        print(f'arch: {arch}, latency: {latency}')
        i += 1
        if i > 100:
            break

def estimate_stage_latency(layers, lut, input_shape=None):
    if input_shape is not None:
        conv_cfgs_list = get_conv_list_from_model(layers, input_shape)
    else:
        conv_cfgs_list = get_conv_list_from_model(layers)
    total_latency = 0
    
    for conv_cfg in conv_cfgs_list:
        if conv_cfg not in lut:
            print(f'conv_cfg not in lut: {conv_cfg}')
            with open(f'missing_2.txt', 'a') as f:
                f.writelines([f'{conv_cfg}\n'])
            continue
        total_latency += lut[conv_cfg]
    return total_latency

def compute_params_and_flops(layers, input_shape):
    dummy = torch.randn(input_shape)
    macs, params = thop_profile(layers, inputs=(dummy,), verbose=False)
    flops = macs * 2
    return params, flops

def build_stage(stage_cfg, input_channel, block=InvertedResidual):
    t,c,n,s = stage_cfg
    output_channel = _make_divisible(c, 4)
    layers = []
    for i in range(n):
        layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
        input_channel = output_channel
    layers = nn.Sequential(*layers)
    return layers


if __name__ == "__main__":
    import pickle
    lut_path = sys.argv[1]
    lut = get_lut(lut_path)
    space = enumerate_arch(expand_chn=True)#[1:]
    #space = enumerate_arch(expand_chn=True)[1:]
    #_first_stage = (1,16,1,1)
    myspace =space# [[_first_stage]] + space


    block = InvertedResidual
    missing  = set()
    all_lines = []
    for stage_idx, stage_cfgs in enumerate(myspace):
        print(f'stage_idx: {stage_idx}, length of stage_cfgs: {len(stage_cfgs)}')

        # instantiate block
        latency_dict = {}
        for stage_cfg in stage_cfgs:
            t,c,n,s = stage_cfg
            output_channel = _make_divisible(c, 4)
            for input_channel in stage_in_chns[stage_idx]:
                layers = build_stage(stage_cfg, input_channel, block)
                input_shape = (1, input_channel, STAGE_INP_SIZE[stage_idx], STAGE_INP_SIZE[stage_idx])

                # store latency in a dictionary
                # print(f'input_shape: {input_shape}, stage_cfg: {stage_cfg}, latency: {latency}')
                new_key = '_'.join([str(i) for i in stage_cfg]) + '-' + '_'.join([str(i) for i in input_shape])
                params, flops = compute_params_and_flops(layers, input_shape)
                latency = estimate_stage_latency(layers, lut, input_shape)
                latency_dict[new_key] = (latency, params, flops)
        with open(f'latency_dict_{stage_idx}.pkl', 'wb') as f:
            pickle.dump(latency_dict, f)
        lines = []
        for k, v in latency_dict.items():
            k0, k1 = k.split('-')
            latency, params, flops = v
            l = f'{stage_idx},{k0},{k1},{latency},{params},{flops}\n'
            lines.append(l)
            all_lines.append(l)
        with open(f'latency_dict_{stage_idx}.csv', 'w') as f:
            f.writelines(lines)


    with open(f'latency_dict_summary_3.csv', 'w') as f:
        f.writelines(all_lines)

    print('written to latency_dict_summary_3.csv')
