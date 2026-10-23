import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
_model_root = os.path.join(PROJ_ROOT, "model")
if _model_root not in sys.path:
    sys.path.insert(0, _model_root)
_utils_root = os.path.join(PROJ_ROOT, "utils")
if _utils_root not in sys.path:
    sys.path.insert(0, _utils_root)

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

import PlainNet
def tensor_variance(x):
    return x.var(unbiased=False)

@torch.no_grad()
def _flatten_features(feat):
    if not isinstance(feat, torch.Tensor) or feat.dim() != 4:
        return None
    b, c, h, w = feat.size()
    return feat.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)

@torch.no_grad()
def block_expressivity(block, input_tensor, eps=1e-8):
    block.eval()
    out = block(input_tensor)
    feat = _flatten_features(out)
    if feat is None:
        return float("-inf")
    feat = feat - feat.mean(dim=0, keepdim=True)
    sigma = torch.mm(feat.transpose(1, 0), feat) / max(feat.size(0), 1)
    try:
        s = torch.linalg.eigvalsh(sigma)
    except RuntimeError:
        jitter = eps * torch.eye(sigma.size(0), device=sigma.device, dtype=sigma.dtype)
        try:
            s = torch.linalg.eigvalsh(sigma + jitter)
        except RuntimeError:
            sv = torch.linalg.svdvals(feat)
            s = (sv * sv) / max(feat.size(0), 1)
    if s.numel() == 0 or torch.all(s <= 0):
        return float("-inf")
    s = torch.clamp(s, min=0)
    prob_s = s / (s.sum() + eps)
    score = (-prob_s) * torch.log(prob_s + eps)
    return float(score.sum().item())

@torch.no_grad()
def block_effective_rank(block, input_tensor, eps=1e-8):
    block.eval()
    out = block(input_tensor)
    feat = _flatten_features(out)
    if feat is None:
        return float("-inf")
    feat = feat - feat.mean(dim=0, keepdim=True)
    sigma = torch.mm(feat.transpose(1, 0), feat) / max(feat.size(0), 1)
    tr = torch.trace(sigma)
    tr2 = torch.trace(torch.mm(sigma, sigma))
    if tr2 <= 0:
        return float("-inf")
    return float(((tr * tr) / (tr2 + eps)).item())

@torch.no_grad()
def block_activation_sparsity(block, input_tensor, eps=1e-6):
    block.eval()
    out = block(input_tensor)
    if not isinstance(out, torch.Tensor):
        return float("-inf")
    active = (out.abs() > eps).float().mean().item()
    return float(active)

@torch.no_grad()
def block_orthogonality(block, input_tensor, eps=1e-8):
    block.eval()
    out = block(input_tensor)
    feat = _flatten_features(out)
    if feat is None:
        return float("-inf")
    feat = feat - feat.mean(dim=0, keepdim=True)
    sigma = torch.mm(feat.transpose(1, 0), feat) / max(feat.size(0), 1)
    diag = torch.sqrt(torch.clamp(torch.diag(sigma), min=eps))
    norm = torch.outer(diag, diag)
    corr = sigma / (norm + eps)
    c = corr.size(0)
    if c <= 1:
        return float("-inf")
    off_diag = corr - torch.diag(torch.diag(corr))
    score = 1.0 - off_diag.abs().sum().item() / (c * (c - 1))
    return float(score)

@torch.no_grad()
def block_signal_preservation(block, input_tensor, eps=1e-8):
    block.eval()
    out = block(input_tensor)
    if not isinstance(out, torch.Tensor):
        return float("-inf")
    var_in = tensor_variance(input_tensor)
    var_out = tensor_variance(out)
    ratio = var_out / (var_in + eps)
    return float(torch.exp(-torch.abs(torch.log(ratio + eps))).item())

@torch.no_grad()
def block_noise_sensitivity(block, input_tensor, noise_std=1e-3, eps=1e-8):
    block.eval()
    noise = torch.randn_like(input_tensor) * noise_std
    out1 = block(input_tensor)
    out2 = block(input_tensor + noise)
    if not isinstance(out1, torch.Tensor) or not isinstance(out2, torch.Tensor):
        return float("-inf")
    num = torch.norm(out2 - out1)
    den = torch.norm(noise) + eps
    return float((num / den).item())


@torch.no_grad()
def block_zen_score(block, input_tensor, mixup_gamma=1e-2, repeat=1, eps=1e-8):
    """
    Compute ZEN score for a single block (mixup sensitivity + BN scaling).
    Higher score indicates higher expressivity/sensitivity to input perturbation.
    """
    block.eval()
    device = input_tensor.device
    dtype = input_tensor.dtype
    shape = tuple(input_tensor.shape)

    scores = []
    for _ in range(repeat):
        network_weight_gaussian_init(block)
        x1 = torch.randn(*shape, device=device, dtype=dtype)
        x2 = torch.randn(*shape, device=device, dtype=dtype)
        mix = x1 + mixup_gamma * x2

        out1 = block(x1)
        out2 = block(mix)

        if not isinstance(out1, torch.Tensor) or not isinstance(out2, torch.Tensor):
            return float("-inf")

        diff = (out1 - out2).float()
        score = torch.mean(torch.sum(torch.abs(diff), dim=[1, 2, 3]))
        score = torch.clamp(score, min=eps)

        log_bn_scaling = 0.0
        for m in block.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_var = m.running_var.float()
                bn_scaling = torch.sqrt(torch.mean(bn_var))
                bn_scaling = torch.clamp(bn_scaling, min=eps)
                log_bn_scaling += torch.log(bn_scaling)

        scores.append(float(torch.log(score) + log_bn_scaling))

    return sum(scores) / len(scores)


def aggregate_block_zen_scores(block_scores, method="sum"):
    """
    Aggregate per-block ZEN scores into a single network-level score.

    Args:
        block_scores: List of block_zen_score values (one per block).
        method: "sum" (default, like LBC - deeper networks score higher) or "mean"
                (normalize by depth - fair comparison across different depths).

    Returns:
        Scalar aggregated score.
    """
    if not block_scores:
        return float("-inf")
    valid = [s for s in block_scores if s != float("-inf") and not math.isnan(s)]
    if not valid:
        return float("-inf")
    if method == "sum":
        return sum(valid)
    elif method == "mean":
        return sum(valid) / len(valid)
    else:
        raise ValueError(f"method must be 'sum' or 'mean', got {method!r}")


@torch.no_grad()
def compute_network_zen_from_blocks(
    struct_str,
    input_shape,
    device="cuda:0",
    aggregate="sum",
    mixup_gamma=1e-2,
    repeat=1,
):
    """
    Compute block_zen_score for each block in a network and aggregate.

    Args:
        struct_str: PlainNet structure string (e.g. "SuperConvK3BNRELU(...)...")
        input_shape: (N,C,H,W) or parseable string/list.
        device: torch device.
        aggregate: "sum" or "mean" for combining block scores.
        mixup_gamma: ZEN mixup factor.
        repeat: number of trials per block (averaged).

    Returns:
        (aggregated_score, block_scores_list)
        block_scores_list: list of dicts with block_idx, block_str, block_zen_score, etc.
    """
    shape = _parse_input_shape(input_shape)
    x = torch.randn(*shape, device=device)
    block_list = PlainNet.create_netblock_list_from_str(struct_str, no_create=False)
    block_scores_list = []
    for idx, block in enumerate(block_list):
        block = block.to(device)
        network_weight_gaussian_init(block)
        block.eval()
        zen = block_zen_score(block, x, mixup_gamma=mixup_gamma, repeat=repeat)
        block_scores_list.append({
            "block_idx": idx,
            "block_str": str(block),
            "block_zen_score": zen,
        })
        out = block(x)
        if isinstance(out, torch.Tensor):
            x = out
        else:
            break
    zen_scores = [d["block_zen_score"] for d in block_scores_list]
    aggregated = aggregate_block_zen_scores(zen_scores, method=aggregate)
    return aggregated, block_scores_list


@torch.no_grad()
def compute_block_scores(block, input_tensor):
    return {
        "block_expressivity": block_expressivity(block, input_tensor),
        "block_effective_rank": block_effective_rank(block, input_tensor),
        "block_activation_sparsity": block_activation_sparsity(block, input_tensor),
        "block_orthogonality": block_orthogonality(block, input_tensor),
        "block_signal_preservation": block_signal_preservation(block, input_tensor),
        "block_noise_sensitivity": block_noise_sensitivity(block, input_tensor),
        "block_zen_score": block_zen_score(block, input_tensor),
    }


def _parse_input_shape(value):
    if isinstance(value, (list, tuple)):
        shape = [int(v) for v in value]
    else:
        cleaned = str(value).strip().strip("()[]")
        shape = [int(x) for x in cleaned.split(",") if x.strip()]
    if len(shape) != 4:
        raise ValueError(f"input_shape must have 4 values, got {shape}")
    return tuple(shape)


@torch.no_grad()
def _block_scores_from_output(
    block, input_tensor, output_tensor, noise_std=1e-3, eps=1e-8
):
    scores = {}
    feat = _flatten_features(output_tensor)
    if feat is None:
        scores["block_expressivity"] = float("-inf")
        scores["block_effective_rank"] = float("-inf")
        scores["block_orthogonality"] = float("-inf")
    else:
        feat = feat - feat.mean(dim=0, keepdim=True)
        sigma = torch.mm(feat.transpose(1, 0), feat) / max(feat.size(0), 1)
        try:
            s = torch.linalg.eigvalsh(sigma)
        except RuntimeError:
            jitter = eps * torch.eye(
                sigma.size(0), device=sigma.device, dtype=sigma.dtype
            )
            try:
                s = torch.linalg.eigvalsh(sigma + jitter)
            except RuntimeError:
                sv = torch.linalg.svdvals(feat)
                s = (sv * sv) / max(feat.size(0), 1)
        if s.numel() == 0 or torch.all(s <= 0):
            scores["block_expressivity"] = float("-inf")
        else:
            s = torch.clamp(s, min=0)
            prob_s = s / (s.sum() + eps)
            score = (-prob_s) * torch.log(prob_s + eps)
            scores["block_expressivity"] = float(score.sum().item())

        tr = torch.trace(sigma)
        tr2 = torch.trace(torch.mm(sigma, sigma))
        if tr2 <= 0:
            scores["block_effective_rank"] = float("-inf")
        else:
            scores["block_effective_rank"] = float(((tr * tr) / (tr2 + eps)).item())

        diag = torch.sqrt(torch.clamp(torch.diag(sigma), min=eps))
        norm = torch.outer(diag, diag)
        corr = sigma / (norm + eps)
        c = corr.size(0)
        if c <= 1:
            scores["block_orthogonality"] = float("-inf")
        else:
            off_diag = corr - torch.diag(torch.diag(corr))
            ortho = 1.0 - off_diag.abs().sum().item() / (c * (c - 1))
            scores["block_orthogonality"] = float(ortho)

    if not isinstance(output_tensor, torch.Tensor):
        scores["block_activation_sparsity"] = float("-inf")
        scores["block_signal_preservation"] = float("-inf")
        scores["block_noise_sensitivity"] = float("-inf")
        return scores

    scores["block_activation_sparsity"] = float(
        (output_tensor.abs() > 1e-6).float().mean().item()
    )

    var_in = tensor_variance(input_tensor)
    var_out = tensor_variance(output_tensor)
    ratio = var_out / (var_in + eps)
    scores["block_signal_preservation"] = float(
        torch.exp(-torch.abs(torch.log(ratio + eps))).item()
    )

    noise = torch.randn_like(input_tensor) * noise_std
    out2 = block(input_tensor + noise)
    if isinstance(out2, torch.Tensor):
        num = torch.norm(out2 - output_tensor)
        den = torch.norm(noise) + eps
        scores["block_noise_sensitivity"] = float((num / den).item())
    else:
        scores["block_noise_sensitivity"] = float("-inf")
    return scores


@torch.no_grad()
def compute_block_scores_for_struct(
    struct_str,
    input_shape,
    device="cuda:0",
    init_weights=True,
    noise_std=1e-3,
):
    shape = _parse_input_shape(input_shape)
    x = torch.randn(*shape, device=device)
    block_list = PlainNet.create_netblock_list_from_str(struct_str, no_create=False)
    results = []
    for idx, block in enumerate(block_list):
        block = block.to(device)
        if init_weights:
            network_weight_gaussian_init(block)
        block.eval()
        out = block(x)
        scores = _block_scores_from_output(
            block,
            x,
            out,
            noise_std=noise_std,
        )
        results.append(
            {
                "block_idx": idx,
                "block_str": str(block),
                "input_shape": list(x.shape),
                "output_shape": list(out.shape) if isinstance(out, torch.Tensor) else [],
                **scores,
            }
        )
        if isinstance(out, torch.Tensor):
            x = out
        else:
            break
    return results

@torch.no_grad()
def compute_expressivity(model, input_tensor, eps=1e-8):
    model.eval()
    layer_features = []
    hooks = []

    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            layer_features.append(out.detach())

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_fn))

    model(input_tensor)

    for h in hooks:
        h.remove()

    expressivity_scores = []
    for feat in layer_features:
        b, c, h, w = feat.size()
        feat = feat.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        feat = feat - feat.mean(dim=0, keepdim=True)
        sigma = torch.mm(feat.transpose(1, 0), feat) / max(feat.size(0), 1)
        # Stabilize eigendecomposition for near-singular covariance matrices.
        try:
            s = torch.linalg.eigvalsh(sigma)
        except RuntimeError:
            jitter = eps * torch.eye(sigma.size(0), device=sigma.device, dtype=sigma.dtype)
            try:
                s = torch.linalg.eigvalsh(sigma + jitter)
            except RuntimeError:
                # Fall back to singular values of centered features (s^2 / N).
                sv = torch.linalg.svdvals(feat)
                s = (sv * sv) / max(feat.size(0), 1)
        if s.numel() == 0 or torch.all(s <= 0):
            expressivity_scores.append(float("-inf"))
            continue
        s = torch.clamp(s, min=0)
        prob_s = s / (s.sum() + eps)
        score = (-prob_s) * torch.log(prob_s + eps)
        expressivity_scores.append(score.sum().item())

    if len(expressivity_scores) < 2:
        progressivity = float("-inf")
    else:
        diffs = torch.tensor(expressivity_scores[1:]) - torch.tensor(expressivity_scores[:-1])
        progressivity = diffs.min().item()

    expressivity = float(torch.tensor(expressivity_scores).sum().item()) if expressivity_scores else float("-inf")
    return expressivity, progressivity

def is_expansion_conv(conv):
    return (
        isinstance(conv, nn.Conv2d)
        and conv.kernel_size == (1, 1)
        and conv.groups == 1
        and conv.out_channels > conv.in_channels
    )

@torch.no_grad()
def compute_EUS(model, input_tensor, eps=1e-6):
    model.eval()
    utilizations = []
    hooks = []

    def hook_fn(module, inp, out):
        var = out.var(dim=(0, 2, 3), unbiased=False)
        active = (var > eps).float().mean().item()
        utilizations.append(active)

    for m in model.modules():
        if (
            isinstance(m, nn.Conv2d)
            and m.kernel_size == (1, 1)
            and m.groups == 1
            and m.out_channels > m.in_channels
        ):
            hooks.append(m.register_forward_hook(hook_fn))

    model(input_tensor)

    for h in hooks:
        h.remove()

    return sum(utilizations) / max(len(utilizations), 1)

# Depthwise Signal Preservation (DSP)
@torch.no_grad()
def compute_DSP(model, input_tensor, eps=1e-6):
    model.eval()
    log_ratios = []

    def pre_hook(module, inp):
        module._input_var = tensor_variance(inp[0])

    def post_hook(module, inp, out):
        var_out = tensor_variance(out)
        var_in = module._input_var
        ratio = var_out / (var_in + eps)
        log_ratios.append(torch.abs(torch.log(ratio + eps)).item())

    hooks = []

    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
            hooks.append(m.register_forward_pre_hook(pre_hook))
            hooks.append(m.register_forward_hook(post_hook))

    model(input_tensor)

    for h in hooks:
        h.remove()

    return torch.exp(-torch.tensor(log_ratios).mean()).item()


def is_pointwise_conv(conv):
    """1x1 pointwise conv: kernel (1,1), groups=1."""
    return (
        isinstance(conv, nn.Conv2d)
        and conv.kernel_size == (1, 1)
        and conv.groups == 1
    )


def is_depthwise_conv(conv):
    """Depthwise conv: groups == in_channels."""
    return isinstance(conv, nn.Conv2d) and conv.groups == conv.in_channels


def iter_projection_convs(model):
    """
    Yield 1x1 pointwise convs that come AFTER a depthwise conv (projection convs).
    Excludes expansion convs (1x1 before DW) and counts only post-DW 1x1 convs.
    """
    last_conv_was_depthwise = False
    for m in model.modules():
        if is_depthwise_conv(m):
            last_conv_was_depthwise = True
            continue
        if is_pointwise_conv(m):
            if last_conv_was_depthwise:
                yield m
            last_conv_was_depthwise = False


def iter_expansion_convs(model):
    """
    Yield 1x1 pointwise convs that expand channels (out_channels > in_channels).
    These are the expansion convs that come before depthwise in inverted residual blocks.
    """
    for m in model.modules():
        if is_expansion_conv(m):
            yield m


def iter_expansion_and_projection_convs(model):
    """
    Yield 1x1 pointwise convs that are either expansion (out>in) or projection (after DW).
    Used for analytic LBC over both expansion and projection convs.
    """
    last_conv_was_depthwise = False
    for m in model.modules():
        if is_depthwise_conv(m):
            last_conv_was_depthwise = True
            continue
        if is_pointwise_conv(m):
            if is_expansion_conv(m) or last_conv_was_depthwise:
                yield m
            last_conv_was_depthwise = False


def iter_depthwise_convs(model):
    """Yield depthwise conv layers (groups == in_channels)."""
    for m in model.modules():
        if is_depthwise_conv(m):
            yield m


def _pointwise_score(cin, cout):
    """Analytic score for 1x1 conv: sqrt(Cin*Cout) / (sqrt(Cin) + sqrt(Cout))."""
    if cin <= 0 or cout <= 0:
        return 0.0
    return math.sqrt(cin * cout) / (math.sqrt(cin) + math.sqrt(cout))


def _depthwise_score(c_in, k):
    """Depthwise conv score: C_in * k / (1 + sqrt(k))^2, where k = kernel_size^2."""
    if c_in <= 0 or k <= 0:
        return 0.0
    denom = (1 + math.sqrt(k)) ** 2
    if denom <= 0:
        return 0.0
    return (c_in * k) / denom


# Linear Bottleneck Conditioning (LBC)
def compute_LBC(model, power_iters=2, eps=1e-6):
    conds = []

    for m in iter_projection_convs(model):
            W = m.weight.flatten(1)  # [out, in]

            # Frobenius norm
            fro = torch.norm(W, p='fro')

            # Approx spectral norm (power iteration)
            u = torch.randn(W.shape[0], 1, device=W.device)
            for _ in range(power_iters):
                v = F.normalize(W.T @ u, dim=0)
                u = F.normalize(W @ v, dim=0)

            spec = torch.norm(W @ v)
            conds.append((fro / (spec + eps)).item())
            # svd = torch.linalg.svdvals(W)[0]
            # print(f"fro/svd: {fro/svd:.6f}")
    return sum(conds) / max(len(conds), 1)


def compute_LBC_sum(model, power_iters=2, eps=1e-6):
    """Sum of LBC scores over all projection convs (not averaged)."""
    conds = []
    for m in iter_projection_convs(model):
        W = m.weight.flatten(1)
        fro = torch.norm(W, p='fro')
        u = torch.randn(W.shape[0], 1, device=W.device)
        for _ in range(power_iters):
            v = F.normalize(W.T @ u, dim=0)
            u = F.normalize(W @ v, dim=0)
        spec = torch.norm(W @ v)
        conds.append((fro / (spec + eps)).item())
    return sum(conds)


def compute_analytic_ssr(model, use_sum = False):
    """Analytic LBC: sqrt(Cin*Cout) / (sqrt(Cin) + sqrt(Cout)) for each projection conv, averaged."""
    scores = []
    for m in iter_projection_convs(model):
            cin = m.in_channels
            cout = m.out_channels
            score = math.sqrt(cin * cout) / (math.sqrt(cin) + math.sqrt(cout))
            scores.append(score)
    if use_sum:
        return sum(scores)
    else:
        return sum(scores) / max(len(scores), 1)


def compute_analytic_ssr_sum(model):
    """Sum of analytic LBC scores over all projection convs (not averaged)."""
    total = 0.0
    for m in iter_projection_convs(model):
        cin = m.in_channels
        cout = m.out_channels
        total += math.sqrt(cin * cout) / (math.sqrt(cin) + math.sqrt(cout))
    return total


def compute_analytic_ssr_exp_proj(model, use_sum=False):
    """
    Analytic LBC applied to both expansion and projection 1x1 convs.
    Same formula as compute_analytic_lbc: sqrt(Cin*Cout) / (sqrt(Cin) + sqrt(Cout)).
    """
    scores = []
    for m in iter_expansion_and_projection_convs(model):
        score = _pointwise_score(m.in_channels, m.out_channels)
        scores.append(score)
    if use_sum:
        return sum(scores)
    return sum(scores) / max(len(scores), 1)


def compute_analytic_ssr_exp_proj_dw(model, use_sum=False):
    """
    Combined score over expansion, projection, and depthwise convs.
    - Expansion & projection 1x1: sqrt(Cin*Cout) / (sqrt(Cin) + sqrt(Cout))
    - Depthwise: C_in * k / (1 + sqrt(k))^2, where k = kernel_size^2
    """
    scores = []
    for m in iter_expansion_and_projection_convs(model):
        scores.append(_pointwise_score(m.in_channels, m.out_channels))
    for m in iter_depthwise_convs(model):
        k = m.kernel_size[0] * m.kernel_size[1]
        scores.append(_depthwise_score(m.in_channels, k))
    if use_sum:
        return sum(scores)
    return sum(scores) / max(len(scores), 1)


def compute_analytic_ssr_exp_proj_dw_sum(model):
    """Sum of expansion+projection+depthwise scores (not averaged)."""
    total = 0.0
    for m in iter_expansion_and_projection_convs(model):
        total += _pointwise_score(m.in_channels, m.out_channels)
    for m in iter_depthwise_convs(model):
        k = m.kernel_size[0] * m.kernel_size[1]
        total += _depthwise_score(m.in_channels, k)
    return total


def compute_analytic_lbc_exp_proj_sum(model):
    """Sum of analytic LBC scores over expansion and projection convs (not averaged)."""
    total = 0.0
    for m in iter_expansion_and_projection_convs(model):
        cin = m.in_channels
        cout = m.out_channels
        total += math.sqrt(cin * cout) / (math.sqrt(cin) + math.sqrt(cout))
    return total


# Latency-Normalized Expressivity (LNE)
def compute_LNE(expressivity_score, latency):
    return expressivity_score / torch.log(torch.tensor(1.0 + latency))

def aggregate_scores(scores_dict):
    # scores_dict: {name: value}
    values = torch.tensor(list(scores_dict.values()))
    ranks = values.argsort().argsort().float()
    return ranks.mean().item()

def aggregate_weighted(EUS, DSP, LBC, LNE):
    return 0.3 * EUS + 0.3 * DSP + 0.2 * LBC + 0.2 * LNE



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test score metrics on a PlainNet structure string.")
    parser.add_argument(
        "--struct_str",
        type=str,
        default="SuperConvK3BNRELU(3,8,2,1)SuperResIDWE6K3(8,32,2,8,1)"
                "SuperResIDWE6K3(32,48,2,32,1)SuperResIDWE6K3(48,96,2,48,1)"
                "SuperResIDWE6K3(96,128,2,96,1)SuperConvK1BNRELU(128,2048,1,1)",
        help="PlainNet structure string",
    )
    parser.add_argument("--input_shape", type=str, default="1,3,224,224")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--latency", type=float, default=None)
    parser.add_argument("--expressivity_score", type=float, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    shape = [int(x) for x in args.input_shape.strip().strip("()[]").split(",") if x.strip()]
    device = 'cpu'
    x = torch.randn(*shape, device=device)

    # model = PlainNet.PlainNet(
    #     num_classes=1000,
    #     plainnet_struct=args.struct_str,
    #     no_create=False,
    # ).to(device)
    model = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
    network_weight_gaussian_init(model)

    LBC = compute_LBC(model)
    analytic_lbc = compute_analytic_lbc(model)
    print(f"LBC: {LBC:.6f}")
    print(f"analytic_lbc: {analytic_lbc:.6f}")
    # expressivity, progressivity = compute_expressivity(model, x)

    # expressivity_score = EUS if args.expressivity_score is None else args.expressivity_score
    # latency = 1.0 if args.latency is None else args.latency
    # LNE = compute_LNE(expressivity_score, latency)

    # final_score = aggregate_weighted(EUS, DSP, LBC, LNE)

    # print(f"EUS: {EUS:.6f}")
    # print(f"DSP: {DSP:.6f}")
    # print(f"LBC: {LBC:.6f}")
    # print(f"Expressivity: {expressivity:.6f}")
    # print(f"Progressivity: {progressivity:.6f}")
    # print(f"LNE: {LNE:.6f}")
    # scores_dict = {
    #     "EUS": EUS,
    #     "DSP": DSP,
    #     "LBC": LBC,
    #     "LNE": LNE,
    # }
    # # print(f"Aggregate: {aggregate_scores(scores_dict):.6f}")
    # # print(f"Final: {final_score:.6f}")
