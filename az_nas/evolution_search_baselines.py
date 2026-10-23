#!/usr/bin/env python3
"""Original AZ-NAS evolutionary search, driven by this package's search space.

This follows the upstream flow of AZ-NAS's `evolution_search_az.py` and
`evolution_search_others.py`: one candidate per iteration, mutated from a parent,
built as a PlainNet model, checked against the budgets, then scored. There is no
pre-screened candidate pool: candidates are generated and rejected one at a
time, as the authors wrote it.

Upstream's two files differ in more than the proxy, and both schemes are kept
here and selected by `--zero_shot_score`:

  az_nas, vkdnw   rank-aggregated score (each component ranked over the whole
                  population, log(rank/n), summed, recomputed every iteration);
                  population grows and is never pruned; the parent is drawn at
                  random from the top (population_size - 1).

  Zen, Syncflow,  single scalar score; population is pruned to population_size
  NASWOT, zico    by dropping the lowest-scoring member; parent is uniform.

Shared with upstream: the population starts empty; while it holds 10 or fewer
members candidates are mutated from the seed structure with one replacement,
otherwise from a parent with two; budgets are checked after mutation and before
scoring; and a rejected candidate does not advance `--evolution_max_iter`, which
therefore counts accepted candidates.

Latency differs from upstream by necessity. Upstream measures each candidate on
the device via `benchmark_network_latency.get_model_latency`; that needs the
target hardware, so here latency is summed from the bundled block lookup keyed
on (block_idx, block_str), i.e. position-aware and identical to what the
enumeration uses. FLOPs and parameters are computed from the model exactly as
upstream does.

Usage:
    python evolution_search_baselines.py \\
        --zero_shot_score az_nas --budget_latency 10 \\
        --evolution_max_iter 100000 --population_size 512 \\
        --save_dir ~/cap_nas_results/az_nas_lat10ms --gpu 0
"""
import argparse
import csv
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

AZ_DIR = Path(__file__).resolve().parent
REPO_ROOT = AZ_DIR.parent
# PlainNet imports global_utils, which lives in the package's utils/ dir
for _p in (AZ_DIR, REPO_ROOT, REPO_ROOT / "model", REPO_ROOT / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import global_utils
import Masternet
import PlainNet

LOOKUP_CSV = REPO_ROOT / "data" / "block_lookup" / "idw_block_lookup_jetson_gpu.csv"
DEFAULT_SEARCH_SPACE = (REPO_ROOT / "search_space"
                        / "search_space_IDW_fixfc_enum_space.py")

# Proxies whose score is a dict of components combined by rank aggregation,
# as in upstream evolution_search_az.py.
RANK_AGG_PROXIES = ("az_nas", "vkdnw")

SEED_STRUCTURE = (
    "SuperConvK3BNRELU(3,8,2,1)"
    "SuperResIDWE6K3(8,32,2,8,1)"
    "SuperResIDWE6K3(32,48,2,32,1)"
    "SuperResIDWE6K3(48,96,2,48,1)"
    "SuperResIDWE6K3(96,128,2,96,1)"
    "SuperConvK1BNRELU(128,2048,1,1)"
)

_block_re = re.compile(r'([A-Za-z0-9_]+)\(([^)]*)\)')
_timing_pos = None


# ---------------------------------------------------------------------------
# Latency from the bundled lookup (position-aware)
# ---------------------------------------------------------------------------
def load_timing(csv_path=None):
    """(block_idx, block_str) -> latency ms, from the bundled block lookup."""
    global _timing_pos
    if _timing_pos is not None:
        return _timing_pos
    _timing_pos = {}
    with open(str(csv_path or LOOKUP_CSV)) as f:
        for row in csv.DictReader(f):
            try:
                _timing_pos[(int(row['block_idx']), row['block_str'])] = \
                    float(row['jetson_gpu_latency_ms_median'])
            except (ValueError, TypeError):
                continue
    logging.info(f"Loaded {len(_timing_pos)} latency entries (position-aware)")
    return _timing_pos


def lookup_latency(structure_str):
    """Sum per-block latency by position. None if any block is missing."""
    timing = load_timing()
    total = 0.0
    for pos, m in enumerate(_block_re.finditer(structure_str)):
        blk = f"{m.group(1)}({m.group(2)})"
        v = timing.get((pos, blk))
        if v is None:
            return None
        total += v
    return total


# ---------------------------------------------------------------------------
# Mutation -- upstream get_new_random_structure_str
# ---------------------------------------------------------------------------
def get_new_random_structure_str(AnyPlainNet, structure_str, num_classes,
                                 get_search_space_func, num_replaces=1):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str,
                          no_create=True)
    assert isinstance(the_net, PlainNet.PlainNet)
    selected_random_id_set = set()
    for _ in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)

        to_search_list_list = get_search_space_func(the_net.block_list, random_id)
        to_search_list = [x for sub in to_search_list_list for x in sub]
        # upstream sorts for reproducibility: importlib ordering is unstable
        to_search_list = sorted(to_search_list)
        new_block_str = random.choice(to_search_list)

        if len(new_block_str) > 0:
            new_block = PlainNet.create_netblock_list_from_str(new_block_str,
                                                               no_create=True)
            assert len(new_block) == 1
            new_block = new_block[0]
            if random_id > 0:
                new_block.set_in_channels(
                    the_net.block_list[random_id - 1].out_channels)
            the_net.block_list[random_id] = new_block
        else:
            the_net.block_list[random_id] = None

    # drop removed blocks, then re-chain channels
    tmp_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = tmp_block_list[0].out_channels
    for block in tmp_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_block_list

    # Upstream stops here: only the 1-2 blocks picked above are replaced, and
    # channels are re-propagated. It deliberately does NOT re-draw every block,
    # which is what keeps a mutation local to its parent.
    return str(the_net)


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str,
                          no_create=True)
    assert hasattr(the_net, 'split')
    return the_net.split(split_layer_threshold=6)


# ---------------------------------------------------------------------------
# Data for the one proxy that needs it
# ---------------------------------------------------------------------------
def build_train_batches(args, gpu):
    """Batches for ZiCo. Synthetic when --datapath is omitted.

    Synthetic batches exercise the pipeline but produce scores that are NOT
    meaningful for ranking architectures.
    """
    n_batches = max(1, args.maxbatch)
    if not args.datapath:
        logging.warning(
            "No --datapath: using synthetic random batches. ZiCo scores from "
            "synthetic data are not meaningful for ranking.")
        out = []
        for _ in range(n_batches):
            x = torch.randn(args.batch_size, 3, args.input_image_size,
                            args.input_image_size)
            y = torch.randint(0, args.num_classes, (args.batch_size,))
            out.append([x.cuda(gpu), y.cuda(gpu)])
        return out

    import DataLoader as _DL
    opt = argparse.Namespace(
        dataset=args.dataset, data_dir=args.datapath,
        input_image_size=args.input_image_size, input_image_crop=0.875,
        batch_size=args.batch_size, batch_size_per_gpu=args.batch_size,
        workers_per_gpu=args.num_worker, dist_mode="none", frac=1, epfrac=1,
        img_per_class=-1, random_erase=False, auto_augment=False, rank=0,
        world_size=1, independent_training=True, no_data_augment=True,
        dataloader_testing=False, class_balanced_sampling=False,
        balanced_epoch_samples=None)
    loader = _DL.get_data(opt, [])["train_loader"]
    out = []
    for i, batch in enumerate(loader):
        if i == n_batches:
            break
        out.append([batch[0].cuda(gpu), batch[1].cuda(gpu)])
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_nas_score(AnyPlainNet, structure_str, gpu, args,
                      trainbatches=None, lossfunc=None):
    """Score one candidate. Returns a dict for rank-aggregated proxies,
    otherwise a float."""
    model = AnyPlainNet(num_classes=args.num_classes,
                        plainnet_struct=structure_str, no_create=False,
                        no_reslink=False).cuda(gpu)
    proxy = args.zero_shot_score
    res, bs = args.input_image_size, args.batch_size

    if proxy == 'az_nas':
        from ZeroShotProxy import compute_az_nas_score
        return compute_az_nas_score.compute_nas_score(
            model=model, gpu=gpu, trainloader=None, resolution=res, batch_size=bs)
    if proxy == 'vkdnw':
        from ZeroShotProxy import compute_vkdnw_score
        return compute_vkdnw_score.compute_nas_score(
            model=model, gpu=gpu, trainloader=None, resolution=res,
            batch_size=min(bs, 16))
    if proxy == 'Zen':
        from ZeroShotProxy import compute_zen_score
        return compute_zen_score.compute_nas_score(
            model=model, gpu=gpu, resolution=res, mixup_gamma=1e-2,
            batch_size=bs, repeat=1, fp16=False)['avg_nas_score']
    if proxy == 'Syncflow':
        from ZeroShotProxy import compute_syncflow_score
        return compute_syncflow_score.do_compute_nas_score(
            model=model, gpu=gpu, resolution=res, batch_size=bs)
    if proxy == 'NASWOT':
        from ZeroShotProxy import compute_NASWOT_score
        return compute_NASWOT_score.compute_nas_score(
            gpu=gpu, model=model, resolution=res, batch_size=bs)
    if proxy.lower() == 'zico':
        from ZeroShotProxy import compute_zico
        return compute_zico.getzico(model, trainbatches, lossfunc)
    raise NotImplementedError(f"unknown --zero_shot_score {proxy!r}")


def rank_aggregate(score_dict):
    """Upstream AZ-NAS composite: per component log(rank/n), summed."""
    total = None
    for key in score_dict:
        n = len(score_dict[key])
        ranked = np.log(stats.rankdata(score_dict[key]) / n)
        total = ranked if total is None else total + ranked
    return total.tolist()


# ---------------------------------------------------------------------------
def main(args, argv):
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    select_search_space = global_utils.load_py_module_from_path(args.search_space)
    AnyPlainNet = Masternet.MasterNet
    use_rank_agg = args.zero_shot_score in RANK_AGG_PROXIES

    if args.budget_latency is not None:
        load_timing(args.timing_results_csv)

    initial_structure_str = SEED_STRUCTURE
    if args.plainnet_struct_txt:
        with open(args.plainnet_struct_txt) as f:
            initial_structure_str = f.read().strip()

    trainbatches = lossfunc = None
    if args.zero_shot_score.lower() == 'zico':
        trainbatches = build_train_batches(args, gpu)
        lossfunc = nn.CrossEntropyLoss().cuda(gpu)

    popu_structure_list = []
    popu_latency_list = []
    popu_score_list = []        # composite (rank-agg) or scalar
    popu_score_dict = None      # rank-agg components
    attempts = 0
    loop_count = 0
    rejects_in_a_row = 0
    start_timer = time.time()

    while loop_count < args.evolution_max_iter:
        attempts += 1
        if rejects_in_a_row >= args.max_rejects_in_a_row:
            logging.error(
                f"{rejects_in_a_row} consecutive candidates failed the budgets. "
                f"The budget is probably infeasible from this seed -- upstream "
                f"would loop here indefinitely. Loosen the budget, or pass a "
                f"seed already inside it via --plainnet_struct_txt.")
            break

        # ----- pick a parent and mutate (upstream schedule) -----
        if len(popu_structure_list) <= 10:
            parent = initial_structure_str
            num_replaces = 1
        elif len(popu_structure_list) < args.population_size - 1:
            parent = popu_structure_list[random.randint(0, len(popu_structure_list) - 1)]
            num_replaces = 2
        elif use_rank_agg:
            # upstream az.py: random among the top (population_size - 1)
            top = np.argsort(popu_score_list, axis=0)[-args.population_size + 1:]
            parent = popu_structure_list[int(np.random.choice(top))]
            num_replaces = 2
        else:
            # upstream others.py: uniform parent
            parent = popu_structure_list[random.randint(0, len(popu_structure_list) - 1)]
            num_replaces = 2

        structure_str = get_new_random_structure_str(
            AnyPlainNet=AnyPlainNet, structure_str=parent,
            num_classes=args.num_classes,
            get_search_space_func=select_search_space.gen_search_space,
            num_replaces=num_replaces)
        structure_str = get_splitted_structure_str(
            AnyPlainNet, structure_str, num_classes=args.num_classes)

        # ----- budgets, upstream order; a reject does not advance loop_count -----
        the_model = None
        if args.max_layers is not None:
            the_model = the_model or AnyPlainNet(
                num_classes=args.num_classes, plainnet_struct=structure_str,
                no_create=True, no_reslink=False)
            if args.max_layers < the_model.get_num_layers():
                rejects_in_a_row += 1
                continue

        if args.budget_model_size is not None:
            the_model = the_model or AnyPlainNet(
                num_classes=args.num_classes, plainnet_struct=structure_str,
                no_create=True, no_reslink=False)
            if args.budget_model_size < the_model.get_model_size():
                rejects_in_a_row += 1
                continue

        if args.budget_params is not None:
            the_model = the_model or AnyPlainNet(
                num_classes=args.num_classes, plainnet_struct=structure_str,
                no_create=True, no_reslink=False)
            if args.budget_params < the_model.get_model_size():
                rejects_in_a_row += 1
                continue

        if args.budget_flops is not None:
            the_model = the_model or AnyPlainNet(
                num_classes=args.num_classes, plainnet_struct=structure_str,
                no_create=True, no_reslink=False)
            if args.budget_flops < the_model.get_FLOPs(args.input_image_size):
                rejects_in_a_row += 1
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            the_latency = lookup_latency(structure_str)
            if the_latency is None:          # a block absent from the lookup
                rejects_in_a_row += 1
                continue
            if args.budget_latency < the_latency:
                rejects_in_a_row += 1
                continue

        # ----- score and insert -----
        rejects_in_a_row = 0
        score = compute_nas_score(AnyPlainNet, structure_str, gpu, args,
                                  trainbatches, lossfunc)

        if use_rank_agg:
            if popu_score_dict is None:
                popu_score_dict = {k: [] for k in score.keys()}
            for k, v in score.items():
                popu_score_dict[k].append(float(v))
            popu_structure_list.append(structure_str)
            popu_latency_list.append(the_latency)
            popu_score_list = rank_aggregate(popu_score_dict)
            # upstream az.py never prunes: the population grows
        else:
            popu_structure_list.append(structure_str)
            popu_latency_list.append(the_latency)
            popu_score_list.append(float(score))
            # upstream others.py: drop the lowest-scoring member
            while len(popu_structure_list) > args.population_size:
                worst = popu_score_list.index(min(popu_score_list))
                popu_score_list.pop(worst)
                popu_structure_list.pop(worst)
                popu_latency_list.pop(worst)

        loop_count += 1

        if loop_count % 100 == 0:
            best = int(np.argmax(popu_score_list))
            logging.info(
                f"iter {loop_count}/{args.evolution_max_iter} "
                f"pop={len(popu_structure_list)} attempts={attempts} "
                f"best_score={popu_score_list[best]:.4f} "
                f"latency={popu_latency_list[best]}")

        if args.max_time_hours is not None and \
                (time.time() - start_timer) / 3600.0 > args.max_time_hours:
            logging.info(f"Time limit reached after {loop_count} iterations")
            break

    if not popu_structure_list:
        logging.error("No candidate satisfied the budgets")
        return popu_structure_list, popu_score_list, popu_latency_list

    best = int(np.argmax(popu_score_list))
    best_struct = popu_structure_list[best]
    # keep only complete block matches: mutation can leave trailing fragments
    # that the fast_* validators tolerate but PlainNet's parser rejects
    best_struct = ''.join(f"{m.group(1)}({m.group(2)})"
                          for m in _block_re.finditer(best_struct))

    best_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_txt)
    with open(best_txt, 'w') as f:
        f.write(best_struct)
    logging.info(f"Exported best (score={popu_score_list[best]:.4f}, "
                 f"latency={popu_latency_list[best]}) to {best_txt}")
    logging.info(f"attempts={attempts} accepted={loop_count} "
                 f"final_population={len(popu_structure_list)}")
    return popu_structure_list, popu_score_list, popu_latency_list


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--zero_shot_score', type=str, default='az_nas',
                        choices=['az_nas', 'vkdnw', 'Zen', 'Syncflow',
                                 'NASWOT', 'zico'])
    parser.add_argument('--search_space', type=str,
                        default=str(DEFAULT_SEARCH_SPACE),
                        help='Path to a search-space module exposing gen_search_space')
    parser.add_argument('--evolution_max_iter', type=int, default=100000,
                        help='Accepted candidates to evaluate; rejects do not count')
    parser.add_argument('--population_size', type=int, default=512)
    parser.add_argument('--budget_latency', type=float, default=None,
                        help='ms, summed from the bundled lookup (position-aware)')
    parser.add_argument('--budget_flops', type=float, default=None)
    parser.add_argument('--budget_params', type=float, default=None)
    parser.add_argument('--budget_model_size', type=float, default=None)
    parser.add_argument('--max_layers', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_image_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='imagenet100')
    parser.add_argument('--datapath', type=str, default=None,
                        help='Dataset root. Only ZiCo needs it; omit to use '
                             'synthetic random batches (scores then are not '
                             'meaningful for ranking).')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--maxbatch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--max_time_hours', type=float, default=None)
    parser.add_argument('--timing_results_csv', type=str, default=str(LOOKUP_CSV),
                        help='Position-aware latency lookup CSV')
    parser.add_argument('--max_rejects_in_a_row', type=int, default=20000,
                        help='Abort if this many candidates fail the budgets '
                             'consecutively, instead of looping forever on an '
                             'infeasible budget')
    parser.add_argument('--plainnet_struct_txt', type=str, default=None,
                        help='Seed structure; defaults to the built-in seed')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_cmd_options(sys.argv[1:])
    os.makedirs(args.save_dir, exist_ok=True)
    log_fn = os.path.join(args.save_dir, 'evolution_search_baselines.log')
    global_utils.create_logging(log_fn)
    main(args, sys.argv[1:])
