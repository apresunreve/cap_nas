"""
Search space constrained to the EXACT same per-stage channel ranges as the
enumeration algorithm (ICS). Valid blocks and the per-position in_ch/out_ch
ranges are both read from the bundled block lookup, so the space always matches
the shipped lookup.

Pre-computes valid blocks indexed by in_channel AND filters by block_idx at
search time. Guarantees zero missing blocks and perfect connectivity.

Usage:
    --search_space SearchSpace/search_space_IDW_fixfc_enum_space.py
"""
import os, sys, re, csv
from collections import defaultdict

from utils.paths import DATA_DIR, LOOKUP_CSV, REPO_ROOT, setup_model_path

setup_model_path()

from PlainNet import super_blocks, SuperResIDWEXKX

# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------
_valid_stem = []
_valid_idw = defaultdict(list)  # in_ch -> [(exp, k, out_ch, btl_ch, sub, block_str), ...]

# Per-position allowed channel ranges (from enumeration CSV)
# Key: block_idx (1-4 for IDW positions). Each maps to (allowed_in_ch, allowed_out_ch) sets.
_enum_in_ch  = {}  # block_idx -> set of allowed in_ch
_enum_out_ch = {}  # block_idx -> set of allowed out_ch

_idw_pat  = re.compile(r'SuperResIDWE(\d+)K(\d+)\((\d+),(\d+),(\d+),(\d+),(\d+)\)')
_stem_pat = re.compile(r'SuperConvK3BNRELU\((\d+),(\d+),(\d+),(\d+)\)')


def _init_lookup():
    global _valid_stem, _valid_idw, _enum_in_ch, _enum_out_ch
    if _valid_idw:
        return

    # Valid block strings and the per-position channel ranges both come from the
    # bundled block lookup. Block strings are collected across positions (keyed
    # by in_ch); gen_search_space then filters per position using the ranges.
    seen_idw = set()
    with open(str(LOOKUP_CSV)) as f:
        for row in csv.DictReader(f):
            idx_raw = row.get('block_idx', '').strip()
            if not idx_raw.isdigit():
                continue
            idx = int(idx_raw)
            block_str = row['block_str']

            m = _idw_pat.match(block_str)
            if m and block_str not in seen_idw:
                seen_idw.add(block_str)
                exp, k, in_ch, out_ch, stride, btl, sub = map(int, m.groups())
                _valid_idw[in_ch].append((exp, k, out_ch, btl, sub, block_str))
            elif _stem_pat.match(block_str) and block_str not in _valid_stem:
                _valid_stem.append(block_str)

            ic = row['in_channels'].strip()
            oc = row['out_channels'].strip()
            if idx >= 1 and ic and oc:
                _enum_in_ch.setdefault(idx, set()).add(int(ic))
                _enum_out_ch.setdefault(idx, set()).add(int(oc))

    print(f"[enum_space] lookup: {len(seen_idw)} IDW blocks over "
          f"{len(_valid_idw)} in_ch keys, {len(_valid_stem)} stem blocks")
    for idx in sorted(_enum_in_ch):
        print(f"  IDW{idx}: in={sorted(_enum_in_ch[idx])}  out={sorted(_enum_out_ch[idx])}")


_init_lookup()


# ---------------------------------------------------------------------------
# Search space interface
# ---------------------------------------------------------------------------

def gen_search_space(block_list, block_id):
    """Return candidate block strings grouped by block type: List[List[str]].

    Filters candidates to only those whose in_ch AND out_ch appear in the
    enumeration CSV's per-position allowed ranges.
    """
    the_block = block_list[block_id]

    # --- Stem or FC ---
    if isinstance(the_block, super_blocks.SuperConvKXBNRELU):
        if the_block.kernel_size == 1:
            current_str = type(the_block).__name__ + '({},{},{},{})'.format(
                the_block.in_channels, the_block.out_channels, the_block.stride, 1)
            return [[current_str]]

        # Stem: out_ch must be a valid in_ch for IDW1
        allowed_idw1_in = _enum_in_ch.get(1, set())
        if _valid_stem:
            valid_stems = [s for s in _valid_stem
                           if int(_stem_pat.match(s).group(2)) in allowed_idw1_in]
            if valid_stems:
                return [valid_stems]
        current_str = type(the_block).__name__ + '({},{},{},{})'.format(
            the_block.in_channels, the_block.out_channels, the_block.stride, 1)
        return [[current_str]]

    # --- IDW block ---
    in_ch = the_block.in_channels
    candidates = _valid_idw.get(in_ch, [])

    if not candidates:
        current_str = type(the_block).__name__ + '({},{},{},{},{})'.format(
            the_block.in_channels, the_block.out_channels, the_block.stride,
            the_block.bottleneck_channels, the_block.sub_layers)
        return [[current_str]]

    # Per-position filter: in_ch and out_ch must be in enumeration's allowed sets
    # Enumeration guarantees connectivity: out_ch at position i ⊂ in_ch at position i+1
    allowed_in  = _enum_in_ch.get(block_id, set())
    allowed_out = _enum_out_ch.get(block_id, set())

    grouped = defaultdict(list)
    for exp, k, out_ch, btl_ch, sub, block_str in candidates:
        if in_ch in allowed_in and out_ch in allowed_out:
            grouped[(exp, k)].append(block_str)

    result = list(grouped.values())
    if not result:
        current_str = type(the_block).__name__ + '({},{},{},{},{})'.format(
            the_block.in_channels, the_block.out_channels, the_block.stride,
            the_block.bottleneck_channels, the_block.sub_layers)
        return [[current_str]]

    return result
