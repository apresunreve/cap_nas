"""
Search space constrained to the EXACT same per-stage channel ranges as the
enumeration algorithm (ICS). Uses the timing JSON for block lookups but filters
candidates to only those in_ch/out_ch combos that appear in the enumeration CSV
at each block index.

Pre-computes valid blocks indexed by in_channel AND filters by block_idx at
search time. Guarantees zero missing blocks and perfect connectivity.

Usage:
    --search_space SearchSpace/search_space_IDW_fixfc_enum_space.py
"""
import os, sys, json, re, csv
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

    # --- Load timing JSON ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = REPO_ROOT / "idw_block_lookup_jetson_gpu_timing_full_merged.json"
    if not json_path.exists():
        json_path = REPO_ROOT / "idw_block_lookup_jetson_gpu_timing_full.json"

    with open(str(json_path)) as f:
        data = json.load(f)

    for block_str in data:
        m = _idw_pat.match(block_str)
        if m:
            exp, k, in_ch, out_ch, stride, btl, sub = map(int, m.groups())
            _valid_idw[in_ch].append((exp, k, out_ch, btl, sub, block_str))
            continue
        m = _stem_pat.match(block_str)
        if m:
            _valid_stem.append(block_str)

    # --- Load enumeration CSV for per-position channel ranges ---
    csv_path = LOOKUP_CSV
    if not csv_path.exists():
        csv_path = REPO_ROOT / "standalone" / "data" / "idw_block_lookup.csv"

    with open(str(csv_path)) as f:
        for row in csv.DictReader(f):
            idx = int(row['block_idx'])
            ic = row['in_channels'].strip()
            oc = row['out_channels'].strip()
            if idx >= 1 and ic and oc:
                _enum_in_ch.setdefault(idx, set()).add(int(ic))
                _enum_out_ch.setdefault(idx, set()).add(int(oc))

    # Report
    stem_outs = set()
    for s in _valid_stem:
        m = _stem_pat.match(s)
        if m: stem_outs.add(int(m.group(2)))

    print(f"[enum_space] JSON: {len(data)} entries, {len(_valid_idw)} in_ch keys, "
          f"{len(_valid_stem)} stem blocks")
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
