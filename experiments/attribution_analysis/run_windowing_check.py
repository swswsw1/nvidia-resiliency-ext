"""
Run FR attribution windowing on a trace directory and report window_idx per PG.

This script bypasses the NVRx package __init__.py (which requires MCP) by shimming
the module imports directly. It loads trace files, runs group_collectives_by_windows(),
and prints the resulting (pg_id, pg_desc, window_idx) keys.

Usage:
    FR_DEBUG=1 python3 run_windowing_check.py <trace_dir>

Example:
    FR_DEBUG=1 python3 run_windowing_check.py \
        ../../tests/attribution/unit/fr_traces/gpu_error_1st
"""

import glob
import importlib.util
import os
import sys
import types

# Shim the package to avoid MCP import chain
sys.modules["nvidia_resiliency_ext"] = types.ModuleType("nvidia_resiliency_ext")
sys.modules["nvidia_resiliency_ext.attribution"] = types.ModuleType(
    "nvidia_resiliency_ext.attribution"
)

# Find repo root (two levels up from this script)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "src")

# Import base and utils directly (no MCP dependency)
for mod_name, rel_path in [
    (
        "nvidia_resiliency_ext.attribution.utils",
        "nvidia_resiliency_ext/attribution/utils.py",
    ),
    (
        "nvidia_resiliency_ext.attribution.base",
        "nvidia_resiliency_ext/attribution/base.py",
    ),
]:
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(SRC, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

# Import fr_attribution
spec = importlib.util.spec_from_file_location(
    "fr_attribution",
    os.path.join(
        SRC,
        "nvidia_resiliency_ext/attribution/trace_analyzer/fr_attribution.py",
    ),
)
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace_dir>")
        sys.exit(1)

    trace_dir = sys.argv[1]

    class MockArgs:
        pattern = "_dump*"
        verbose = False
        health_check = False
        llm_analyze = False
        scheduling_order_file = None
        model = ""
        debug = True
        fr_path = trace_dir

    analyzer = fr.CollectiveAnalyzer(MockArgs())

    # Load trace files
    json_files = sorted(glob.glob(os.path.join(trace_dir, "_dump*")))
    print(f"Processing {len(json_files)} files from {trace_dir}")
    for f in json_files:
        analyzer.process_file(f)

    print(f"Loaded {len(analyzer.collectives_by_file)} ranks")
    for rank, colls in sorted(analyzer.collectives_by_file.items()):
        print(f"  rank {rank}: {len(colls)} entries")

    # Run windowing
    analyzer.collective_groups = analyzer.group_collectives_by_windows()

    # Report results
    print("\n=== WINDOW KEYS ===")
    for key in sorted(
        analyzer.collective_groups.keys(), key=lambda x: (x[1], x[0], x[2])
    ):
        pg_id, pg_desc, win_idx = key
        entries = analyzer.collective_groups[key]
        ranks = sorted(set(c.file_id for c in entries))
        print(
            f"  pg={pg_id:>3s}  {pg_desc:45s}  window={win_idx}  "
            f"entries={len(entries):3d}  ranks={ranks}"
        )

    # Summary: any PG with multiple windows?
    max_windows = {}
    for key in analyzer.collective_groups:
        pg_key = (key[0], key[1])
        max_windows[pg_key] = max(max_windows.get(pg_key, 0), key[2])

    print()
    multi = {k: v for k, v in max_windows.items() if v > 0}
    if multi:
        print(f"PGs with multiple windows: {multi}")
    else:
        print("NO PG has window_idx > 0. Everything is window 0.")


if __name__ == "__main__":
    main()
