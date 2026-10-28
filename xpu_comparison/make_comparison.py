import json
import re

import pandas as pd

RUNS = {
    "main (0dbff3e)": "benchmark_tabicl_devices_xpu_main.ipynb",
    "PR #162 (4ac890e)": "benchmark_tabicl_devices_xpu_pr162.ipynb",
}
METRICS = [
    "fit_s",
    "predict_s",
    "total_s",
    "speedup_vs_cpu",
    "r2",
    "max_abs_diff_vs_cpu",
    "host_rss_peak_increment_gb",
    "device_mem_peak_increment_gb",
    "swap_grew",
]


def cell_text(nb, idx):
    out = []
    for o in nb["cells"][idx].get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain")
        if t:
            out.append("".join(t))
    return "\n".join(out)


def results_table(nb):
    text = nb["cells"][12]["outputs"][-1]["data"]["text/plain"]
    text = "".join(text)
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    frames = []
    for b in blocks:
        lines = [l.rstrip(" \\") for l in b.splitlines()]
        header = lines[0].split()
        rows = []
        for l in lines[1:]:
            if "label" in header:
                m = re.match(r"\s*(\d+)\s+(.+? / kv_cache=\w+)\s+(.*)", l)
                rows.append([m.group(1), m.group(2), *m.group(3).split()])
            else:
                rows.append(l.split())
        frames.append(pd.DataFrame(rows, columns=["idx", *header]).set_index("idx"))
    return pd.concat(frames, axis=1).set_index("label")


nbs = {k: json.load(open(v)) for k, v in RUNS.items()}
tables = {k: results_table(nb) for k, nb in nbs.items()}

lines = ["# TabICL XPU benchmark: main vs PR #162", ""]
lines += [
    "Same machine, same pixi env, runs executed back to back with "
    "`jupyter nbconvert --execute`. PR: https://github.com/soda-inria/tabicl/pull/162",
    "",
]
first = next(iter(nbs.values()))
lines += ["## Environment", "", "```", cell_text(first, 3), "", cell_text(first, 6), "```", ""]
lines += ["## Dataset", "", "```", cell_text(first, 8), "```", ""]

labels = next(iter(tables.values())).index
for metric in METRICS:
    lines += [f"## `{metric}`", "", "| run | " + " | ".join(RUNS) + " |"]
    lines += ["|---|" + "---|" * len(RUNS)]
    for label in labels:
        vals = [tables[k].loc[label, metric] for k in RUNS]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

for name, nb in nbs.items():
    lines += [f"## Run log: {name}", "", "```", cell_text(nb, 12).split("\n                              label")[0].strip(), "```", ""]

open("comparison.md", "w").write("\n".join(lines))
print("\n".join(lines))
