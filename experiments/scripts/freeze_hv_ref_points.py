from __future__ import annotations
import json
from pathlib import Path

REPO = Path.cwd()

stats_path = REPO / "artifacts" / "tidy" / "hv_archive_campaign_core_metrics.problem_stats.json"
out_path   = REPO / "experiments" / "catalog" / "hv_ref_points.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

stats = json.loads(stats_path.read_text(encoding="utf-8"))
ref_points = {prob: v["ref"] for prob, v in stats.items()}

out_path.write_text(json.dumps(ref_points, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote:", out_path)
print("Count:", len(ref_points))
