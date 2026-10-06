from pathlib import Path
from datetime import datetime, timezone
import json
ROOT=Path(__file__).resolve().parents[1]
items={}
for version in ["v12_exact_v5_fullstack","v13_xgb_full","v14_catboost_feature_hpo","v15_shift_audit","v16_ensemble_scan"]:
    p=ROOT/"artifacts"/version/"results.json"
    if p.is_file():
        try: items[version]=json.loads(p.read_text(encoding="utf-8"))
        except Exception as e: items[version]={"read_error":str(e)}
    else: items[version]={"status":"missing_or_failed"}
out={"completed_utc":datetime.now(timezone.utc).isoformat(),"experiments":items}
(ROOT/"reports"/"overnight_summary.json").write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
print(json.dumps({k:("complete" if "status" not in v else v["status"]) for k,v in items.items()},indent=2))
