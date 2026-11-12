"""
Refit the best K candidates of a finished hyperparameter search on the full
training set, and keep whichever scores best on validation AUC-PR.

Why: d8_tuning.py ranks candidates on a 25% training subsample and then refits
only the subsample winner. Subsample rankings are noisy -- in the trans_time-free
search, XGBoost's subsample winner refitted to AUC-PR 0.8609, while a candidate
from the same search space reached 0.8692 on the full data. That gap is larger
than the spread between the three boosting libraries, so a single refit is not a
safe basis for comparing them. Refitting the top K reduces that noise at the cost
of K-1 extra full-data fits per model.

Reads  output/d8_trials<TAG>.csv  (needs the params column -- restored script only)
Writes output/d8_final_top<K><TAG>.csv, same columns as d8_final<TAG>.csv plus the
       subsample rank of the chosen candidate.

Usage: python refit_top_k.py [--drop-trans-time] [--client-features] [--raw-baseline]
                             [--data FILE --tag NAME] [--models M1,M2] [--k 3]
"""
import importlib.util, json, sys, time
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef

HERE = Path(__file__).resolve().parent
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 3
flags = [f for f in ("--drop-trans-time", "--client-features", "--raw-baseline") if f in sys.argv]
for _opt in ("--data", "--tag", "--models"):   # Phase 7: the generic configurations
    if _opt in sys.argv:
        flags += [_opt, sys.argv[sys.argv.index(_opt) + 1]]

# Import the search script for its data set-up and fitters; its search only
# runs under __main__, and it deletes nothing on import.
sys.argv = ["d8_tuning.py", *flags]
spec = importlib.util.spec_from_file_location("d8", HERE / "d8_tuning.py")
d8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d8)

trials = pd.read_csv(d8.TRIALS_CSV)
assert "params" in trials.columns, "trials file has no params column (pre-restoration run)"
rows = []
for name, _grid, fit in d8.MODELS:
    if d8.ONLY is not None and name not in d8.ONLY:
        continue
    t = trials[(trials.model == name) & trials.aucpr.notna()].sort_values("aucpr", ascending=False)
    cand = []
    for rank, (_, r) in enumerate(t.head(K).iterrows(), start=1):
        p = json.loads(r.params)
        s = time.time()
        proba = fit(p, d8.TRAIN)
        cand.append(dict(rank=rank, trial=int(r.trial), p=p, sub=float(r.aucpr),
                         full=average_precision_score(d8.yv, proba),
                         auc=roc_auc_score(d8.yv, proba),
                         mcc=matthews_corrcoef(d8.yv, (proba > 0.5).astype(int))))
        d8.log(f"  {name:20s} rank {rank} (trial {int(r.trial):2d}): subsample {r.aucpr:.4f} "
               f"-> full {cand[-1]['full']:.4f}  [{time.time()-s:.0f}s]")
    best = max(cand, key=lambda c: c["full"])
    rows.append(dict(model=name, aucpr_subsample=best["sub"], aucpr_full=best["full"],
                     auc_full=best["auc"], mcc_full_at_0p5=best["mcc"],
                     params=json.dumps(best["p"]), subsample_rank=best["rank"],
                     aucpr_full_rank1=cand[0]["full"]))
    d8.log(f"  {name}: kept subsample rank {best['rank']} "
           f"({best['full']:.4f}; rank-1 refit was {cand[0]['full']:.4f})")

out = HERE / "output" / f"d8_final_top{K}{d8.TAG}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}")
