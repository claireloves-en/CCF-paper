"""
Package the saved per-transaction predictions for a data archive (Revision_Plan_V07.md, R5).

The predictions behind every test-period and confirmation-period result (~260 MB) are too large for the git
repository. This bundles them with a README and SHA-256 checksums into one zip for deposit (e.g. Zenodo or
OSF), whose DOI then goes into the manuscript's data and code availability statement next to the repository
URL. Nothing is uploaded here.

  python package_predictions.py   ->  output/CCF_predictions_V01.zip
"""
import hashlib
import zipfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FILES = sorted(OUT.glob("test_predictions_temporal_p7_*.csv.gz")) + sorted(OUT.glob("confirm_predictions_temporal_p7_*.csv.gz"))

README = f"""Per-transaction model scores behind the results of
"Evaluating Theory-Informed Feature Encoding for Credit Card Fraud Detection:
Accuracy Costs and Explanation Limits on Synthetic Transactions"

Packaged {date.today().isoformat()}. Code: see the repository named in the paper's data and code availability statement
(scripts/ablation_run.py writes test_predictions_*, scripts/confirm_run.py writes confirm_predictions_*).

test_predictions_temporal_<config>.csv.gz
    Test period, 6 March to 21 June 2020 (259,284 transactions of fraudTrain.csv).
confirm_predictions_temporal_<config>.csv.gz
    Confirmation period, 21 June to 31 December 2020 (555,719 transactions of fraudTest.csv), scored once.

Columns: one per model (predicted fraud probability; Logistic Regression, Decision Tree, Random Forest, AdaBoost,
CatBoost, XGBoost, LightGBM, K-NN, or the subset fitted for that configuration), is_fraud (label), and row (the
transaction's 0-based row number in the source CSV, so timestamps and card numbers can be re-attached).

Configurations (<config>): p7_notime = encoded (E); p7_notime_client = E + client history (H);
p7_notime_raw = conventional (R); p7_notime_raw_client = R + H; p7_hybrid[_client] = R plus E;
p7_catnative[_client] = CatBoost with raw categorical fields; p7_abl_* = one-change configurations of E;
p7_red_raw / p7_red_enc = amount, category and time only; p7_grid_train[_client] = E with the location grid
fitted on training-period bounds. Definitions: docs/ in the repository and the paper's Section 3.2.

SHA256SUMS lists a checksum for every file.
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert FILES, "no prediction files in output/"
    sums = "".join(f"{sha256(p)}  {p.name}\n" for p in FILES)
    out = OUT / "CCF_predictions_V01.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:     # already gzip-compressed
        z.writestr("README.txt", README)
        z.writestr("SHA256SUMS", sums)
        for p in FILES:
            z.write(p, p.name)
    print(f"wrote {out} ({len(FILES)} files, {out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
