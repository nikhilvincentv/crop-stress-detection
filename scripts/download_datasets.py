#!/usr/bin/env python3
"""
TASK 1 - Download all real datasets for AgriDefend multi-species training.

Zero synthetic data: every source below is a real measured dataset. Each
download is wrapped in try/except and prints exactly which dataset failed and
why. A dataset inventory report is printed at the end.

Datasets
--------
1. PlantVillage          -> data/plantvillage/        (visual CNN training)
2. Kaggle Crop Recommend -> data/crop_recommendation/ (soil RF baselines)
3. NASA SRDB             -> data/srdb/                 (CO2 flux calibration)
4. COSORE                -> data/cosore/               (hourly CO2 flux series)
5. USDA Cook Agronomy    -> data/cook_farm/            (soil moisture series)
6. Rice leaf disease     -> data/rice/                 (CNN supplement)
7. Wheat leaf disease    -> data/wheat/                (CNN supplement)

Usage:  python scripts/download_datasets.py
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Per-dataset outcome record for the final inventory report.
REPORT = {}


def _record(name, target_dir, error=None, samples=None):
    """Compute file count / size for a directory and store the outcome."""
    target_dir = Path(target_dir)
    n_files, total_bytes = 0, 0
    if target_dir.exists():
        for p in target_dir.rglob("*"):
            if p.is_file():
                n_files += 1
                total_bytes += p.stat().st_size
    REPORT[name] = {
        "dir": str(target_dir.relative_to(ROOT)),
        "files": n_files,
        "bytes": total_bytes,
        "samples": samples,
        "error": error,
    }


def _human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _download(url, dest, timeout=120, chunk=1 << 20):
    """Stream a URL to disk, following redirects."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout,
                      headers={"User-Agent": "AgriDefend/1.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
    return dest


def _kaggle_available():
    if shutil.which("kaggle") is None:
        return False
    cred = Path.home() / ".kaggle" / "kaggle.json"
    return cred.exists() or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)


def _kaggle_download(slug, out_dir):
    """Download + unzip a Kaggle dataset via the CLI. Raises on failure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"],
        check=True, capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# DATASET 1: PlantVillage  (54,304 leaf images, 14 species, 38 disease classes)
# ---------------------------------------------------------------------------
def dataset_plantvillage():
    name = "1_PlantVillage"
    target = DATA / "plantvillage"
    target.mkdir(parents=True, exist_ok=True)
    try:
        # Preferred path: TensorFlow Datasets (no login required).
        try:
            import tensorflow_datasets as tfds  # noqa
            ds, info = tfds.load("plant_village", with_info=True,
                                 as_supervised=True, data_dir=str(target))
            n = int(info.splits["train"].num_examples)
            _record(name, target, samples=n)
            print(f"[OK] PlantVillage via TFDS: {n} images -> {target}")
            return
        except Exception as e_tfds:
            print(f"[..] PlantVillage TFDS path unavailable ({e_tfds}); trying Kaggle")

        if _kaggle_available():
            _kaggle_download("abdallahalidev/plantvillage-dataset", target)
            _record(name, target)
            print(f"[OK] PlantVillage via Kaggle -> {target}")
        else:
            raise RuntimeError(
                "tensorflow_datasets not installed AND no Kaggle credentials "
                "(~/.kaggle/kaggle.json). Install TF or add Kaggle API key.")
    except Exception as e:
        _record(name, target, error=str(e))
        print(f"[FAIL] PlantVillage: {e}")


# ---------------------------------------------------------------------------
# DATASET 2: Kaggle Crop Recommendation  (real IoT NPK/pH/moisture, 22 crops)
# ---------------------------------------------------------------------------
def dataset_crop_recommendation():
    name = "2_CropRecommendation"
    target = DATA / "crop_recommendation"
    target.mkdir(parents=True, exist_ok=True)
    csv_out = target / "Crop_recommendation.csv"
    try:
        if _kaggle_available():
            try:
                _kaggle_download("atharvaingle/crop-recommendation-dataset", target)
                # normalise filename
                for p in target.glob("*.csv"):
                    if p.name != csv_out.name:
                        shutil.copy(p, csv_out)
                rows = sum(1 for _ in open(csv_out)) - 1
                _record(name, target, samples=rows)
                print(f"[OK] Crop Recommendation via Kaggle: {rows} rows")
                return
            except Exception as e_k:
                print(f"[..] Kaggle path failed ({e_k}); trying public mirror")

        # Fallback: canonical public mirror of the identical Kaggle CSV
        # (same real measured values: N,P,K,temperature,humidity,ph,rainfall,label).
        mirror = ("https://raw.githubusercontent.com/gireesh777/"
                  "Crop_Recommendation_System_using_ML/master/Dataset/"
                  "Crop_recommendation.csv")
        _download(mirror, csv_out, timeout=60)
        rows = sum(1 for _ in open(csv_out)) - 1
        header = open(csv_out).readline().strip()
        assert "N,P,K" in header.replace(" ", ""), f"unexpected schema: {header}"
        _record(name, target, samples=rows)
        print(f"[OK] Crop Recommendation via mirror: {rows} rows -> {csv_out}")
    except Exception as e:
        _record(name, target, error=str(e))
        print(f"[FAIL] Crop Recommendation: {e}")


# ---------------------------------------------------------------------------
# DATASET 3: NASA SRDB  (10,000+ field soil CO2 flux measurements)
# ---------------------------------------------------------------------------
def dataset_srdb():
    name = "3_SRDB"
    target = DATA / "srdb"
    csv_out = target / "srdb-data.csv"
    try:
        url = "https://raw.githubusercontent.com/bpbond/srdb/master/srdb-data.csv"
        _download(url, csv_out, timeout=120)
        rows = sum(1 for _ in open(csv_out, encoding="utf-8", errors="replace")) - 1
        _record(name, target, samples=rows)
        print(f"[OK] SRDB: {rows} records -> {csv_out}")
    except Exception as e:
        _record(name, target, error=str(e))
        print(f"[FAIL] SRDB: {e}")


# ---------------------------------------------------------------------------
# DATASET 4: COSORE  (continuous hourly soil CO2 flux time series)
# ---------------------------------------------------------------------------
def dataset_cosore():
    name = "4_COSORE"
    target = DATA / "cosore"
    target.mkdir(parents=True, exist_ok=True)
    try:
        # Resolve the latest release asset (cosore-X.Y.Z.zip) via GitHub API.
        api = "https://api.github.com/repos/bpbond/cosore/releases"
        rels = requests.get(api, timeout=60,
                            headers={"User-Agent": "AgriDefend/1.0"}).json()
        asset = None
        for rel in rels:
            for a in rel.get("assets", []):
                if a["name"].endswith(".zip"):
                    asset = a
                    break
            if asset:
                break
        if asset is None:
            raise RuntimeError("no .zip asset found in COSORE releases")

        zip_path = target / asset["name"]
        print(f"[..] COSORE downloading {asset['name']} "
              f"({_human(asset['size'])}) ...")
        _download(asset["browser_download_url"], zip_path, timeout=600)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
        names = [n for n in zipfile.ZipFile(zip_path).namelist()]
        csvs = [n for n in names if n.lower().endswith(".csv")]
        rds = [n for n in names if n.lower().endswith(".rds")]
        note = None
        if not csvs and rds:
            note = (f"archive ships {len(rds)} .rds R-binary files, no flat CSV; "
                    f"prepare step will read via pyreadr if available")
        _record(name, target, samples=len(csvs) or len(rds), error=note)
        print(f"[OK] COSORE extracted: {len(csvs)} CSV, {len(rds)} RDS files")
        if note:
            print(f"     note: {note}")
    except Exception as e:
        _record(name, target, error=str(e))
        print(f"[FAIL] COSORE: {e}")


# ---------------------------------------------------------------------------
# DATASET 5: USDA Cook Agronomy Farm  (hourly volumetric soil moisture, wheat)
# ---------------------------------------------------------------------------
def dataset_cook_farm():
    name = "5_CookFarm"
    target = DATA / "cook_farm"
    target.mkdir(parents=True, exist_ok=True)
    try:
        # Ag Data Commons (figshare) article for DOI 10.15482/USDA.ADC/1349683.
        meta = requests.get("https://api.figshare.com/v2/articles/24852273",
                            timeout=60).json()
        files = meta.get("files", [])
        if not files:
            raise RuntimeError("figshare article returned no files")
        for f in files:
            dest = target / f["name"]
            print(f"[..] Cook Farm downloading {f['name']} "
                  f"({_human(f['size'])}) ...")
            _download(f["download_url"], dest, timeout=600)
            if dest.suffix.lower() == ".zip":
                with zipfile.ZipFile(dest) as z:
                    z.extractall(target)
        txt = list(target.rglob("*.txt")) + list(target.rglob("*.csv"))
        _record(name, target, samples=len(txt))
        print(f"[OK] Cook Farm: {len(txt)} data files extracted -> {target}")
    except Exception as e:
        _record(name, target, error=str(e))
        print(f"[FAIL] Cook Farm: {e}")


# ---------------------------------------------------------------------------
# DATASET 6 & 7: Rice / Wheat leaf disease images (Kaggle; try each in order)
# ---------------------------------------------------------------------------
def _dataset_kaggle_images(name, target, slugs):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if not _kaggle_available():
        msg = ("no Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/"
               "KAGGLE_KEY); cannot download Kaggle image dataset")
        _record(name, target, error=msg)
        print(f"[FAIL] {name}: {msg}")
        return
    last = None
    for slug in slugs:
        try:
            _kaggle_download(slug, target)
            imgs = sum(1 for _ in target.rglob("*")
                       if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
            _record(name, target, samples=imgs)
            print(f"[OK] {name} via {slug}: {imgs} images")
            return
        except Exception as e:
            last = f"{slug}: {e}"
            print(f"[..] {name} slug failed -> {last}")
    _record(name, target, error=last)
    print(f"[FAIL] {name}: all candidate datasets failed ({last})")


def dataset_rice():
    _dataset_kaggle_images(
        "6_Rice", DATA / "rice",
        ["minhhuy2810/rice-diseases-image-dataset", "shayanriyaz/riceleafs"])


def dataset_wheat():
    _dataset_kaggle_images(
        "7_Wheat", DATA / "wheat",
        ["kushagra3/wheat-disease-dataset", "aman9026/wheat-leaf-dataset"])


def main():
    print("=" * 70)
    print("AGRIDEFEND DATASET DOWNLOADER  (Task 1)")
    print("=" * 70)
    DATA.mkdir(exist_ok=True)

    dataset_plantvillage()
    dataset_crop_recommendation()
    dataset_srdb()
    dataset_cosore()
    dataset_cook_farm()
    dataset_rice()
    dataset_wheat()

    # -------- Inventory report --------
    print("\n" + "=" * 70)
    print("DATASET INVENTORY REPORT")
    print("=" * 70)
    print(f"{'Dataset':<22}{'Files':>7}{'Size':>12}{'Samples':>10}  Status")
    print("-" * 70)
    ok = 0
    for nm, r in REPORT.items():
        status = "ERROR: " + r["error"][:60] if r["error"] else "OK"
        if not r["error"]:
            ok += 1
        samp = "" if r["samples"] is None else str(r["samples"])
        print(f"{nm:<22}{r['files']:>7}{_human(r['bytes']):>12}{samp:>10}  {status}")
    print("-" * 70)
    print(f"{ok}/{len(REPORT)} datasets downloaded without errors.")
    print("=" * 70)


if __name__ == "__main__":
    main()
