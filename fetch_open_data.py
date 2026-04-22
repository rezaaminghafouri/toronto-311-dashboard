"""
fetch_open_data.py
──────────────────
Downloads the City of Toronto 311 Service Requests dataset
directly from the Open Data portal API — no local files needed.

Run this script once to test the connection and see what resources
are available:
    python fetch_open_data.py

It will:
  1. List all available CSV resources and their URLs
  2. Download and concatenate them into a single DataFrame
  3. Cache the result to  cache/311_data.parquet  so future
     runs skip the download (fast reload in ~2 seconds)

Usage in dashboard.py:
    from fetch_open_data import load_311_data
    df = load_311_data()
"""

import io
import time
import hashlib
from pathlib import Path

import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
PACKAGE_ID  = "311-service-requests-customer-initiated"
CACHE_DIR   = Path(__file__).parent / "cache"
CACHE_FILE  = CACHE_DIR / "311_data.parquet"
CACHE_HOURS = 24          # re-download if cache is older than this

# Columns we actually need (skip the rest to save memory)
KEEP_COLS = [
    "Service Request ID",
    "Status",
    "Open Date",           # may also appear as "Creation Date"
    "Closed Date",
    "Division",
    "Section",
    "Ward",
    "Servicerequesttype",  # may also appear as "Service Request Type"
    "FSA",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_resources(data_only: bool = True) -> list[dict]:
    """Return resources for the 311 package."""
    url = f"{BASE_URL}/api/3/action/package_show"
    resp = requests.get(url, params={"id": PACKAGE_ID}, timeout=30)
    resp.raise_for_status()
    package = resp.json()
    if not package["success"]:
        raise RuntimeError(f"CKAN API error: {package}")

    resources = []
    for r in package["result"]["resources"]:
        fmt = r.get("format", "").upper()
        name = r.get("name", "")
        # Skip the readme when data_only=True
        if data_only and "readme" in name.lower():
            continue
        resources.append({
            "name":             name,
            "url":              r.get("url", ""),
            "id":               r["id"],
            "format":           fmt,
            "datastore_active": r.get("datastore_active", False),
        })
    return resources


def _download_zip(url: str, name: str) -> pd.DataFrame:
    """Download a ZIP file, extract the CSV inside, return a DataFrame."""
    import zipfile

    print(f"  Downloading: {name} ...", flush=True)
    resp = requests.get(url, timeout=180, stream=True)
    resp.raise_for_status()

    # Read the zip from memory
    zip_bytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_bytes) as zf:
        # Find the CSV inside (ignore __MACOSX junk files)
        csv_files = [f for f in zf.namelist()
                     if f.lower().endswith(".csv") and not f.startswith("__")]
        if not csv_files:
            raise ValueError(f"No CSV found in ZIP for {name}. Contents: {zf.namelist()}")
        csv_name = csv_files[0]
        print(f"    Extracting: {csv_name}", flush=True)
        with zf.open(csv_name) as f:
            raw = f.read()
        # Try encodings in order — government CSVs are often cp1252 (Windows Latin)
        for enc in ("utf-8", "cp1252", "latin-1", "iso-8859-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), low_memory=False,
                                 encoding=enc, on_bad_lines="skip")
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            raise ValueError(f"Could not decode CSV with any known encoding")

    print(f"    → {len(df):,} rows, {len(df.columns)} columns", flush=True)
    return df


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names and types across yearly files."""
    # Lowercase + underscore column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename all known variants to standard names
    rename_map = {
        "open_date":                      "creation_date",
        "creation_date":                  "creation_date",
        "servicerequesttype":             "service_request_type",
        "service_request_type":           "service_request_type",
        "service_request_id":             "service_request_id",
        "closed_date":                    "closed_date",
        "ward":                           "ward_raw",
        "division":                       "division",
        "section":                        "section",
        "status":                         "status",
        "fsa":                            "fsa",
        "first_3_chars_of_postal_code":   "fsa",   # API name
        "intersection_street_1":          "street_1",
        "intersection_street_2":          "street_2",
    }
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # Parse dates
    for col in ("creation_date", "closed_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ward: handle both "19" (old local files) and "Beaches-East York (19)" (API)
    if "ward_raw" in df.columns:
        ward_str = df["ward_raw"].astype(str)
        # Try extracting number from parentheses first: "Name (19)" → 19
        extracted = ward_str.str.extract(r"\((\d+)\)", expand=False)
        # Fall back to direct numeric: "19" → 19
        direct = pd.to_numeric(ward_str, errors="coerce")
        df["ward"] = pd.to_numeric(extracted, errors="coerce").fillna(direct)
        df["ward_name"] = ward_str.str.replace(r"\s*\(\d+\)\s*", "", regex=True).str.strip()
        df.drop(columns=["ward_raw"], inplace=True)

    # Derived time columns
    if "creation_date" in df.columns:
        df["year"]  = df["creation_date"].dt.year
        df["month"] = df["creation_date"].dt.month

    return df


def _cache_valid() -> bool:
    """True if cache file exists and is fresh."""
    if not CACHE_FILE.exists():
        return False
    age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
    return age_hours < CACHE_HOURS


# ── Main public function ──────────────────────────────────────────────────────

def load_311_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the 311 dataset.  Uses local cache if available and fresh.

    Parameters
    ----------
    force_refresh : bool
        If True, ignores the cache and re-downloads from the API.

    Returns
    -------
    pd.DataFrame  with standardised columns.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if not force_refresh and _cache_valid():
        print(f"Loading from cache: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
        print(f"  → {len(df):,} rows loaded from cache")
        return df

    print("Fetching resource list from Toronto Open Data API...")
    url = f"{BASE_URL}/api/3/action/package_show"
    resp = requests.get(url, params={"id": PACKAGE_ID}, timeout=30)
    resp.raise_for_status()
    all_resources = resp.json()["result"]["resources"]

    # Only ZIP data files, skip the readme
    resources = [r for r in all_resources
                 if r.get("format", "").upper() == "ZIP"
                 and "readme" not in r.get("name", "").lower()
                 and r.get("url")]

    print(f"  Found {len(resources)} yearly ZIP files (2010-2026)")

    print("\nDownloading data files...")
    frames = []
    for r in resources:
        try:
            df = _download_zip(r["url"], r["name"])
            df = _normalise(df)
            frames.append(df)
        except Exception as e:
            print(f"  ⚠  Skipping {r['name']}: {e}")

    if not frames:
        raise RuntimeError("No data could be downloaded. Check your internet connection.")

    print("\nCombining all years...")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["service_request_id"] if "service_request_id" in combined.columns else None)
    combined = combined.sort_values("creation_date").reset_index(drop=True)

    print(f"  Total rows: {len(combined):,}")
    print(f"  Date range: {combined['creation_date'].min()} → {combined['creation_date'].max()}")
    print(f"  Divisions:  {sorted(combined['division'].dropna().unique())}")

    print(f"\nSaving to cache: {CACHE_FILE}")
    combined.to_parquet(CACHE_FILE, index=False)
    print("  Cache saved.")

    return combined


# ── Run standalone to test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Toronto 311 Open Data — API Connection Test")
    print("=" * 60)

    # Step 1: List ALL resources (including readme) so we can see everything
    print("\nStep 1: Listing ALL resources in the package...")
    try:
        url = f"{BASE_URL}/api/3/action/package_show"
        resp = requests.get(url, params={"id": PACKAGE_ID}, timeout=30)
        all_resources = resp.json()["result"]["resources"]
        print(f"  Total resources found: {len(all_resources)}")
        print()
        for i, r in enumerate(all_resources):
            print(f"  [{i}] name:             {r.get('name')}")
            print(f"       format:           {r.get('format')}")
            print(f"       datastore_active: {r.get('datastore_active')}")
            print(f"       id:               {r.get('id')}")
            print(f"       url:              {r.get('url', 'N/A')[:120]}")
            print()
    except Exception as e:
        print(f"  ❌  Error: {e}")
        exit(1)

    # Step 2: Download ONE year to check column names
    print("\nStep 2: Downloading most recent year to verify column names...")
    data_resources = [r for r in all_resources
                      if r.get("format", "").upper() == "ZIP"
                      and "readme" not in r.get("name", "").lower()]
    if not data_resources:
        print("  ❌  No ZIP data resources found.")
    else:
        r = data_resources[0]  # most recent year
        try:
            df_sample = _download_zip(r["url"], r["name"])
            print(f"\n  Raw columns from '{r['name']}':")
            for c in df_sample.columns:
                print(f"    '{c}'  — sample value: {str(df_sample[c].iloc[0])[:60]}")

            df_norm = _normalise(df_sample)
            print(f"\n  After normalisation: {list(df_norm.columns)}")
            print(f"\n  ✅  ZIP download working. Column names confirmed above.")
        except Exception as e:
            print(f"  ❌  Download error: {e}")
