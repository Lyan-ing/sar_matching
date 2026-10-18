#!/usr/bin/env python3
"""Download a tiny real Sentinel-1 multi-acquisition SAR study set.

Source: AWS Registry of Open Data / Indigo RTC (public anonymous S3 objects).
No ground-truth pixel correspondence is implied by RTC geocoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

BUCKET = "sentinel-s1-rtc-indigo"
ENDPOINT = f"https://{BUCKET}.s3.us-west-2.amazonaws.com"
SOURCE_URL = "https://registry.opendata.aws/sentinel-1-rtc-indigo/"
KEY_PATTERN = re.compile(r"/((S1[ABC])_(\d{8})_([0-9A-Z]{5})_(ASC|DSC))/Gamma0_VV\.tif$")


def parse_s3_page(xml_bytes: bytes) -> tuple[list[str], bool, str | None]:
    root = ET.fromstring(xml_bytes)
    nsp = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys = [(node.text or "") for node in root.findall(".//s:Contents/s:Key", nsp)]
    trunc = (root.findtext("s:IsTruncated", "false", nsp) or "false").lower() == "true"
    token = root.findtext("s:NextContinuationToken", None, nsp)
    if trunc and not token:
        raise RuntimeError("S3 response truncated without continuation token")
    return keys, trunc, token


def list_tile_keys(session: requests.Session, tile: str, year: int, timeout: int) -> list[str]:
    prefix = f"tiles/RTC/1/IW/{tile[:2]}/{tile[2]}/{tile[3:]}/{year}/"
    token = None
    keys = []
    for page in range(50):
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(ENDPOINT + "/", params=params, timeout=timeout)
        r.raise_for_status()
        batch, trunc, token = parse_s3_page(r.content)
        keys.extend(batch)
        if not trunc:
            return keys
    raise RuntimeError(f"Too many S3 listing pages for {prefix}; refusing ambiguous result")


def scene_from_key(key: str, tile: str) -> dict | None:
    m = KEY_PATTERN.search(key)
    if not m or m.group(4) != tile:
        return None
    name, satellite, date, _tile, direction = m.groups()
    return {
        "scene_name": name,
        "satellite": satellite,
        "date": date,
        "orbit_direction": "ascending" if direction == "ASC" else "descending",
        "key": key,
        "url": ENDPOINT + "/" + quote(key, safe="/"),
    }


def choose_scenes(scenes: list[dict], count: int, min_days: int = 12) -> list[dict]:
    """Choose at least two dates; prefer 2 of same direction and 1 opposite."""
    # One record per (satellite, date, orbit direction), oldest first.
    seen = set()
    unique = []
    for obj in sorted(scenes, key=lambda x: (x["date"], x["scene_name"])):
        k = (obj["satellite"], obj["date"], obj["orbit_direction"])
        if k not in seen:
            unique.append(obj)
            seen.add(k)
    if len(unique) < count:
        raise RuntimeError(f"Only {len(unique)} scenes found; need {count}")
    # Prefer two images with the same pass direction for fair repeat-pass comparison.
    groups = {d: [x for x in unique if x["orbit_direction"] == d]
              for d in ("ascending", "descending")}
    primary = max(groups, key=lambda k: len(groups[k]))
    pp = groups[primary]
    if len(pp) < 2:
        raise RuntimeError("No two acquisitions from the same orbit direction")
    selected = [pp[len(pp) // 3]]
    def days_between(a, b):
        fmt = "%Y%m%d"
        return abs((datetime.strptime(a["date"], fmt) - datetime.strptime(b["date"], fmt)).days)
    other_primary = [x for x in pp if x["scene_name"] != selected[0]["scene_name"]
                     and days_between(x, selected[0]) >= min_days]
    if not other_primary:
        raise RuntimeError(f"No repeat pass pair with separation >= {min_days} days")
    selected.append(max(other_primary, key=lambda x: days_between(x, selected[0])))
    if count > 2:
        other = [x for x in unique if x["orbit_direction"] != primary]
        if other:
            selected.append(other[len(other) // 2])
    remaining = [x for x in unique if x not in selected]
    while len(selected) < count:
        candidates = [x for x in remaining if min(days_between(x, y) for y in selected) >= min_days]
        if not candidates:
            raise RuntimeError("Insufficient temporally distinct acquisitions")
        pick = max(candidates, key=lambda x: min(days_between(x, y) for y in selected))
        selected.append(pick)
        remaining.remove(pick)
    return selected


def raster_subwindow(url: str, *, x_fraction: float, y_fraction: float, size: int,
                     reference=None) -> tuple[np.ndarray, dict]:
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window, transform as window_transform
        from rasterio.vrt import WarpedVRT
    except ImportError as exc:
        raise RuntimeError("Please install rasterio: python -m pip install rasterio") from exc
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
                      GDAL_HTTP_TIMEOUT="35", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(url) as src:
            if src.crs is None:
                raise RuntimeError(f"Input is not georeferenced: {url}")
            if reference is None:
                if src.width < size or src.height < size:
                    raise RuntimeError("Requested chip is larger than source raster")
                x0 = max(0, min(src.width - size, int(src.width * x_fraction - size / 2)))
                y0 = max(0, min(src.height - size, int(src.height * y_fraction - size / 2)))
                win = Window(x0, y0, size, size)
                arr = src.read(1, window=win, masked=True).filled(np.nan).astype("float32")
                affine = window_transform(win, src.transform)
                return arr, {"crs": src.crs.to_string(), "transform": list(affine)[:6],
                             "width": size, "height": size,
                             "original_crs": src.crs.to_string(), "original_size": [src.width, src.height],
                             "nodata": src.nodata}
            from affine import Affine
            dst_transform = Affine(*reference["transform"])
            with WarpedVRT(src, crs=reference["crs"], transform=dst_transform,
                           width=size, height=size, resampling=Resampling.bilinear,
                           nodata=src.nodata) as vrt:
                arr = vrt.read(1, masked=True).filled(np.nan).astype("float32")
            meta = dict(reference)
            meta.update(original_crs=src.crs.to_string(), original_size=[src.width, src.height],
                        nodata=src.nodata)
            return arr, meta


def gamma_to_db(arr: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(arr)
    db[~np.isfinite(db)] = np.nan
    return db


def save_data(scene: dict, arr: np.ndarray, ref: dict, outdir: Path,
              db_min: float, db_max: float) -> dict:
    import rasterio
    from affine import Affine
    from PIL import Image
    valid = np.isfinite(arr) & (arr > 0)
    coverage = float(valid.mean())
    db = gamma_to_db(arr)
    std = float(np.nanstd(db)) if valid.any() else 0.0
    if coverage < 0.70 or not (std > 0.5):
        raise RuntimeError(f"Low usable content in {scene['scene_name']}: valid={coverage:.1%} db_std={std:.3f}")
    outdir.mkdir(parents=True, exist_ok=True)
    name = scene["scene_name"]
    tiff = outdir / (name + "_linear_gamma0.tif")
    preview = outdir / (name + "_display.png")
    profile = {"driver": "GTiff", "width": arr.shape[1], "height": arr.shape[0],
               "count": 1, "dtype": "float32", "crs": ref["crs"],
               "transform": Affine(*ref["transform"]), "compress": "deflate", "nodata": -9999.0}
    with rasterio.open(tiff, "w", **profile) as ds:
        ds.write(np.where(valid, arr, -9999).astype("float32"), 1)
        ds.update_tags(SOURCE_SCENE=name, SAR_PRODUCT="Sentinel-1 RTC Gamma0 VV",
                       NOTE="True SAR acquisitions, map-grid aligned; NOT independent pixel GT")
    scaled = 255 * (np.clip(np.nan_to_num(db, nan=db_min), db_min, db_max) - db_min) / (db_max - db_min)
    Image.fromarray(np.uint8(np.round(scaled))).save(preview)
    summary = dict(scene)
    summary.update(geotiff=str(tiff.name), preview=str(preview.name),
                   valid_fraction=round(coverage, 6), db_std=round(std, 4),
                   geotiff_sha256=hashlib.sha256(tiff.read_bytes()).hexdigest())
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download small genuine repeat-pass Sentinel-1 RTC SAR chips")
    p.add_argument("--tile", default="14TPN", help="5-character MGRS tile; 14TPN is a documented public example")
    p.add_argument("--years", nargs="+", type=int, default=[2020], help="Acquisition years")
    p.add_argument("--n", type=int, default=3, help="Number of distinct acquisitions (min 2)")
    p.add_argument("--size", type=int, default=512, help="Square output chip width in pixels")
    p.add_argument("--x-fraction", type=float, default=0.5, help="Chip center fraction across source")
    p.add_argument("--y-fraction", type=float, default=0.5, help="Chip center fraction down source")
    p.add_argument("--min-days", type=int, default=12)
    p.add_argument("--db-min", type=float, default=-25)
    p.add_argument("--db-max", type=float, default=0)
    p.add_argument("--timeout", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("real_sar_samples"))
    p.add_argument("--discover-only", action="store_true", help="List acquisitions only; no raster downloads")
    args = p.parse_args(argv)
    if not re.fullmatch(r"\d{2}[A-Z]{3}", args.tile):
        p.error("--tile must be a 5-character MGRS tile, e.g. 14TPN")
    if args.n < 2 or args.size < 64 or not (0 < args.x_fraction < 1) or not (0 < args.y_fraction < 1):
        p.error("n >= 2, size >= 64, x/y-fraction strictly in (0,1) required")
    if not args.db_max > args.db_min:
        p.error("--db-max must exceed --db-min")
    if args.out.exists() and any(args.out.iterdir()):
        p.error(f"Refuse overwrite of nonempty output: {args.out}")
    session = requests.Session()
    session.headers.update({"User-Agent": "MaritimeSAR-RealSAR-Research-Downloader/0.1"})
    scene_by_key = {}
    for year in args.years:
        print(f"Searching anonymously: tile={args.tile}, year={year}", flush=True)
        keys = list_tile_keys(session, args.tile, year, args.timeout)
        for key in keys:
            scene = scene_from_key(key, args.tile)
            if scene:
                scene_by_key[key] = scene
    scenes = sorted(scene_by_key.values(), key=lambda obj: obj["date"])
    print(f"VV COG files found: {len(scenes)}", flush=True)
    if args.discover_only:
        print(json.dumps(scenes[:50], ensure_ascii=False, indent=2))
        return 0
    selected = choose_scenes(scenes, args.n, args.min_days)
    for i, scene in enumerate(selected):
        print(f"  #{i}: {scene['date']} {scene['orbit_direction']} {scene['url']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    reference = None
    entries = []
    for i, scene in enumerate(selected):
        print(f"Reading source COG window {i+1}/{len(selected)} ...", flush=True)
        array, metadata = raster_subwindow(scene["url"], x_fraction=args.x_fraction,
                                         y_fraction=args.y_fraction, size=args.size, reference=reference)
        if reference is None:
            reference = {"crs": metadata["crs"], "transform": metadata["transform"],
                         "width": args.size, "height": args.size}
        entry = save_data(scene, array, metadata, args.out, args.db_min, args.db_max)
        entries.append(entry)
    manifest = {
        "dataset_kind": "real_sentinel1_distinct_acquisitions_on_common_rtc_map_grid",
        "warning": "Not true cross-look pixel GT. RTC orthorectification is already applied."
                   "Same-direction observations are mainly repeat-pass; distinct geometry requires verified orbit/look metadata.",
        "source": SOURCE_URL,
        "bucket": BUCKET,
        "license": "Indigo Atlas Sentinel License; confirm terms before redistribution",
        "tile": args.tile, "requested_years": args.years, "band": "VV", "n_scenes": len(entries),
        "db_display_range": [args.db_min, args.db_max], "common_grid": reference,
        "scenes": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from PIL import Image
        previews = [Image.open(args.out / x["preview"]).convert("RGB") for x in entries]
        canvas = Image.new("RGB", (args.size * len(previews), args.size))
        for i, preview in enumerate(previews):
            canvas.paste(preview, (i * args.size, 0))
        canvas.save(args.out / "quicklook_strip.png")
    except Exception as exc:
        print(f"Preview montage failed (GeoTIFF data still saved): {exc}", file=sys.stderr)
    print(f"DONE: {args.out} ({len(entries)} real Sentinel-1 acquisitions)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.exceptions.RequestException as exc:
        sys.exit(f"Network/API error: {exc}\nTry direct Internet access or configure HTTPS_PROXY in your shell.")
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
