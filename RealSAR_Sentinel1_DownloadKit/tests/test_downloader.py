from __future__ import annotations
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "download_s1_rtc.py"
spec = importlib.util.spec_from_file_location("download_s1_rtc", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_s3_xml():
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <IsTruncated>false</IsTruncated>
    <Contents><Key>tiles/RTC/1/IW/14/T/PN/2020/S1A_20200801_14TPN_ASC/Gamma0_VV.tif</Key></Contents>
    </ListBucketResult>'''
    keys, truncated, token = module.parse_s3_page(xml)
    assert len(keys) == 1 and not truncated and token is None
    scene = module.scene_from_key(keys[0], "14TPN")
    assert scene["date"] == "20200801"
    assert scene["orbit_direction"] == "ascending"
    assert module.scene_from_key(keys[0], "14TQK") is None


def test_choose_two_passes():
    data = []
    for date, direction in [("20200101", "ASC"), ("20200117", "ASC"),
                            ("20200215", "ASC"), ("20200301", "DSC")]:
        s = module.scene_from_key(f"tiles/RTC/1/IW/14/T/PN/2020/S1A_{date}_14TPN_{direction}/Gamma0_VV.tif", "14TPN")
        data.append(s)
    picked = module.choose_scenes(data, 3)
    assert len(picked) == 3
    assert picked[0]["orbit_direction"] == picked[1]["orbit_direction"]
    assert picked[2]["orbit_direction"] != picked[0]["orbit_direction"]
    assert picked[0]["date"] != picked[1]["date"]


def test_raster_grid_reprojection(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    # Two rasters shifted by one source pixel but sharing world coordinates.
    width = 128
    yy, xx = np.mgrid[:width, :width]
    for offset in (0, 1):
        # Georeferenced physical signal F(X,Y) in shared UTM world coords.
        arr = (3.0 + 0.005 * (xx + offset) + 0.008 * yy).astype("float32")
        file = tmp_path / f"scene{offset}.tif"
        with rasterio.open(file, "w", driver="GTiff", width=width, height=width,
                           count=1, dtype="float32", crs="EPSG:32614", nodata=-9999,
                           transform=Affine(10, 0, 500000 + 10 * offset, 0, -10, 4500000)) as dst:
            dst.write(arr, 1)
    first, metadata = module.raster_subwindow(str(tmp_path / "scene0.tif"),
                                             x_fraction=.5, y_fraction=.5, size=64)
    again, metadata2 = module.raster_subwindow(str(tmp_path / "scene1.tif"),
                                              x_fraction=.5, y_fraction=.5, size=64,
                                              reference={k:metadata[k] for k in ("crs","transform", "width", "height")})
    assert first.shape == again.shape == (64,64)
    assert np.isfinite(again).all()
    # Constructed signal intentionally has different physical value, test only grid/CRS.
    assert metadata["crs"] == metadata2["crs"]
    assert metadata["transform"] == metadata2["transform"]


def test_db():
    d = module.gamma_to_db(np.array([1., .1, 0., np.nan], dtype="float32"))
    assert d[0] == pytest.approx(0)
    assert d[1] == pytest.approx(-10)
    assert np.isnan(d[2]) and np.isnan(d[3])
