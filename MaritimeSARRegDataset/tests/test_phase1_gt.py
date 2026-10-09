"""Independent known geometry and small corruption regressions; fixtures are fake."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sarreg.geometry import overlap_mask, transform_points, warp
from sarreg.io import read_gray, write_png
from sarreg.qa import validate_generated, validate_pair
from sarreg.synthesis import synthesize, load_cfg, generate
from sarreg.indexer import prepare_index
from sarreg.toy import make_toy


def test_known_rotation_translation_and_sea_projection():
    # x'= -y+50, y'=x+3. Expected values calculated explicitly.
    H = np.array([[0., -1., 50.], [1., 0., 3.], [0., 0., 1.]])
    points = np.array([[10., 20.], [22., 7.], [1., 1.]])
    np.testing.assert_allclose(transform_points(H, points), [[30, 13], [43, 25], [49, 4]])
    sea = np.zeros((64, 64), np.uint8); sea[5:15, 8:18] = 255
    expected = np.zeros_like(sea)
    # Do not call the generator's transform or warp for the oracle.
    for y in range(5, 15):
        for x in range(8, 18):
            expected[x+3, 50-y] = 255
    got = warp(sea, H, 64, mask=True)
    np.testing.assert_array_equal(got, expected)
    from scripts.summarize_phase1 import nearest_reference
    np.testing.assert_array_equal(nearest_reference(sea, H, 64), expected)
    restored = warp(got, np.linalg.inv(H), 64, mask=True)
    np.testing.assert_array_equal(restored, sea)
    a = np.full((64, 64), 255, np.uint8)
    T = np.array([[1., 0., 7.], [0., 1., -4.], [0., 0., 1.]])
    oi, oj = overlap_mask(a, a, T)
    ei = np.zeros_like(a); ei[4:, :57] = 255
    ej = np.zeros_like(a); ej[:60, 7:] = 255
    np.testing.assert_array_equal(oi, ei)
    np.testing.assert_array_equal(oj, ej)


def pair_fixture(tmp_path):
    image = np.full((512, 512), 52, np.uint8)
    image[90:150, 250:280] = 190
    source = tmp_path/'source.png'; write_png(source, image)
    row = {'dataset': 'generic', 'source_id': 'fixture', 'source_group': 'fixture',
           'image_path': str(source), 'split': 'train', 'annotations': []}
    folder = tmp_path/'out'/'train'/'pair_000000'
    synthesize(row, load_cfg('configs/easy.yaml'), folder, seed=23)
    return folder


def test_validator_rejects_matches_in_dynamic_region(tmp_path):
    folder = pair_fixture(tmp_path)
    with np.load(folder/'matches_0_1.npz') as pairs:
        x, y = np.rint(pairs['points_i'][0]).astype(int)
    dynamic = read_gray(folder/'dynamic_0.png'); dynamic[y, x] = 255
    write_png(folder/'dynamic_0.png', dynamic)
    result = validate_pair(folder)
    assert not result['passed']
    assert any('dynamic' in x.lower() for x in result['issues'])


def test_validator_records_missing_match_file(tmp_path):
    folder = pair_fixture(tmp_path)
    (folder/'matches_0_1.npz').unlink()
    result = validate_generated(folder.parents[1])
    assert result['failed'] == 1 and result['checked'] == 1


def test_existing_generation_preserves_manifest_and_config(tmp_path):
    root = tmp_path/'toy'; make_toy(root, 3, 12)
    index = tmp_path/'index.jsonl'; prepare_index('hrsid', root, index)
    out = tmp_path/'generated'
    assert generate(index, 'configs/easy.yaml', out, split='all', count=1)['generated'] == 1
    files = list(out.rglob('*')); before = {p: p.read_bytes() for p in files if p.is_file()}
    with pytest.raises(FileExistsError):
        generate(index, 'configs/easy.yaml', out, split='all', count=1, seed=99)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_duplicate_split_components_include_background_and_keep_official():
    from scripts.make_leakage_safe_index import derive
    rows = [
        {'dataset': 'hrsid', 'source_group': 'P1', 'image_path': '/a', 'split': 'train', 'official_split': 'train'},
        {'dataset': 'hrsid', 'source_group': 'P2', 'image_path': '/b', 'split': 'val', 'official_split': 'train'},
        {'dataset': 'background', 'source_group': 'P1', 'image_path': '/c', 'split': 'train', 'official_split': None},
        {'dataset': 'ssdd', 'source_group': '1', 'image_path': '/d', 'split': 'test', 'official_split': 'test'},
        {'dataset': 'ssdd', 'source_group': '2', 'image_path': '/e', 'split': 'train', 'official_split': 'train'}]
    audit = {'byte_identical_sources': [[{'path': '/a'}, {'path': '/b'}], [{'path': '/d'}, {'path': '/e'}]]}
    output, changes = derive(rows, audit)
    assert [r['split'] for r in output] == ['val', 'val', 'val', 'test', 'test']
    assert [r['official_split'] for r in output] == [r['official_split'] for r in rows]
    assert len(output) == len(rows) and len(changes) == 3
    assert output[0]['source_group'] == output[1]['source_group'] == output[2]['source_group']


def test_preview_red_reference_cyan_aligned_and_mask_panels(tmp_path):
    from sarreg.qa import visualize_pair
    folder = pair_fixture(tmp_path)
    output = tmp_path/'preview.png'; visualize_pair(folder, output)
    meta = json.loads((folder/'meta.json').read_text()); size = meta['output_size']
    display = cv2.imread(str(output))
    assert display.shape == (2*(size+28), 3*size, 3)
    overlay = display[28:size+28, size*2:size*3]
    reference = read_gray(folder/'view_0.png')
    secondary = cv2.warpPerspective(read_gray(folder/'view_1.png'), np.asarray(meta['pairs'][0]['H_j_to_i']), (size,size))
    allowed = read_gray(folder/'overlap_0_1_in_0.png')
    with np.load(folder/'matches_0_1.npz') as data:
        for x,y in data['points_i'][:40]:
            cv2.circle(allowed,(int(round(x)),int(round(y))),3,0,-1)
    keep = allowed>0
    np.testing.assert_array_equal(overlay[:,:,2][keep],reference[keep])
    np.testing.assert_array_equal(overlay[:,:,0][keep],secondary[keep])
    np.testing.assert_array_equal(overlay[:,:,1][keep],secondary[keep])


def test_validator_rejects_nonfinite_matrix(tmp_path):
    folder=pair_fixture(tmp_path)
    meta=json.loads((folder/'meta.json').read_text())
    meta['H_source_to_view'][0][0][0]=float('nan')
    (folder/'meta.json').write_text(json.dumps(meta))
    result=validate_pair(folder)
    assert not result['passed']
    assert any('non-finite' in issue for issue in result['issues'])
