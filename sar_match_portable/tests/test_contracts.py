import itertools
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from sar_match.external import View,scan,make_pairs,category,annotation_diagnostics,position_labels,pixel_consistency
from sar_match.pipeline import verified_file,load_config
from sar_match.matchers import Prediction
from sar_match.estimate import estimate,perspective
from sar_match.reliability import geometry_diagnostics,gate


class Contracts(unittest.TestCase):
    def test_complete_18_and_153_without_cross_scene(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for scene in ['scene_000001','scene_000002']:
                folder=root/scene;folder.mkdir()
                for inc,az in itertools.product([20,30,45],[0,60,120,180,240,300]):
                    cv2.imwrite(str(folder/f'{scene}_inc{inc}_az{az:03d}.png'),np.zeros((8,8),np.uint8))
            records,warnings=scan(root);self.assertEqual(warnings,[])
            for r in records.values():
                self.assertTrue(r['complete']);pairs=make_pairs(r['views']);self.assertEqual(len(pairs),153)
                self.assertEqual(Counter(category(a,b) for a,b in pairs),{'same_inc_different_az':45,'same_az_different_inc':18,'both_different':90})
                self.assertEqual(len(make_pairs(r['views'],smoke=True)),3)
                self.assertEqual(len(make_pairs(r['views'],'same_inc')),45)
            with self.assertRaises(ValueError): category(records['scene_000001']['views'][0],records['scene_000002']['views'][0])

    def test_missing_duplicate_unreadable_and_unparseable(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'nested').mkdir()
            for f in [p/'scene_000001_inc20_az000.png',p/'nested/scene_000001_inc20_az000.png',p/'noise.png']:
                cv2.imwrite(str(f),np.zeros((8,8),np.uint8))
            (p/'scene_000001_inc30_az060.png').write_bytes(b'not a PNG')
            scenes,warnings=scan(p);types={w['type'] for w in warnings}
            self.assertTrue({'duplicate_view_excluded','missing_view','unreadable_image','unparseable_filename'}.issubset(types))
            self.assertEqual(len(scenes['scene_000001']['views']),1)
            self.assertFalse(scenes['scene_000001']['complete'])

    def test_no_gt_from_annotations_and_instance_ids(self):
        z=dict(matches_A=np.array([[2,2],[8,8],[2,2],[8,8]],float),matches_B=np.array([[2,2],[8,8],[8,8],[2,2]],float),inliers=np.array([1,1,0,0],bool))
        e={'complete':True,'boxes':[{'xyxy':[0,0,5,5],'instance_id':'boat1'}]}
        r=annotation_diagnostics([e,e],z,[(10,10)]*2,'.')
        self.assertEqual(r['region_counts'],dict(background=1,ship=1,cross_region=2,unknown=0));self.assertEqual(r['same_ship_instance']['same_id'],1)
        e2={'complete':True,'boxes':[{'xyxy':[0,0,5,5]}]}
        r=annotation_diagnostics([e2,e2],z,[(10,10)]*2,'.');self.assertEqual(r['same_ship_instance']['unknown_id'],1)
        self.assertFalse(annotation_diagnostics([None,e],z,[(10,10)]*2,'.')['annotations_available'])

    def test_partial_boxes_do_not_assert_background(self):
        labels,_=position_labels({'boxes':[{'xyxy':[0,0,5,5]}]},np.array([[2,2],[8,8]]),(10,10),'.')
        self.assertGreater(labels[0],0);self.assertEqual(labels[1],-2)

    def test_mask_and_ids(self):
        with tempfile.TemporaryDirectory() as d:
            mask=np.zeros((10,10),np.uint16);mask[:5,:5]=7;cv2.imwrite(str(Path(d)/'mask.png'),mask)
            labels,ids=position_labels({'complete':True,'mask':'mask.png','instance_ids':{'7':'ship7'}},np.array([[2,2],[8,8]]),(10,10),d)
            self.assertEqual(labels.tolist(),[7,0]);self.assertEqual(ids.tolist(),['ship7',None])

    def test_geometry_direction_and_original_full_gate(self):
        x,y=np.meshgrid(np.linspace(40,450,8),np.linspace(40,450,8));pts=np.c_[x.ravel(),y.ravel()].astype(np.float32)
        H=np.array([[1,0,5],[0,1,-7],[0,0,1]],float);p=Prediction('known test',pts,perspective(H,pts).astype(np.float32))
        e=estimate(p,(512,512));d=geometry_diagnostics(e,p,(512,512),(512,512));e=gate(e,d,'full')
        self.assertTrue(e.accepted);np.testing.assert_allclose(e.candidate_H,H,atol=1e-4)
        np.testing.assert_allclose(perspective(e.candidate_H,[[20,30]]),[[25,23]],atol=1e-4)

    def test_missing_or_tampered_weights_fail(self):
        with self.assertRaises(FileNotFoundError):verified_file({'file':'checkpoints/nonexistent.pth','sha256':'0'*64})
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'tampered.pth';p.write_bytes(b'not authorized model weights')
            with self.assertRaises(RuntimeError):verified_file({'file':str(p),'sha256':'0'*64})
        self.assertEqual(load_config()['extractor']['max_keypoints'],1024)

    @unittest.skipUnless((ROOT/'checkpoints/final_inference.pth').exists() and (ROOT/'sar_match/_vendor/superpoint.py').exists(), 'Authorized local weights/SuperPoint required')
    def test_final_model_blank_input_and_input_validation(self):
        from sar_match.pipeline import Matcher
        m=Matcher(device='cpu')
        r,z=m.predict(np.zeros((128,128),np.uint8),np.zeros((128,128),np.uint8))
        self.assertFalse(r['accepted'])
        self.assertEqual(r['n_matches'],0)
        self.assertEqual(r['failure_type'],'no_keypoints')
        self.assertEqual(z['H_AB'].shape,(0,3))
        self.assertEqual(z['inliers'].shape,(0,))
        with self.assertRaises(ValueError):m.predict(np.zeros((128,128),np.uint8),np.zeros((256,256),np.uint8))
        with self.assertRaises(ValueError):m.predict(np.zeros((512,512),np.uint16),np.zeros((512,512),np.uint16))
        with self.assertRaises(ValueError):m.predict(np.zeros((641,641),np.uint8),np.zeros((641,641),np.uint8))

    def test_pixel_diagnostics_do_not_assume_background(self):
        r=pixel_consistency(np.zeros((5,5),np.uint8),np.zeros((5,5),np.uint8));self.assertEqual(r['equal_position_pixel_fraction'],1.)
        self.assertIn('unconfirmed',r['background_reuse_note'])

if __name__=='__main__':unittest.main()
