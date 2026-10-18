import sys
sys.dont_write_bytecode=True
import json, unittest, tempfile
from pathlib import Path
import numpy as np
from affine import Affine
import audit as a
from finish import parse_utc

class AuditTests(unittest.TestCase):
    def test_native_center_convention(self):
        m=a.geo.PixelMap(Affine(2,0,10,0,-2,40),'EPSG:32610',Affine(1,0,10,0,-1,40),'EPSG:32610')
        np.testing.assert_allclose(m([[0,0],[5,6]]),[[.5,.5],[10.5,12.5]],atol=1e-12)

    def test_window_origin_translation(self):
        t=Affine(.3,.1,20,.05,-.4,80)
        crop=t*Affine.translation(100,200)
        m=a.geo.PixelMap(crop,'EPSG:32610',t,'EPSG:32610')
        np.testing.assert_allclose(m([[0,0],[511,511]]),[[100,200],[611,711]],atol=1e-10)

    def test_cross_crs(self):
        x,y=a.transform('EPSG:4326','EPSG:32610',[-122.4],[37.8])
        m=a.geo.PixelMap(Affine(1e-5,0,-122.4,0,-1e-5,37.8),'EPSG:4326',Affine(1,0,x[0],0,-1,y[0]),'EPSG:32610')
        p=np.array([[0,0],[40,120],[511,511]],float)
        np.testing.assert_allclose(m.reverse()(m(p)),p,atol=1e-6)

    def test_valid_zero_and_nodata(self):
        raw=np.array([[0.,1.],[100.,-9999.]])
        d,m,n=a.db_image(raw,np.array([[1,1],[1,0]],bool),'linear_amplitude')
        self.assertTrue(m[0,0]);self.assertFalse(m[1,1]);self.assertEqual(n['valid_zero_count'],1)
        self.assertEqual(d[0,0],0);self.assertEqual(d[1,0],40)

    def test_power_amplitude_equivalence(self):
        raw=np.array([[.01,.1],[1.,100.]])
        p,_,_=a.db_image(raw,np.ones((2,2),bool),'linear_power')
        d,_,_=a.db_image(np.sqrt(raw),np.ones((2,2),bool),'linear_amplitude')
        np.testing.assert_allclose(p,d)

    def test_no_bilinear_crossing_invalid(self):
        m=np.ones((4,4),bool);m[2,2]=False
        self.assertFalse(a.geo.sample_valid(m,[[1.5,1.5]])[0])
        self.assertTrue(a.geo.sample_valid(m,[[1,1]])[0])
        self.assertFalse(a.geo.sample_valid(m,[[4,1]])[0])

    def test_metric_intersection_large_northing(self):
        p=np.array([[0,0],[20,0],[20,20],[0,20]],float)+[500000,5245000]
        q=p+[10,0]
        area,inter,ar,am=a.intersection(p,q)
        self.assertAlmostEqual(area,200,places=4);self.assertAlmostEqual(ar,400,places=4)
        self.assertAlmostEqual(am,400,places=4)

    def test_unknown_is_not_gt(self):
        rows=json.loads((a.ROOT/'candidates.json').read_text())
        self.assertGreaterEqual(len(rows),8);self.assertLessEqual(len(rows),12)
        self.assertTrue(all(not r['GT_qualified'] and r['static_physical_correspondence']=='pending' for r in rows))

    def test_five_native_contexts(self):
        for sid in a.FIVE:
            r=json.loads((a.ROOT/'pairs'/sid/'audit.json').read_text())
            self.assertTrue(r['engineering_metadata_checks_passed'])
            self.assertTrue(r['distinct_acquisition_and_date'])
            for z in r['checks']:
                c=z['context']
                self.assertEqual(c['offset_xy'],[256,256])
                self.assertTrue(all(c[k] for k in ['same_url','pixels_equal_on_valid','validity_equal','transform_equal','source_window_transform_equal']))

    def test_identity_ledger(self):
        rows=json.loads((a.ROOT/'audit/identity_ledger.json').read_text())
        for r in rows:
            self.assertEqual(r['physical_collect_id'],r['indexed_acquisition_id'])
            self.assertTrue(r['datetime_agrees'])
            self.assertIn(r['identity_verified_against_embedded'],[None,True])

    def test_no_matching_or_annotation_input(self):
        # Enforce concrete direct dependencies; geography module has no matching code.
        text=(a.ROOT/'code/audit.py').read_text()
        self.assertNotIn('model_grades.jsonl',text)
        self.assertNotIn('annotation_store',text)
        self.assertNotIn('pseudo_labels/',text)
        self.assertNotIn('lightglue',text.lower())

    def test_python310_nanosecond_dates(self):
        self.assertEqual(parse_utc('2025-03-06T14:46:36.184635418Z'),parse_utc('2025-03-06T14:46:36.184635+00:00'))

    def test_sealed_output_rejected(self):
        original=a.ROOT
        try:
            with tempfile.TemporaryDirectory(prefix='sar_audit_guard_',dir='/tmp') as d:
                a.ROOT=Path(d);(a.ROOT/'DELIVERY_COMPLETE.json').write_text('{}')
                with self.assertRaises(RuntimeError):a.scan()
                self.assertEqual(sorted(x.name for x in a.ROOT.iterdir()),['DELIVERY_COMPLETE.json'])
        finally:a.ROOT=original

if __name__=='__main__':unittest.main(verbosity=2)
