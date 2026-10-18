"""Focused synthetic invariants and direct-input real-data acceptance; no human GT."""
import sys
sys.dont_write_bytecode=True
import json,tempfile,unittest,ast,builtins
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from affine import Affine
from PIL import Image
from geo_tools import *
from build import utc,components
from registration import register,fit
from evaluation import evaluate,qualified_quality
from readability_server import Reviews
class GeometryTests(unittest.TestCase):
    def test_nanosecond_identity_date(self):self.assertEqual(utc('2025-03-06T14:46:36.184635418Z').microsecond,184635)
    def test_pixel_centers_scales_and_CRS(self):
        a=Affine(2,0,500000,0,-2,4500000);b=Affine(4,0,500000,0,-4,4500000);p=np.array([[0,0],[19.25,20.5]])
        m=PixelMap(a,'EPSG:32610',b,'EPSG:32610');np.testing.assert_allclose(m(p),(p+.5)/2-.5,atol=1e-9)
        cross=PixelMap(a,'EPSG:32610',Affine(.001,0,-124,0,-.001,41),'EPSG:4326');np.testing.assert_allclose(cross.reverse()(cross(p)),p,atol=1e-7);self.assertIsNone(cross.affine_matrix())
    def test_native_window_zero_NoData(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            p=Path(d)/'source.tif';a=np.arange(100,dtype=np.float32).reshape(10,10);v=np.ones_like(a,bool);v[3,3]=False;z=dict(crs='EPSG:32610',transform=[1,0,500000,0,-1,4500000]);write_tif(p,a,v,z)
            with rasterio.open(p) as ds:
                raw,mask,zz=read_window(ds,Window(0,0,6,6),'linear_amplitude',True);self.assertTrue(mask[0,0]);self.assertFalse(mask[3,3]);np.testing.assert_array_equal(raw,a[:6,:6])
                with self.assertRaises(ValueError):read_window(ds,Window(-1,0,4,4),'linear_amplitude')
                with self.assertRaises(FileExistsError):write_tif(p,a,v,z)
    def test_common_sampling_conservative_mask(self):
        a=np.arange(100,dtype=np.float32).reshape(10,10);v=np.ones_like(a,bool);v[4,4]=False;z=dict(crs='EPSG:32610',transform=[1,0,500000,0,-1,4500000],width=10,height=10)
        raw,mask=sample_grid(a,v,z,z);np.testing.assert_array_equal(mask,v);np.testing.assert_allclose(raw[v],a[v])
        dest=dict(z,transform=[1,0,500000.5,0,-1,4499999.5]);raw,mask=sample_grid(a,v,z,dest);self.assertFalse(mask[3,3]);self.assertFalse(mask[4,4]);self.assertFalse(mask[-1,-1])
    def test_ROI_mapping(self):
        z=dict(crs='EPSG:32610',transform=[2,0,500000,0,-2,4500000],width=100,height=100);roi=box(500020,4499880,500100,4499960);w=roi_window(z,roi,z['crs'],pad=0)
        self.assertEqual((w.col_off,w.row_off,w.width,w.height),(10,20,40,40))
    def test_dependency_components(self):
        r=[dict(pair_id='a',sensor='x',site_name='port1',component='old1',views=[dict(acquisition_id='s1'),dict(acquisition_id='s2')]),dict(pair_id='b',sensor='x',site_name='port2',component='old2',views=[dict(acquisition_id='s2'),dict(acquisition_id='s3')]),dict(pair_id='c',sensor='y',site_name='port1',component='old3',views=[dict(acquisition_id='q1'),dict(acquisition_id='q2')]),dict(pair_id='d',sensor='y',site_name='port3',component='old4',views=[dict(acquisition_id='z1'),dict(acquisition_id='z2')])]
        c=components(r);self.assertEqual(c['a'],c['b']);self.assertEqual(c['a'],c['c']);self.assertNotEqual(c['a'],c['d'])
    def test_global_local_geometry(self):
        p=geo.regular_grid((512,512),8);q=p+np.array([3.,-2.]);e,d=fit(p,q,(512,512),(512,512));self.assertTrue(e.accepted);np.testing.assert_allclose(e.H,[[1,0,3],[0,1,-2],[0,0,1]],atol=1e-5)
        e,_=fit(p[:3],q[:3],(512,512),(512,512));self.assertFalse(e.accepted)
    def test_predictor_no_GT_or_geography_A(self):
        images=np.zeros((512,512),np.uint8)
        with self.assertRaises(ValueError):register(images,images,'orb','A',geography=[{},{}])
        with self.assertRaises(TypeError):register(images,images,'orb',manual_points=[])
        import registration,inspect
        text=inspect.getsource(registration);self.assertNotIn('from evaluation',text);self.assertNotIn('benchmark_store',text);self.assertNotIn('/annotations',text)
        result=register(images,images,'orb');self.assertEqual(result['decision'],'no_output')
    def test_pending_GT_does_not_compute_success(self):
        r=lines(ROOT/'regions.jsonl')[0];t=dict(sample_id=r['sample_id'],height0=r['views'][0]['native_info']['height'],width0=r['views'][0]['native_info']['width'],height1=r['views'][1]['native_info']['height'],width1=r['views'][1]['native_info']['width'])
        out=evaluate(r,dict(method='georeference'),dict(sample_id=t['sample_id'],points=[]),t);self.assertFalse(out['Success_computed']);self.assertFalse(out['independent_accuracy_available'])
    def test_simulated_independent_checks_exposure(self):
        r=lines(ROOT/'regions.jsonl')[0];sid=r['sample_id'];z0,z1=[v['native_info'] for v in r['views']];t=dict(sample_id=sid,height0=z0['height'],width0=z0['width'],height1=z1['height'],width1=z1['width']);p=np.array([[80,80],[250,80],[80,250],[250,250]],float);q=p.copy();pts=[]
        for i,(a,b) in enumerate(zip(p,q)):
            pts.append(dict(sample_id=sid,point_id=f'SIMULATED_P{i}',xy0=a.tolist(),xy1=b.tolist(),role='check',annotator='SIMULATED_A',annotated_at='SIMULATED',landmark_type='SIMULATED_STATIC',is_static=True,visible_both=True,confidence='medium',sigma0_px=.5,sigma1_px=.5,used_assistance=False,review=dict(reviewer='SIMULATED_B',reviewed_at='SIMULATED',xy0=a.tolist(),xy1=b.tolist(),conclusion='confirmed',static_confirmed=True,visible_confirmed=True,sigma0_px=.5,sigma1_px=.5)))
        doc=dict(sample_id=sid,points=pts,evaluation_scope='local',disposition='local_only');pred=dict(method='SIMULATED',accepted=True,global_H=np.eye(3).tolist(),sampled_support_polygon=[[0,0],[400,0],[400,400],[0,400]])
        result=evaluate(r,pred,doc,t);self.assertTrue(result['sparse_Success@3px']);self.assertEqual(result['n_checks'],4)
        # This is a synthetic unit test, never exported as real human evidence.
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            folder=Path(d)/'readability';folder.mkdir();(folder/'assistance_events.jsonl').write_text(json.dumps(dict(sample_id=sid,actor='SIMULATED_A',source='common_fov_geographic_grid'))+'\n');q=qualified_quality(doc,t,Path(d));self.assertEqual(q['eligible_checks'],0)
    def test_quick_review_immutable(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);(root/'regions.jsonl').write_text(json.dumps(dict(sample_id='SIMULATED'))+'\n');store=Reviews(root)
            p=dict(sample_id='SIMULATED',expected_revision=0,actor='SIMULATED_ONLY',verdict='C',note='Synthetic test only');doc=store.commit(p);self.assertFalse(doc['GT_qualified']);old=(root/'readability/SIMULATED/v000001.json').read_bytes()
            with self.assertRaises(ValueError):store.commit(p)
            p.update(expected_revision=1,verdict='A');store.commit(p);self.assertEqual((root/'readability/SIMULATED/v000001.json').read_bytes(),old)
            p.update(expected_revision=2,GT_qualified=True)
            with self.assertRaises(ValueError):store.commit(p)
    def test_real_native_and_common_acceptance(self):
        for r in lines(ROOT/'regions.jsonl'):
            folder=ROOT/'regions'/r['sample_id']
            for i,v in enumerate(r['views']):
                ca,cm,cz=read_tif(folder/f'context{i}.tif');na,nm,nz=read_tif(folder/f'native{i}.tif');x,y,w,h=v['context_native_window_xywh'];np.testing.assert_array_equal(na,ca[y:y+h,x:x+w]);np.testing.assert_array_equal(nm,cm[y:y+h,x:x+w]);np.testing.assert_allclose(nz['transform'],list(Affine(*cz['transform'])*Affine.translation(x,y))[:6],atol=1e-8)
                self.assertGreater(min(ca.shape),512);common,vm,zz=read_tif(folder/f'common{i}.tif');expected,ev=sample_grid(ca,cm,cz,zz);np.testing.assert_array_equal(vm,ev);np.testing.assert_allclose(common[vm],expected[vm],atol=0)
                self.assertEqual(common.shape,(512,512));self.assertGreater(min(na.shape),255)
                if v['context_provenance']['actual_read']=='existing_unresampled_context':
                    a,b,z=read_tif(v['context_provenance']['path']);np.testing.assert_array_equal(ca,a);np.testing.assert_array_equal(cm,b)
            joint=np.array(Image.open(folder/'common_joint_geographic_valid.png'))>0;self.assertGreaterEqual(joint.mean(),.75);self.assertFalse(r['gt_qualified'])
    def test_synthetic_demo_GT(self):
        folder=ROOT/'training_examples/single_image_known_geometry';m=json.loads((folder/'meta.json').read_text());self.assertEqual(m['source_resize_factors_xy'],[1.,1.]);self.assertEqual(m['output_size'],512);p=m['pairs'][0];z=np.load(folder/'matches_0_1.npz')
        from sarbench.estimate import perspective
        np.testing.assert_allclose(perspective(p['H_i_to_j'],z['points_i']),z['points_j'],atol=1e-4)
    def test_training_contract_and_no_real_positives(self):
        schema=json.loads((ROOT/'training_examples/schema.json').read_text());rows=lines(ROOT/'training_examples/normalized_index.jsonl')
        self.assertEqual(len(rows),24)
        for r in rows:
            self.assertTrue(set(schema['required_fields'])<=set(r));self.assertFalse(r['allow_training_now'])
            if r['layer']=='difficult_unknown':self.assertIsNone(r['positive_label']);self.assertIsNone(r['negative_label'])
        from training_examples import export_real_local
        r=lines(ROOT/'regions.jsonl')[0]
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            out=Path(d)/'forbidden';task=dict(sample_id=r['sample_id'],height0=r['views'][0]['native_info']['height'],width0=r['views'][0]['native_info']['width'],height1=r['views'][1]['native_info']['height'],width1=r['views'][1]['native_info']['width'])
            with self.assertRaises(ValueError):export_real_local(r,dict(sample_id=r['sample_id'],points=[]),task,out)
            self.assertFalse(out.exists())
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    save(ROOT/('unit_test_results_'+stamp+'.json'),dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),simulated_check_fixtures_not_real_GT=True))
    sys.exit(0 if result.wasSuccessful() else 1)
