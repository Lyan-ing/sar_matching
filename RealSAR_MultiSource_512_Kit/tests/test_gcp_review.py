from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_gcps import fit_affine,validate_gcps
import numpy as np

def test_affine_fit():
    a=np.array([[0,0],[0,100],[100,0],[100,100]],float)
    b=np.column_stack((a[:,0]+5,a[:,1]-3))
    T=fit_affine(a,b)
    assert np.allclose(np.column_stack((a,np.ones(len(a))))@T,b)

def test_independent_checks(tmp_path):
    p=tmp_path/'gcp.csv'
    with p.open('w') as f:
        f.write('pair_id,chip_id,x_a,y_a,x_b,y_b,role,reviewer,source\n')
        for i,(x,y) in enumerate([(10,10),(950,10),(10,950),(950,950),(150,150),(850,150),(150,850),(850,850)]):
            role='control' if i<4 else 'check'
            f.write(f'p1,chip_000,{x},{y},{x+2},{y+1},{role},human,independent_annotation\n')
    s=validate_gcps(p,1000)
    assert s['chips_passed_affine_check']==1
