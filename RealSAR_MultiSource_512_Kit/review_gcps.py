#!/usr/bin/env python3
"""Independent manual GCP quality report. Does not create GT from algorithmic correspondences."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

REQUIRED=('pair_id','chip_id','x_a','y_a','x_b','y_b','role','reviewer','source')

def fit_affine(a,b):
    # pB = [x_A, y_A, 1] @ theta ; no RANSAC: independent manual control points should be reviewed
    M=np.column_stack((a,np.ones(len(a))))
    if np.linalg.matrix_rank(M)<3:raise ValueError('Control points are collinear')
    theta=np.linalg.lstsq(M,b,rcond=None)[0]
    return theta

def validate_gcps(path,size,min_control=4,min_check=4,max_p90=3.):
    records=defaultdict(list)
    with open(path,newline='',encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not set(REQUIRED).issubset(rd.fieldnames or []):
            raise ValueError('Missing '+str(set(REQUIRED)-set(rd.fieldnames or [])))
        for row in rd:
            if row['role'] not in ('control','check'):raise ValueError('role must be control or check')
            if not row['reviewer'] or not row['source']:raise ValueError('Independent reviewer and source mandatory')
            for k in ('x_a','y_a','x_b','y_b'):
                row[k]=float(row[k]);
                if not (0<=row[k]<size):raise ValueError('GCP outside image: '+repr(row))
            records[(row['pair_id'],row['chip_id'])].append(row)
    result=[]
    for (pid,cid),items in records.items():
        a=np.array([[z['x_a'],z['y_a']] for z in items],float)
        b=np.array([[z['x_b'],z['y_b']] for z in items],float)
        controls=np.array([z['role']=='control' for z in items])
        checks=~controls
        summary={'pair_id':pid,'chip_id':cid,'controls':int(controls.sum()),
                 'checks':int(checks.sum()),'accepted':False,
                 'status':'INDEPENDENT_GCP_NOT_ENOUGH','pixel_error_p90':None}
        if controls.sum()>=min_control and checks.sum()>=min_check:
            try:
                T=fit_affine(a[controls],b[controls])
                projected=np.column_stack((a[checks],np.ones(checks.sum())))@T
                err=np.linalg.norm(projected-b[checks],axis=1)
                p90=float(np.quantile(err,.9))
                # Spatial reach: control-point bbox must span at least half the image in both axes
                reach=np.ptp(a[controls],axis=0)/size
                summary.update({'accepted':bool(p90<=max_p90 and np.all(reach>=.5)),
                   'status':'CHECKED_AFFINE_FIT' if p90<=max_p90 and np.all(reach>=.5)
                    else 'FAIL_GCP_RESIDUAL_OR_SPATIAL_COVERAGE',
                   'pixel_error_p50':float(np.median(err)),'pixel_error_p90':p90,
                   'max_error_px':float(err.max()), 'control_span_fraction':reach.tolist(),
                   'reference_to_moving_affine_2x3':T.T.tolist()})
            except ValueError as e:summary['status']='FAIL_GCP_GEOMETRY:'+str(e)
        result.append(summary)
    return {'NOTE':'Independent GCPs only. Model residual is a review signal, not independent pixel-level GT itself. Affine unsuitable for significant perspective or layover.',
            'chips_reviewed':len(result),'chips_passed_affine_check':sum(x['accepted'] for x in result),
            'results':result}

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--gcp-csv',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--chip-size',type=int,default=512)
    a.add_argument('--max-check-p90-px',type=float,default=3.)
    args=a.parse_args()
    result=validate_gcps(args.gcp_csv,args.chip_size,max_p90=args.max_check_p90_px)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'reviewed':result['chips_reviewed'],'passed':result['chips_passed_affine_check']}))

if __name__=='__main__':main()
