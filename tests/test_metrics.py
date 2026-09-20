import numpy as np
from causal_se2_occ.metrics.iou import raw_counts
from causal_se2_occ.metrics.bootstrap import paired_scene_bootstrap
from causal_se2_occ.protocol import REPORT_HORIZONS_S
def test_hand_counts():
 gt=np.array([4,4,17,7,17,17]);pred=np.array([4,17,4,7,17,17]);mov=np.array([1,1,0,1,0,0],dtype=bool);r=raw_counts(pred,gt,mov);assert (r['occ_inter'],r['occ_union'])==(2,4);assert (r['sem_inter'][4],r['sem_union'][4])==(1,3);assert (r['mov_inter'][4],r['mov_union'][4])==(1,2)
def test_scene_bootstrap_raw():
 rows={}
 for s,mults in [('a',(1,1,1)),('b',(2,3,4))]:
  for h,k in zip(REPORT_HORIZONS_S,mults):
   base={'occ_inter':10*k,'occ_union':20*k,'sem_inter':{c:(10*k if c==4 else 0) for c in range(17)},'sem_union':{c:(20*k if c==4 else 0) for c in range(17)},'mov_inter':{c:(10*k if c==4 else 0) for c in (2,3,4,5,6,7,9,10)},'mov_union':{c:(20*k if c==4 else 0) for c in (2,3,4,5,6,7,9,10)}};cand={k2:(dict(v) if isinstance(v,dict) else v) for k2,v in base.items()};cand['occ_inter']=12*k;cand['sem_inter'][4]=12*k;cand['mov_inter'][4]=12*k;rows[(s,'ref',float(h))]=base;rows[(s,'cand',float(h))]=cand
 b=paired_scene_bootstrap(rows,'ref','cand',['a','b'],samples=20,seed=1);assert b['aggregation']=='paired_scene_resample_then_sum_all_selected_scene_window_raw_intersection_union';assert abs(b['metrics']['IoU']['point_delta_pp']-10.)<1e-12
