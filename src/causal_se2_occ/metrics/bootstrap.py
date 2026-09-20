from __future__ import annotations
import numpy as np
from .iou import add_counts,metrics_from_horizon_counts
from ..protocol import REPORT_HORIZONS_S
def aggregate_scene_rows(scene_rows,variant,scenes):
    by={float(h):None for h in REPORT_HORIZONS_S}
    for s in scenes:
        for h in REPORT_HORIZONS_S:
            r=scene_rows[(str(s),str(variant),float(h))];by[float(h)]=add_counts(by[float(h)],r)
    return by
def paired_scene_bootstrap(scene_rows,ref,cand,scenes,*,samples=2000,seed=20260917):
    names=list(scenes);rng=np.random.default_rng(seed); keys=('IoU','mIoU','MacroMoving','MicroMoving');vals={k:[] for k in keys};point_r=metrics_from_horizon_counts(aggregate_scene_rows(scene_rows,ref,names));point_c=metrics_from_horizon_counts(aggregate_scene_rows(scene_rows,cand,names))
    for _ in range(samples):
        draw=rng.choice(names,size=len(names),replace=True); rr=metrics_from_horizon_counts(aggregate_scene_rows(scene_rows,ref,draw));cc=metrics_from_horizon_counts(aggregate_scene_rows(scene_rows,cand,draw))
        for k in keys:vals[k].append(cc[k]-rr[k])
    return {'samples':samples,'seed':seed,'num_scenes':len(names),'aggregation':'paired_scene_resample_then_sum_all_selected_scene_window_raw_intersection_union','metrics':{k:{'point_delta_pp':point_c[k]-point_r[k],'mean_pp':float(np.mean(vals[k])),'p2_5_pp':float(np.quantile(vals[k],.025)),'p97_5_pp':float(np.quantile(vals[k],.975))} for k in keys}}
