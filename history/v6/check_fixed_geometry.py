"""Check analytic jaw conversion against the unchanged original CAD model."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.paths import resource_root
from a3_dual_arm_sim.sim.model import build_model

config=load_config(resource_root()/'configs/cookie_batch.yaml')
model=build_model(config,scene='cookie_transfer').model
data=mujoco.MjData(model)
geoms=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'L_finger_{f}_geom') for f in ('inner','outer')]
joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f'L_finger_{f}_joint') for f in ('inner','outer')]
site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'L_eef')
if site<0:
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_SITE,i) for i in range(model.nsite)]
    raise ValueError(f'EEF site name missing: {names}')
rows=[]
for opening in (0.,.1,.5,1.):
    for joint in joints:
        data.qpos[model.jnt_qposadr[joint]]=(1-opening)*.0425
    mujoco.mj_forward(model,data)
    gap=float(np.linalg.norm(data.geom_xpos[geoms[0]]-data.geom_xpos[geoms[1]])-
        model.geom_size[geoms[0],1]-model.geom_size[geoms[1],1])
    offset=data.site_xmat[site].reshape(3,3).T@(data.geom_xpos[geoms].mean(0)-data.site_xpos[site])
    rows.append(dict(opening=opening,cad_gap_m=gap,analytic_gap_m=.0053616+.085*opening,
        pad_midpoint_local_to_eef_m=offset.tolist()))
    if abs(gap-(.0053616+.085*opening))>1e-8:
        raise ValueError('Analytic jaw conversion does not match CAD')
payload=dict(check='original_CAD_geometry',samples=rows,
    max_gap_error_m=max(abs(r['cad_gap_m']-r['analytic_gap_m']) for r in rows))
text=json.dumps(payload,indent=2)+'\n'
Path(sys.argv[1]).write_text(text)
print(text)
