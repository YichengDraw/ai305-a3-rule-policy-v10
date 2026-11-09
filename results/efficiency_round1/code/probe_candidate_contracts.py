#!/usr/bin/env python3
"""Original-function synthetic probes; no policy rollout, physics or renderer."""
import argparse,ast,json,re,traceback
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from dataclasses import dataclass

def load_fit(path,repair=False):
    text=path.read_text()
    if repair:text=re.sub(r'(?<![A-Za-z0-9_])distance\[name\]\.shape','kind_distance[name].shape',text)
    tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VisionEstimator')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_fit_boxes')
    helper=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_rot_z','_project','_sample_map')]
    scope={'np':np,'cv2':cv2,'least_squares':least_squares}
    exec(compile(ast.fix_missing_locations(ast.Module(body=helper+[fn],type_ignores=[])),str(path),'exec'),scope)
    return scope['_fit_boxes']

def state():
    scene=NS(source_bin_center_m=[0.,0.],source_wall_base_z_m=0.,source_bin_half_size_m=[.08,.1],target_bin_world_position_m=[0.,0.,0.],target_bin_half_size_m=[.08,.1],bin_wall_thickness_m=.005,source_bin_wall_height_m=.05,target_bin_wall_height_m=.05)
    s=NS(scene=scene,step=1,v6_target_anchor=None,source_disturbed=True,v6_source_anchor={'center':np.zeros(3),'yaw':0.,'confidence':.8},boxes=None)
    s._box_points=lambda kind:np.array([[x,y,.05] for x in (-.08,.08) for y in (-.1,.1)])
    s._v7_target_instance=lambda *a:{'distance':np.zeros((256,256)),'sample':np.array([[100.,100.],[156.,100.],[156.,156.],[100.,156.]]),'component':1,'initial_iou':1.}
    s._v6_target_in_wrist=lambda *a:True;s._v7_poly_distance=lambda points,poly:np.zeros(len(points))
    s._v9_fit_target=lambda *a:{'center':np.zeros(3),'yaw':0.,'confidence':.8}
    return s

def fit_probe(parent,candidate):
    obs={f'observation.images.{n}':np.full((256,256,3),150,np.uint8) for n in ('front','left_wrist')}
    cams={n:{'pos':np.array([0.,0.,1.]),'rotation':np.eye(3),'fovy':70.} for n in ('front','left_wrist')};r={}
    for name,p,fix in [('parent',parent,False),('candidate',candidate,False),('memory_only_shape_fix',candidate,True)]:
        try:
            z=load_fit(p,fix)(state(),obs,cams)
            r[name]={'returned':True,'source_center':z['source_box']['center'].tolist(),'target_center':z['target_box']['center'].tolist()}
        except Exception as e:
            loc=traceback.extract_tb(e.__traceback__)[-1]
            r[name]={'returned':False,'exception':type(e).__name__,'message':str(e),'function':loc.name,'line':loc.lineno}
    return r

def alias_probe(path):
    tree=ast.parse(path.read_text());keep=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_unique_axes','_poly'):keep.append(n)
        if isinstance(n,ast.ClassDef) and n.name=='Poly':keep.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_POLY_CACHE' for x in n.targets):keep.append(n)
    env={'np':np,'ConvexHull':ConvexHull,'dataclass':dataclass}
    exec(compile(ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[])),str(path),'exec'),env)
    v=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    a=env['_poly'](v);b=env['_poly'](v.copy());old=float(b.vertices[0,0]);a.vertices[0,0]+=10
    return {'same_returned_object':a is b,'array_writeable':bool(a.vertices.flags.writeable),'mutation_propagated':bool(b.vertices[0,0]!=old)}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path);a=ap.parse_args();ns='v10_leaves_17e0363838a9/'
    r={'scope':'synthetic branch inputs; target association stub; original AST functions; no policy rollout','initial_state':{'source_disturbed':True,'v6_source_anchor':'present','v6_target_anchor':None,'target_component':'present'},'fit_probes':{},'cache_alias_probes':{},'simulation_steps':0,'render_calls':0}
    for rel in ['two10_source1/perception.py','two5_source1/legacy/perception.py','two5_source1/target_vision.py']:r['fit_probes'][rel]=fit_probe(a.parent/(ns+rel),a.candidate/(ns+rel))
    for rel in ['two10_source1/path_feasibility.py','two5_source1/legacy/path_feasibility.py','two5_source1/path_feasibility.py']:r['cache_alias_probes'][rel]=alias_probe(a.candidate/(ns+rel))
    r['alias_scope_note']='Mutation was injected in isolated helper test; no claim of observed production mutation.'
    s=json.dumps(r,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.write_text(s)
    print(s)
    assert all(x['parent']['returned'] and not x['candidate']['returned'] and x['memory_only_shape_fix']['returned'] for x in r['fit_probes'].values())
if __name__=='__main__':main()
