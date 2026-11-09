#!/usr/bin/env python3
"""Original-function synthetic probes; no policy rollout, physics or renderer."""
import argparse,ast,json,re,traceback
from pathlib import Path
from types import SimpleNamespace as NS
from collections import OrderedDict
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


def load_poly(path):
    tree=ast.parse(path.read_text());nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_unique_axes','_poly'):nodes.append(n)
        elif isinstance(n,ast.ClassDef) and n.name=='Poly':nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_POLY_CACHE' for t in n.targets):nodes.append(n)
    env={'np':np,'ConvexHull':ConvexHull,'dataclass':dataclass,'OrderedDict':OrderedDict}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
    return env

def cache_probe(path):
    env=load_poly(path);fn=env['_poly'];cache=env['_POLY_CACHE']
    v=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    a=fn(v);b=fn(v.copy());blocked=False
    try:a.vertices[0,0]+=1
    except ValueError:blocked=True
    fn(v+.01);k=(v.shape,v.tobytes());fn(v)
    hit_moved=list(cache)[-1]==k
    cache.clear()
    for i in range(8193):cache[('probe_sentinel',i)]=a
    fn(v+.02);size=len(cache)
    # Field rebinding is an interface probe, not an observed production action.
    env=load_poly(path);a=env['_poly'](v);replacement=a.vertices.copy();replacement[0,0]+=1
    try:
        a.vertices=replacement
        rebind=True
    except Exception:rebind=False
    return dict(array_element_mutation_blocked=blocked,returns_same_object=a is env['_poly'](v),
                cache_hit_moves_to_most_recent=hit_moved,size_after_injected_8193_entries_and_one_miss=size,
                field_rebinding_allowed=rebind,scope='isolated synthetic cache contract; no claim of production mutation')

def polygon_probe(path):
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_distances','_prepare_polygon','_distances_prepared')]
    env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    rng=np.random.default_rng(271828);cases=0
    for vertices in [1,2,3,4,8,17]:
        for n in [0,1,32,127]:
            for _ in range(5):
                poly=rng.normal(size=(vertices,2));points=rng.normal(size=(n,2))
                x=env['_distances'](points,poly);y=env['_distances_prepared'](points,env['_prepare_polygon'](poly))
                assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                cases+=1
    return {'synthetic_inputs_tested':cases,'returned_array_bytes_equal':True,'scope':'helper arithmetic only; not whole optimizer or episode'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path)
    a=ap.parse_args();n='v10_leaves_17e0363838a9/'
    result={'scope':'AST-isolated original functions; target association is a stub; no policy rollout, no renderer','fit':{},'cache':{},'polygon':{},'simulation_steps':0,'render_calls':0}
    for rel in ['two10_source1/perception.py','two5_source1/legacy/perception.py','two5_source1/target_vision.py']:
        result['fit'][rel]=fit_probe(a.parent/(n+rel),a.candidate/(n+rel))
        assert result['fit'][rel]['parent']['returned'] and result['fit'][rel]['candidate']['returned']
    for rel in ['two10_source1/path_feasibility.py','two5_source1/legacy/path_feasibility.py','two5_source1/path_feasibility.py']:
        result['cache'][rel]=cache_probe(a.candidate/(n+rel))
    for rel in ['two10_source1/placement_audit.py','two5_source1/legacy/placement_audit.py']:result['polygon'][rel]=polygon_probe(a.candidate/(n+rel))
    s=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.write_text(s)
    print(s,end='')
if __name__=='__main__':main()
