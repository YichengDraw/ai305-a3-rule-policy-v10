"""RGB/CAD geometry estimator. Runtime inputs are public robot observations only.

A separate nominal robot model supplies camera mounts and visible robot geometry.
No episode environment, object state, contact graph, seed-derived object state, or
segmentation renderer is used. Initial cookie poses are layout hypotheses anchored
to the visually fitted source box; their confidence explicitly reflects this.
"""
from __future__ import annotations
import math
from typing import Any
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS
from a3_dual_arm_sim.sim.model import build_model


def _rot_z(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)


def _sample_rectangle(hx,hy,z,n=24):
    corners=np.array([[-hx,-hy,z],[hx,-hy,z],[hx,hy,z],[-hx,hy,z]],float)
    return np.concatenate([corners[i]+np.linspace(0,1,n,endpoint=False)[:,None]*(corners[(i+1)%4]-corners[i]) for i in range(4)])


def _project(points,camera,shape):
    h,w=shape[:2];f=h/2/np.tan(np.deg2rad(camera['fovy'])/2)
    p=(np.asarray(points)-camera['pos'])@camera['rotation']
    depth=np.maximum(-p[:,2],1e-5)
    return np.column_stack([(w-1)/2+f*p[:,0]/depth,(h-1)/2-f*p[:,1]/depth]),-p[:,2]


def _sample_map(array,points,default=20.):
    h,w=array.shape;xy=np.asarray(points);valid=(xy[:,0]>=0)&(xy[:,0]<w-1)&(xy[:,1]>=0)&(xy[:,1]<h-1)
    result=np.full(len(xy),default,float)
    if np.any(valid):
        q=xy[valid];x=np.floor(q[:,0]).astype(int);y=np.floor(q[:,1]).astype(int);a=q[:,0]-x;b=q[:,1]-y
        result[valid]=(1-a)*(1-b)*array[y,x]+a*(1-b)*array[y,x+1]+(1-a)*b*array[y+1,x]+a*b*array[y+1,x+1]
    return result,valid


class VisionEstimator:
    def __init__(self,config):
        self.config=config;self.scene=config.cookie_transfer
        self.layout=np.asarray(self.scene.cookie_source_positions_m,float)
        self.columns=np.unique(self.layout[:,0]);self.cookie_half=np.asarray(self.scene.cookie_half_size_m,float)
        bundle=build_model(config,scene='cookie_transfer')
        self.robot_model=bundle.model;self.robot_data=mujoco.MjData(self.robot_model)
        self.joint_addresses=[int(self.robot_model.jnt_qposadr[mujoco.mj_name2id(self.robot_model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in (*LEFT_JOINTS,*RIGHT_JOINTS)]
        self.finger_addresses=[]
        for side in ('L','R'):
            self.finger_addresses.append([int(self.robot_model.jnt_qposadr[mujoco.mj_name2id(self.robot_model,mujoco.mjtObj.mjOBJ_JOINT,f'{side}_finger_{finger}_joint')]) for finger in ('inner','outer')])
        self.camera_ids={n:mujoco.mj_name2id(self.robot_model,mujoco.mjtObj.mjOBJ_CAMERA,n) for n in ('front','left_wrist','right_wrist')}
        self.dark_robot=[]
        for i in range(self.robot_model.ngeom):
            body=mujoco.mj_id2name(self.robot_model,mujoco.mjtObj.mjOBJ_BODY,int(self.robot_model.geom_bodyid[i])) or ''
            if self.robot_model.geom_group[i]!=2 or not body.startswith(('L_','R_')) or np.max(self.robot_model.geom_rgba[i,:3])>.2:continue
            typ=int(self.robot_model.geom_type[i])
            if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
                mid=int(self.robot_model.geom_dataid[i]);start=int(self.robot_model.mesh_vertadr[mid]);num=int(self.robot_model.mesh_vertnum[mid]);verts=self.robot_model.mesh_vert[start:start+num].copy()
                if len(verts)>250:verts=verts[np.linspace(0,len(verts)-1,250).astype(int)]
            elif typ==int(mujoco.mjtGeom.mjGEOM_BOX):
                verts=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*self.robot_model.geom_size[i]
            else:continue
            self.dark_robot.append((i,verts))
        self.reset(None)

    def reset(self,context=None):
        self.step=0;self.last_key=None;self.last_count_key=None;self.last_held_count=(None,0.);self.camera_adjustments={};self.boxes=None;self.last_fit_step=-999
        self.cookie_delta=None;self.cookie_fit_confidence=0.;self.cookie_fit_error=None;self.selected_group=None;self.selected_group_key=None;self.held_anchor=None
        self.initial_source_image=None;self.context_task=getattr(context,'task','')

    def _robot_cameras(self,observation):
        state=np.asarray(observation['observation.state'],float)
        self.robot_data.qpos[self.joint_addresses]=np.r_[state[:7],state[8:15]]
        for side,opening in enumerate((state[7],state[15])):
            self.robot_data.qpos[self.finger_addresses[side]]=(1-np.clip(opening,0,1))*.0425
        mujoco.mj_forward(self.robot_model,self.robot_data)
        return {name:{'pos':self.robot_data.cam_xpos[cid].copy(),'rotation':self.robot_data.cam_xmat[cid].reshape(3,3).copy(),'fovy':float(self.robot_model.cam_fovy[cid])} for name,cid in self.camera_ids.items()}

    @staticmethod
    def _adjust_camera(camera,params):
        return {'pos':camera['pos']+np.asarray(params[:3]),'rotation':Rotation.from_rotvec(params[3:6]).as_matrix()@camera['rotation'],'fovy':camera['fovy']+params[6]}

    def _calibrate_robot(self,observation,cameras):
        """Fit visible dark robot contours to public-state robot CAD once/reset."""
        results={}
        meshes=[(self.robot_data.geom_xpos[i]+verts@self.robot_data.geom_xmat[i].reshape(3,3).T) for i,verts in self.dark_robot]
        for name in ('front','left_wrist'):
            image=np.asarray(observation['observation.images.'+name]);h,w=image.shape[:2]
            camera=cameras[name];pred=np.zeros((h,w),np.uint8);world_by_pixel=[]
            all_world=[]
            for verts in meshes:
                pixels,depth=_project(verts,camera,image.shape)
                if np.any(depth<=0):continue
                if np.all((pixels[:,0]<-30)|(pixels[:,0]>w+30)|(pixels[:,1]<-30)|(pixels[:,1]>h+30)):continue
                hull=cv2.convexHull(pixels.astype(np.float32)).reshape(-1,2)
                cv2.fillConvexPoly(pred,np.rint(hull).astype(np.int32),255)
                all_world.append(verts)
            if not all_world:continue
            world=np.concatenate(all_world);pixels,depth=_project(world,camera,image.shape)
            # Select 3D vertices lying on the union's visible silhouette.
            edge=cv2.Canny(pred,50,100);dist=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
            d,valid=_sample_map(dist,pixels)
            keep=valid&(d<1.8)&(pixels[:,0]>4)&(pixels[:,0]<w-5)&(pixels[:,1]>4)&(pixels[:,1]<h-5)
            sample=world[keep]
            if len(sample)<20:continue
            if len(sample)>180:sample=sample[np.linspace(0,len(sample)-1,180).astype(int)]
            dark=(np.max(image,axis=2)<48).astype(np.uint8)*255
            roi=cv2.dilate(pred,np.ones((17,17),np.uint8))
            dark=cv2.bitwise_and(dark,roi)
            real_edge=cv2.Canny(dark,50,100)
            dt=cv2.distanceTransform(255-real_edge,cv2.DIST_L2,3)
            def residual(p):
                adjusted=self._adjust_camera(camera,p);uv,z=_project(sample,adjusted,image.shape);err,v=_sample_map(dt,uv)
                # Public geometry anchors camera translation. Small mount prior resolves
                # weak directions from the small, partly clipped silhouette.
                return np.r_[np.minimum(err,12),p[:3]/.009,p[3:6]/.045,p[6:]/2.5]
            bound=np.array([.014,.014,.014,.065,.065,.065,3.])
            fit=least_squares(residual,np.zeros(7),bounds=(-bound,bound),diff_step=1e-3,max_nfev=35,loss='soft_l1',f_scale=1.)
            err=float(np.median(residual(fit.x)[:len(sample)]))
            # Silhouette is a coarse calibration cue, with residual error disclosed.
            if err<3.0:self.camera_adjustments[name]=fit.x
            results[name]={'residual_px':err,'samples':len(sample),'adjustment':fit.x.tolist()}
        return results

    def _box_points(self,kind):
        t=self.scene.bin_wall_thickness_m
        if kind=='source':
            hx,hy=self.scene.source_bin_half_size_m;z=self.scene.source_wall_base_z_m+self.scene.source_bin_wall_height_m
        else:
            hx,hy=self.scene.target_bin_half_size_m;z=self.scene.target_bin_wall_height_m
        return np.vstack([_sample_rectangle(hx+t,hy+t,z),_sample_rectangle(hx-t,hy-t,z)])

    def _fit_boxes(self,observation,cameras):
        masks={};distance={}
        for name in ('front','left_wrist'):
            image=np.asarray(observation['observation.images.'+name]);mx=image.max(2);mn=image.min(2)
            gray=((mx.astype(int)-mn.astype(int)<38)&(mx>65)&(mx<250)&(image[:,:,2].astype(int)+10>=image[:,:,0].astype(int))).astype(np.uint8)*255
            masks[name]=gray;edge=np.zeros_like(gray)
            contours,_=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edge,[c for c in contours if cv2.contourArea(c)>15],-1,255,1)
            distance[name]=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
        boxes={}
        for kind in ('source','target'):
            if kind=='source':initial=np.r_[self.scene.source_bin_center_m,0.];center_z=0.;bounds=np.array([.035,.035,1e-7])
            else:initial=np.r_[self.scene.target_bin_world_position_m[:2],0.];center_z=float(self.scene.source_wall_base_z_m)-.00025;bounds=np.array([.04,.04,.13])
            if self.boxes is not None:initial=np.r_[self.boxes[kind+'_box']['center'][:2],self.boxes[kind+'_box']['yaw']]
            points=self._box_points(kind)
            # Only fit CAD edges expected to be inside each image. Off-screen
            # fragments do not pull the box toward an image boundary.
            cameras_used=[]
            t=self.scene.bin_wall_thickness_m
            half=np.asarray(self.scene.source_bin_half_size_m if kind=='source' else self.scene.target_bin_half_size_m)+t
            zb=self.scene.source_wall_base_z_m if kind=='source' else 0.
            zt=zb+(self.scene.source_bin_wall_height_m if kind=='source' else self.scene.target_bin_wall_height_m)
            corners=np.array([[x,y,z] for z in (zb,zt) for x in (-half[0],half[0]) for y in (-half[1],half[1])])
            for name in ('front','left_wrist'):
                nominal=corners@_rot_z(initial[2]).T+np.r_[initial[:2],center_z]
                uv,z=_project(nominal,cameras[name],distance[name].shape)
                if np.any(z<=0):continue
                hull=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel()
                subset=np.concatenate([corners[hull[j]]+np.linspace(0,1,32,endpoint=False)[:,None]*(corners[hull[(j+1)%len(hull)]]-corners[hull[j]]) for j in range(len(hull))])
                puv,pz=_project(subset@_rot_z(initial[2]).T+np.r_[initial[:2],center_z],cameras[name],distance[name].shape)
                valid=(pz>0)&(puv[:,0]>2)&(puv[:,0]<distance[name].shape[1]-3)&(puv[:,1]>2)&(puv[:,1]<distance[name].shape[0]-3)
                if np.sum(valid)>25:cameras_used.append((name,subset[valid]))
            if not cameras_used:
                boxes[kind+'_box']={'center':np.r_[initial[:2],center_z],'yaw':float(initial[2]),'confidence':0.,'position_error_m':.025,'residual_px':None};continue
            def residual(p):
                pts=points@_rot_z(p[2]).T+np.r_[p[:2],center_z];res=[]
                for name,subset in cameras_used:
                    uv,z=_project(subset@_rot_z(p[2]).T+np.r_[p[:2],center_z],cameras[name],distance[name].shape)
                    err,v=_sample_map(distance[name],uv)
                    res.extend(np.minimum(err,15))
                return np.r_[res,(p[:2]-initial[:2])/.035,(p[2]-initial[2])/.12]
            lo=initial-bounds;hi=initial+bounds
            fit=least_squares(residual,initial,bounds=(lo,hi),diff_step=1e-3,max_nfev=35,loss='soft_l1',f_scale=1.)
            err=float(np.median(residual(fit.x)[:-3]));confidence=float(np.clip(1-err/4,0,1))
            boxes[kind+'_box']={'center':np.r_[fit.x[:2],center_z],'yaw':float(fit.x[2]),'confidence':confidence,'position_error_m':max(.0015,.0015+err*.0015),'residual_px':err}
        return boxes

    def _refine_cookie_layout(self,observation,cameras,source,request):
        """Align repeated visible cookie top edges after the box's coarse fit."""
        start=int(request.get('source_start_index',0));column=int(request.get('source_column',1))-1
        keep=np.ones(len(self.layout),bool)
        inds=np.flatnonzero(np.isclose(self.layout[:,0],self.columns[np.clip(column,0,len(self.columns)-1)]))
        inds=inds[np.argsort(self.layout[inds,1])]
        keep[inds[:start]]=False
        indices=np.flatnonzero(keep)
        # Two long bevel edges are more visible than the narrow top face.
        hx,hy,hz=self.cookie_half;bevel=self.scene.cookie_edge_bevel_m
        zc=float(self.scene.cookie_model_z_m)
        local=np.vstack([np.column_stack([np.linspace(-hx,hx,8),np.full(8,y),np.full(8,z)]) for y,z in ((-hy+bevel,hz),(hy-bevel,hz),(-hy,hz-bevel),(hy,hz-bevel))])
        nominal=np.column_stack([self.layout[indices],np.full(len(indices),zc)])
        points=(nominal[:,None,:]+local[None,:,:]).reshape(-1,3)
        initial=np.asarray(source['center'][:2])-np.asarray(self.scene.source_bin_center_m)
        maps=[]
        for name in ('front','left_wrist'):
            im=np.asarray(observation['observation.images.'+name]);gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            edge=cv2.Canny(gray,35,85);dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
            uv,z=_project(points+np.r_[initial,0.],cameras[name],im.shape)
            valid=(z>0)&(uv[:,0]>2)&(uv[:,0]<im.shape[1]-3)&(uv[:,1]>2)&(uv[:,1]<im.shape[0]-3)
            if valid.sum()>100:maps.append((name,dt,points[valid]))
        if not maps:return initial,source['confidence'],None
        def residual(delta):
            errs=[]
            for name,dt,pts in maps:
                uv,z=_project(pts+np.r_[delta,0.],cameras[name],dt.shape);r,v=_sample_map(dt,uv)
                errs.extend(np.minimum(r,7.))
            return np.r_[errs,(delta-initial)/.004]
        fit=least_squares(residual,initial,bounds=(initial-.0045,initial+.0045),diff_step=1e-3,max_nfev=25,loss='soft_l1',f_scale=.8)
        err=float(np.median(residual(fit.x)[:-2]));confidence=float(np.clip(1-err/2.5,0,.8))
        return fit.x,confidence,err

    def _fit_selected_group(self,observation,cameras,centers,indices,source_error):
        """Fit a common physical lean from RGB cookie silhouettes and top edges."""
        if len(indices)==0:return None
        center=centers[indices].mean(0);hx,hy,hz=self.cookie_half;bevel=self.scene.cookie_edge_bevel_m
        yz=np.array([[-hy,-hz+min(bevel,.001)],[ -hy+min(bevel,.001),-hz],[hy-min(bevel,.001),-hz],[hy,-hz+min(bevel,.001)],[hy,hz-bevel],[hy-bevel,hz],[-hy+bevel,hz],[-hy,hz-bevel]])
        vertices=np.array([[x,y,z] for x in (-hx,hx) for y,z in yz])
        relative=centers[indices]-center;maps=[]
        for name in ('front','left_wrist'):
            im=np.asarray(observation['observation.images.'+name]);edge=cv2.Canny(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),35,85);dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
            nominal=(centers[indices,None,:]+vertices[None,:,:]).reshape(-1,3)
            uv,z=_project(nominal,cameras[name],im.shape)
            valid=(z>0)&(uv[:,0]>2)&(uv[:,0]<im.shape[1]-3)&(uv[:,1]>2)&(uv[:,1]<im.shape[0]-3)
            if valid.sum()>10:maps.append((name,dt))
        if not maps:return None
        def geometry(p):
            angle=p[2];c,s=np.cos(angle),np.sin(angle);rot=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            centers_fit=centers[indices].copy();centers_fit[:,:2]+=p[:2]
            # Bottom support constrains height; the common lean displaces the top.
            centers_fit[:,1]-=np.sin(angle)*hz
            centers_fit[:,2]=float(self.scene.cookie_model_z_m)-hz+np.cos(angle)*hz
            return centers_fit,rot
        def residual(p):
            fitted,rot=geometry(p);points=[]
            # Use top long edges and the two exposed group's end faces.
            top=np.vstack([np.column_stack([np.linspace(-hx,hx,10),np.full(10,y),np.full(10,z)]) for y,z in ((-hy+bevel,hz),(hy-bevel,hz),(-hy,hz-bevel),(hy,hz-bevel))])
            pts=(fitted[:,None,:]+(top@rot.T)[None,:,:]).reshape(-1,3)
            err=[]
            for name,dt in maps:
                uv,z=_project(pts,cameras[name],dt.shape);r,valid=_sample_map(dt,uv)
                err.extend(np.minimum(r,7.))
            return np.r_[err,p[:2]/max(source_error,.0015),p[2]/.6]
        best=None
        for angle in (0.,-.45,.45):
            fit=least_squares(residual,[0.,0.,angle],bounds=([-.005,-.006,-1.05],[.005,.006,1.05]),max_nfev=18,diff_step=1e-3,loss='soft_l1',f_scale=.8)
            score=float(np.mean(np.minimum(residual(fit.x)[:-3],4.)**2))
            if best is None or score<best[0]:best=(score,fit.x)
        base=least_squares(lambda xy:residual(np.r_[xy,0.]),[0.,0.],bounds=([-.003,-.003],[.003,.003]),max_nfev=18,diff_step=1e-3,loss='soft_l1',f_scale=.8)
        base_p=np.r_[base.x,0.];base_score=float(np.mean(np.minimum(residual(base_p)[:-3],4.)**2))
        score,p=best
        # A lean is accepted only when it explains substantially more image
        # evidence than an upright group; translation/tilt otherwise alias.
        tilt_observed=score < base_score*.70 and abs(p[2])>np.deg2rad(8.)
        if not tilt_observed:score,p=base_score,base_p
        fitted,rotation=geometry(p);err=float(np.sqrt(score));confidence=float(np.clip(1-err/2.5,0,.8))
        return {'centers':fitted,'center':fitted.mean(0),'rotation':rotation,'tilt_deg':float(abs(np.rad2deg(p[2]))),'confidence':confidence,'edge_residual_px':err,'tilt_verified':bool(tilt_observed and confidence>=.6),'position_error_m':max(source_error,.0015+err*.001)}

    def _count_visible_held(self,observation,cameras,group,request):
        """Count narrow, repeated RGB top-edge bands in the moving group ROI.

        The estimate is available only during trial/full lift and uses a tracked
        RGB group location plus public TCP motion; expected task count is never
        returned as an observation.
        """
        stage=str(request.get('stage','')).upper()
        if stage not in ('TRIAL_LIFT','LIFT','TRANSPORT','PLACE'):return None,0.
        pose=np.asarray(observation['observation.eef_pose'],float)
        if self.held_anchor is None:self.held_anchor=(pose[:3].copy(),np.asarray(group['center']).copy())
        center=self.held_anchor[1]+pose[:3]-self.held_anchor[0]
        name='left_wrist';im=np.asarray(observation['observation.images.'+name]);camera=cameras[name]
        # Rectify the cookie top plane. The broad search includes extra/missing
        # cookies; count evidence is the number of separated edge/colour bands.
        half_width=.050;hx=self.cookie_half[0]
        xs=np.linspace(-hx*.65,hx*.65,24);ys=np.linspace(-half_width,half_width,240)
        points=np.array([[center[0]+x,center[1]+y,center[2]+self.cookie_half[2]] for y in ys for x in xs])
        uv,z=_project(points,camera,im.shape)
        valid=(z>0)&(uv[:,0]>1)&(uv[:,0]<im.shape[1]-2)&(uv[:,1]>1)&(uv[:,1]<im.shape[0]-2)
        if valid.mean()<.85:return None,0.
        sample=cv2.remap(im,uv[:,0].astype(np.float32).reshape(240,24),uv[:,1].astype(np.float32).reshape(240,24),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        hsv=cv2.cvtColor(sample,cv2.COLOR_RGB2HSV);sat=hsv[:,:,1].mean(1);value=hsv[:,:,2].mean(1)
        # A cookie has a saturated body; require colour support before any count.
        support=(sat>65)&(value>35)
        if support.sum()<10:return None,0.
        # Gaps create dark valleys between top bands. Gaussian smoothing handles
        # subpixel aliases; accepted peaks need plausible physical pitch.
        signal=cv2.GaussianBlur(value.astype(np.float32)[:,None],(1,5),0).ravel()
        from scipy.signal import find_peaks
        peaks,properties=find_peaks(signal,prominence=max(5.,float(np.std(signal))*.3),distance=9)
        peaks=np.asarray([k for k in peaks if support[k]])
        if not len(peaks):return None,0.
        # Restrict to the contiguous colour-supported group nearest predicted centre.
        near=peaks[np.abs(ys[peaks])<.045]
        if len(near)<1 or len(near)>12:return None,0.
        gaps=np.diff(ys[near]);regular= len(gaps)==0 or np.all((gaps>.0045)&(gaps<.0115))
        confidence=.65 if regular and len(near)>=2 and float(np.mean(sat[near]))>90 else .35
        return int(len(near)),confidence

    def estimate(self,observation,request=None):
        request=request or {};self.step+=1
        cameras=self._robot_cameras(observation)
        if self.step==1:self.calibration={}
        cameras={n:self._adjust_camera(c,self.camera_adjustments.get(n,np.zeros(7))) for n,c in cameras.items()}
        key=(request.get('batch_index'),request.get('stage'))
        stage=str(request.get('stage',''))
        refit=self.boxes is None or (key!=self.last_key and any(s in stage.upper() for s in ('SELECT','APPROACH','VERIFY','OBSERVE','PLAN'))) or self.step-self.last_fit_step>=30
        if refit:
            self.boxes=self._fit_boxes(observation,cameras);self.last_fit_step=self.step
            self.cookie_delta,self.cookie_fit_confidence,self.cookie_fit_error=self._refine_cookie_layout(observation,cameras,self.boxes['source_box'],request)
        self.last_key=key
        source=self.boxes['source_box'];delta=self.cookie_delta if self.cookie_delta is not None else np.asarray(source['center'][:2])-np.asarray(self.scene.source_bin_center_m)
        centers=np.column_stack([self.layout+delta,np.full(len(self.layout),self.scene.cookie_model_z_m)])
        rotations=np.repeat(np.eye(3)[None],len(centers),axis=0)
        confidence=np.full(len(centers),min(.65,self.cookie_fit_confidence))
        column=int(request.get('source_column',1));count=int(request.get('count',5));start=int(request.get('source_start_index',0));column=np.clip(column,1,len(self.columns))
        indices=np.flatnonzero(np.isclose(self.layout[:,0],self.columns[column-1]))
        indices=indices[np.argsort(self.layout[indices,1])][start:start+count]
        group_center=centers[indices].mean(axis=0) if len(indices) else np.zeros(3)
        group={'indices':indices.tolist(),'center':group_center,'rotation':np.eye(3),'tilt_deg':0.,'width_m':float((max(count,1)-1)*.0088333333333+2*self.cookie_half[1]),'confidence':float(confidence[indices].min()) if len(indices) else 0.,'position_error_m':source['position_error_m'],'geometry_source':'visual_cookie_edges_plus_fixed_layout','edge_residual_px':self.cookie_fit_error,'tilt_verified':False,'count_verified':False}
        group_key=(request.get('batch_index'),column,start,count)
        stage=str(request.get('stage','')).upper()
        if self.selected_group_key!=group_key:
            self.selected_group=None;self.held_anchor=None;self.selected_group_key=group_key
        if self.selected_group is None or (refit and stage in ('SELECT','APPROACH','DESCEND','CLOSE')):
            fitted=self._fit_selected_group(observation,cameras,centers,indices,source['position_error_m'])
            if fitted is not None:self.selected_group=fitted
        if self.selected_group is not None:
            fitted=self.selected_group;group.update({k:v for k,v in fitted.items() if k!='centers'})
            centers[indices]=fitted['centers'];rotations[indices]=fitted['rotation'];confidence[indices]=fitted['confidence']
        if stage in ('SELECT','APPROACH','DESCEND','CLOSE'):
            self.held_anchor=(np.asarray(observation['observation.eef_pose'],float)[:3].copy(),np.asarray(group['center']).copy())
        held_count,held_confidence=self._count_visible_held(observation,cameras,group,request) if self.step%5==0 or key!=self.last_count_key else self.last_held_count
        self.last_count_key=key;self.last_held_count=(held_count,held_confidence)
        return {**self.boxes,'cookies':{'centers':centers,'rotations':rotations,'visible':confidence>.6,'confidence':confidence},'group':group,'held_count':held_count,'held_count_confidence':held_confidence,'target_count':None,'target_count_confidence':0.,'camera_calibration':getattr(self,'calibration',{}),'source_tilt_deg':group['tilt_deg'],'position_error_m':source['position_error_m']}
