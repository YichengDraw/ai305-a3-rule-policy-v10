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
        self.placement_auditor=None
        self._v6_gripper_meshes=self._v6_prepare_gripper()
        self.reset(None)

    def reset(self,context=None):
        self.step=0;self.last_key=None;self.last_count_key=None;self.last_held_count=(None,0.);self.last_target_key=None;self.last_target_count=(None,0.);self.target_count_diagnostics={};self.source_disturbed=False;self.target_placement_valid=None;self.target_placement_confidence=0.;self.camera_adjustments={};self.boxes=None;self.last_fit_step=-999
        self.cookie_delta=None;self.cookie_fit_confidence=0.;self.cookie_fit_error=None;self.selected_group=None;self.selected_group_key=None;self.held_anchor=None
        self.initial_source_image=None;self.context_task=getattr(context,'task','')
        self.cookie_colors=None;self.wrist_mount=np.zeros(7);self.wrist_calibration_samples=[];self.wrist_last_opening=None;self.wrist_calibration={'state':'unobserved','uncertainty_m':.004,'reason':'not fitted'}
        self.v6_source_anchor=None;self.v6_target_anchor=None;self.calibration_generation=0;self.v6_request_stage='SELECT';self.v6_frames=[];self.v6_batch_key=None;self.v6_held_reference=None;self.v6_held_method=None;self.v6_held_generation=0;self.v6_held_geometry_history=[];self.v6_use_bevel_held=False;self.v6_wrist_target_complete=False;self.v6_held_points=None;self.v6_held_previous=None;self.v6_placement_tracks=None
        if self.placement_auditor is not None:self.placement_auditor.reset(context)
        self.v6_placement_cache={'state':'unknown','confidence':0.,'count':None,'count_confidence':0.,'count_scope':'batch','reason':'audit not observed','repairable':False}
        self.v6_local_align={'state':'unknown','valid':False,'confidence':0.,'reason':'not observed'}

    def _robot_cameras(self,observation):
        state=np.asarray(observation['observation.state'],float)
        self.robot_data.qpos[self.joint_addresses]=np.r_[state[:7],state[8:15]]
        for side,opening in enumerate((state[7],state[15])):
            self.robot_data.qpos[self.finger_addresses[side]]=(1-np.clip(opening,0,1))*.0425
        mujoco.mj_kinematics(self.robot_model,self.robot_data)
        mujoco.mj_camlight(self.robot_model,self.robot_data)
        return {name:{'pos':self.robot_data.cam_xpos[cid].copy(),'rotation':self.robot_data.cam_xmat[cid].reshape(3,3).copy(),'fovy':float(self.robot_model.cam_fovy[cid])} for name,cid in self.camera_ids.items()}

    @staticmethod
    def _adjust_camera(camera,params):
        return {'pos':camera['pos']+np.asarray(params[:3]),'rotation':Rotation.from_rotvec(params[3:6]).as_matrix()@camera['rotation'],'fovy':camera['fovy']+params[6]}

    def _v6_prepare_gripper(self):
        model=self.robot_model;root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'L_robotiq_2f85');items=[]
        box_faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        for geom in range(model.ngeom):
            body=int(model.geom_bodyid[geom]);descendant=False
            while body>0:
                if body==root:descendant=True;break
                body=int(model.body_parentid[body])
            if not descendant or model.geom_group[geom]!=2:continue
            if int(model.geom_type[geom])==int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);num=int(model.mesh_vertnum[mesh]);fs=int(model.mesh_faceadr[mesh]);fn=int(model.mesh_facenum[mesh])
                vertices=model.mesh_vert[start:start+num].copy();faces=model.mesh_face[fs:fs+fn].copy()
            elif int(model.geom_type[geom])==int(mujoco.mjtGeom.mjGEOM_BOX):
                vertices=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[geom];faces=box_faces
            else:continue
            items.append((geom,vertices,faces,bool(np.max(model.geom_rgba[geom,:3])<.3)))
        return items

    @staticmethod
    def _eef_rotation(observation):
        pose=np.asarray(observation['observation.eef_pose'],float)[:7];q=pose[3:7]
        return pose[:3],Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()

    def _v6_wrist_camera(self,nominal,observation,parameters=None):
        p=self.wrist_mount if parameters is None else np.asarray(parameters,float)
        _,mount_rotation=self._eef_rotation(observation)
        return {'pos':nominal['pos']+mount_rotation@p[:3],
                'rotation':nominal['rotation']@Rotation.from_rotvec(p[3:6]).as_matrix(),
                'fovy':nominal['fovy']+p[6]}

    def _v6_gripper_silhouette(self,camera,shape):
        # Exact fixed robot CAD triangles, transformed only by public joints.
        h,w=shape[:2];polygons=[];candidate=[]
        for geom,vertices,faces,dark in self._v6_gripper_meshes:
            world=vertices@self.robot_data.geom_xmat[geom].reshape(3,3).T+self.robot_data.geom_xpos[geom]
            uv,depth=_project(world,camera,shape);tri=uv[faces];td=depth[faces].mean(1)
            visible=np.all(depth[faces]>.01,axis=1)&(tri[:,:,0].max(1)>=0)&(tri[:,:,0].min(1)<w)&(tri[:,:,1].max(1)>=0)&(tri[:,:,1].min(1)<h)
            polygons.extend((float(td[i]),np.rint(tri[i]).astype(np.int32),255 if dark else 0) for i in np.flatnonzero(visible))
            # Mesh edge midpoints give enough exact 3-D contour candidates.
            if dark:
                edge_points=np.concatenate([world,world[faces[:,0]]*.5+world[faces[:,1]]*.5,world[faces[:,1]]*.5+world[faces[:,2]]*.5])
                candidate.append(edge_points)
        mask=np.zeros((h,w),np.uint8)
        for _,triangle,color in sorted(polygons,key=lambda item:item[0],reverse=True):cv2.fillConvexPoly(mask,triangle,color)
        if not candidate:return mask,np.empty((0,3))
        points=np.concatenate(candidate);uv,z=_project(points,camera,shape)
        edge=cv2.Canny(mask,50,100);dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,3);err,valid=_sample_map(dt,uv)
        keep=valid&(err<.8)&(uv[:,0]>5)&(uv[:,0]<w-6)&(uv[:,1]>5)&(uv[:,1]<h-6)
        points=points[keep];uv=uv[keep]
        if len(points):
            bins=np.floor(uv/3).astype(int);_,indices=np.unique(bins,axis=0,return_index=True);points=points[np.sort(indices)]
        if len(points)>260:points=points[np.linspace(0,len(points)-1,260).astype(int)]
        return mask,points

    def _v6_calibrate_wrist(self,observation,nominal):
        image=np.asarray(observation['observation.images.left_wrist']);predicted,points=self._v6_gripper_silhouette(nominal,image.shape)
        if len(points)<35:return
        dark=(np.max(image,axis=2)<48).astype(np.uint8)*255
        roi=cv2.dilate(predicted,np.ones((17,17),np.uint8));dark=cv2.bitwise_and(dark,roi)
        # The clipped bottom of the image carries no contour information.
        dt=cv2.distanceTransform(255-dark,cv2.DIST_L2,3)-cv2.distanceTransform(dark,cv2.DIST_L2,3)
        opening=float(np.asarray(observation['observation.state'])[7])
        sample={'observation':{'observation.eef_pose':np.asarray(observation['observation.eef_pose']).copy()},'nominal':nominal,'points':points,'dt':dt,'shape':image.shape,'opening':opening}
        if not self.wrist_calibration_samples or abs(opening-self.wrist_calibration_samples[-1]['opening'])>.06:
            self.wrist_calibration_samples.append(sample);self.wrist_calibration_samples=self.wrist_calibration_samples[-3:]
        elif len(self.wrist_calibration_samples)>1:self.wrist_calibration_samples[-1]=sample
        self.wrist_last_opening=opening
        samples=self.wrist_calibration_samples
        def residual(p,regularize=True):
            errors=[]
            for item in samples:
                camera=self._v6_wrist_camera(item['nominal'],item['observation'],p);uv,z=_project(item['points'],camera,item['shape']);err,valid=_sample_map(item['dt'],uv)
                errors.extend(np.clip(err,-10,10))
            r=np.asarray(errors)
            return np.r_[r,2*p[:3]/.003,2*p[3:6]/.02,2*p[6:]/2.] if regularize else r
        bound=np.array([.0045,.0045,.0045,.026,.026,.026,2.5])
        fit=least_squares(residual,self.wrist_mount,bounds=(-bound,bound),max_nfev=45,diff_step=1e-3,loss='soft_l1',f_scale=.7)
        before=float(np.median(np.abs(residual(np.zeros(7),False))));after=float(np.median(np.abs(residual(fit.x,False))))
        # This is a mount-frame CAD pose fit. Its uncertainty stays observable;
        # a tiny residual alone does not establish metric camera calibration.
        data_n=sum(len(item['points']) for item in samples);jac=np.asarray(fit.jac[:data_n]);cov=np.linalg.pinv(jac.T@jac+np.diag(1/np.array([.003,.003,.003,.02,.02,.02,2.])**2))*.5**2
        sigma=np.sqrt(np.maximum(np.diag(cov),0))
        uncertainty=float(np.clip(np.linalg.norm(sigma[:3])+.12*np.linalg.norm(sigma[3:6])+.001,.0015,.004))
        accepted=after<1.6 and after<before*.90 and before-after>.20
        if accepted:self.wrist_mount=fit.x
        self.wrist_calibration={'state':'fitted' if accepted else 'nominal', 'method':'known_gripper_CAD_from_public_joints',
             'mount_parameters':self.wrist_mount.tolist(),'residual_px_before':before,'residual_px_after':after,
             'samples':data_n,'frames':len(samples),'uncertainty_m':uncertainty if accepted else .003,
             'parameter_std':sigma.tolist(),'mount_frame':'EEF rigid mounting frame',
             'reason':'RGB gripper contour fit' if accepted else 'insufficient improvement'}
        self.v6_calibration_debug={'predicted':predicted,'observed':dark,'points':points}

    def _v6_color_mask(self,image):
        rgb=np.asarray(image,float);chrom=rgb/np.maximum(rgb.sum(2,keepdims=True),1.)
        references=np.asarray([[.9,.62,.12],[1.,.78,.2],[.88,.50,.04],[.38,.18,.08],[.4,.6,.2],[.78,.3,.38]],float)
        references=references/references.sum(1,keepdims=True)
        if getattr(self,'cookie_colors',None) is not None:references=self.cookie_colors
        distance=np.min(np.linalg.norm(chrom[:,:,None,:]-references[None,None,:,:],axis=3),axis=2)
        hsv=cv2.cvtColor(np.asarray(image,np.uint8),cv2.COLOR_RGB2HSV)
        return ((distance<.075)&(hsv[:,:,1]>60)&(hsv[:,:,2]>35)).astype(np.uint8)*255

    def _v6_learn_cookie_colors(self,observation,cameras):
        image=np.asarray(observation['observation.images.left_wrist']);source=self.boxes['source_box'];delta=np.asarray(source['center'][:2])-self.scene.source_bin_center_m
        points=np.column_stack([self.layout+delta,np.full(len(self.layout),self.scene.cookie_model_z_m+self.cookie_half[2])]);uv,z=_project(points,cameras['left_wrist'],image.shape)
        mask=np.zeros(image.shape[:2],np.uint8)
        for xy,depth in zip(uv,z):
            if depth>0 and 1<xy[0]<image.shape[1]-2 and 1<xy[1]<image.shape[0]-2:cv2.circle(mask,tuple(np.rint(xy).astype(int)),2,255,-1)
        mask=cv2.bitwise_and(mask,self._v6_color_mask(image));values=image[mask>0].astype(float)
        if len(values)<30:return
        chrom=values/np.maximum(values.sum(1,keepdims=True),1.)
        _,labels,centers=cv2.kmeans(chrom.astype(np.float32),min(3,len(values)),None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,.0005),1,cv2.KMEANS_PP_CENTERS)
        counts=np.bincount(labels.ravel(),minlength=len(centers));self.cookie_colors=centers[counts>max(8,len(values)*.08)]

    def _v6_local_alignment(self,observation,cameras,group,request):
        unknown={'state':'unknown','valid':False,'confidence':0.,'correction_world':np.zeros(3),
                 'group_center':np.asarray(group['center']),'rotation':np.asarray(group['rotation']),
                 'rear_gap_center':None,'gap_width_m':None,'recommended_opening':None,
                 'position_error_m':group.get('position_error_m',.004),'relative_error_m':None,'uncertainty_m':.004,'reason':'not in a pre-insertion view'}
        if str(request.get('stage','')).upper() not in ('LOCAL_ALIGN','ALIGN','APPROACH','OBSERVE_SOURCE'):return unknown
        center=np.asarray(group['center'],float);rotation=np.asarray(group['rotation'],float);count=int(request.get('count',len(group['indices'])))
        eef_pos,eef_rotation=self._eef_rotation(observation);pad_mid=eef_pos+eef_rotation@np.array([0.,-.014676,0.])
        if count<1 or pad_mid[2]-center[2]>.075:return unknown
        im=np.asarray(observation['observation.images.left_wrist']);camera=cameras['left_wrist'];pitch=.008833333333333;thickness=2*self.cookie_half[1]
        first=-(count-1)*pitch/2;last=(count-1)*pitch/2
        ys=np.linspace(first-.006,last+pitch+.006,max(160,int((count*pitch+.012)/.00016)))
        xs=np.linspace(-self.cookie_half[0]*.65,self.cookie_half[0]*.65,20)
        points=np.array([center+rotation@np.array([x,y,self.cookie_half[2]]) for y in ys for x in xs]);uv,depth=_project(points,camera,im.shape)
        valid=(depth>0)&(uv[:,0]>1)&(uv[:,0]<im.shape[1]-2)&(uv[:,1]>1)&(uv[:,1]<im.shape[0]-2)
        if valid.mean()<.86:unknown['reason']='cookie ends or rear neighbour clipped';return unknown
        patch=cv2.remap(im,uv[:,0].astype(np.float32).reshape(len(ys),len(xs)),uv[:,1].astype(np.float32).reshape(len(ys),len(xs)),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        colors=self._v6_color_mask(patch)>0;value=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)[:,:,2].astype(float)
        signal=np.mean(value*np.where(colors,1.,.15),axis=1);signal=cv2.GaussianBlur(signal.astype(np.float32)[:,None],(1,5),0).ravel()
        from scipy.signal import find_peaks
        sample_step=float(ys[1]-ys[0]);peaks,props=find_peaks(signal,prominence=max(5.,float(np.std(signal))*.16),distance=max(4,int(.0045/sample_step)))
        peaks=np.asarray([p for p in peaks if colors[p].mean()>.25],dtype=int);locations=ys[peaks]
        if len(peaks)<(1 if count==1 and int(request.get('source_start_index',0))+count>=20 else max(2,min(count,3))):unknown['reason']='too few visible cookie top bands';return unknown
        def nearest(expected,tolerance=.005):
            if len(locations)==0:return None
            index=int(np.argmin(abs(locations-expected)))
            return float(locations[index]) if abs(locations[index]-expected)<tolerance else None
        observed_first=nearest(first);observed_last=nearest(last)
        has_neighbour=int(request.get('source_start_index',0))+count<20
        observed_neighbour=nearest(last+pitch) if has_neighbour else None
        if observed_first is None or observed_last is None or (has_neighbour and observed_neighbour is None):unknown['reason']='first, last, or rear-neighbour band ambiguous';return unknown
        if has_neighbour:
            separation=observed_neighbour-observed_last
            if not .0045<separation<.014:unknown['reason']='rear neighbour does not define a credible gap';return unknown
            gap=separation-thickness;rear_pad=(observed_neighbour+observed_last)/2
        else:
            gap=.006;rear_pad=observed_last+thickness/2+.00125
        front_pad=observed_first-thickness/2+.00125*-1
        width=rear_pad-front_pad;recommended=float(np.clip((width-.0063616)/.085,0,1))
        desired_mid=center+rotation[:,1]*((front_pad+rear_pad)/2)
        axis=rotation[:,1];relative=float((desired_mid-pad_mid)@axis);correction=axis*relative
        broad=rotation[:,0];broad_error=float((desired_mid-pad_mid)@broad)
        if float(group.get('confidence',0))>=.6:correction+=broad*np.clip(broad_error,-.020,.020)
        # Align transversely at safe height; descent remains the controller's job.
        correction[2]=0.
        gap_center=center+axis*rear_pad+rotation[:,2]*self.cookie_half[2]
        q0,_=_project(gap_center[None],camera,im.shape);q1,_=_project((gap_center+axis*.001)[None],camera,im.shape);px_per_m=max(float(np.linalg.norm(q1-q0))/.001,1.)
        camera_depth=float(np.mean(depth[valid]));height_difference=abs(pad_mid[2]-gap_center[2])
        mount_uncertainty=float(self.wrist_calibration.get('uncertainty_m',.003))
        uncertainty=float(np.sqrt((.45/px_per_m)**2+(.00020+mount_uncertainty*height_difference/max(camera_depth,.04))**2))
        actual_gap=.0053616+.085*float(np.asarray(observation['observation.state'])[7]);opening_error=abs((actual_gap+.001)-width)
        margin=(gap-.001)/2
        selected=locations[(locations>=observed_first-.001)&(locations<=observed_last+.001)]
        consistent=len(selected)==count and (len(selected)<3 or float(np.std(np.diff(selected)))<.0015)
        quality=float(np.clip(.80 if consistent else .5,0,.9))
        if gap<=.001:state='unknown';reason='observed rear gap is narrower than the pad'
        elif abs(relative)+uncertainty<margin and opening_error<.0008 and abs(broad_error)<.003:state='aligned';reason='relative pad-gap error fits observed clearance'
        elif quality>=.6 and (abs(relative)>.0004 or opening_error>.0006 or abs(broad_error)>.003):state='correct';reason='visible end bands provide a row-axis correction'
        else:state='unknown';reason='relative uncertainty exceeds available gap clearance'
        group_center=center+axis*((observed_first+observed_last)/2)
        self.v6_local_debug={'patch':patch,'signal':signal,'ys':ys,'peaks':peaks}
        return {'state':state,'valid':state=='aligned','confidence':quality,'correction_world':np.clip(correction,-.004,.004),
                'group_center':group_center,'rotation':rotation,'rear_gap_center':gap_center,'gap_width_m':float(gap),
                'recommended_opening':recommended,'gap_inner_width_m':float(width-.001),'position_error_m':group.get('position_error_m',.004),
                'relative_error_m':abs(relative),'relative_signed_error_m':relative,'opening_error_m':opening_error,'uncertainty_m':uncertainty,
                'observable_axis_world':axis,'broad_error_m':broad_error,'measurement_step':self.step,'measurement_id':f'{self.selected_group_key}:{self.step}','reason':reason,'visible_peak_count':int(len(selected)),'expected_count':count}

    def _v6_fit_colored_box(self,image,camera,center,rotation,half,translation_bound=.025,rotation_bound=.5,multistart=False,use_bevel=False):
        center=np.asarray(center,float);rotation=np.asarray(rotation,float);half=np.asarray(half,float)
        if use_bevel:
            bevel=min(float(self.scene.cookie_edge_bevel_m),half[1]*.95);bottom=min(float(self.scene.cookie_bottom_edge_bevel_m or self.scene.cookie_edge_bevel_m),half[1]*.95)
            hx,hy,hz=half;profile=np.array([[-hy,-hz+bottom],[-hy+bottom,-hz],[hy-bottom,-hz],[hy,-hz+bottom],[hy,hz-bevel],[hy-bevel,hz],[-hy+bevel,hz],[-hy,hz-bevel]])
            corners=np.array([[x,y,z] for x in(-hx,hx) for y,z in profile])
        else:
            corners=np.array([[x,y,z] for x in (-half[0],half[0]) for y in (-half[1],half[1]) for z in (-half[2],half[2])])
        world=corners@rotation.T+center;uv,depth=_project(world,camera,image.shape)
        if np.any(depth<.01):return None
        hull=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel();base_points=np.concatenate([corners[hull[j]]+np.linspace(0,1,22,endpoint=False)[:,None]*(corners[hull[(j+1)%len(hull)]]-corners[hull[j]]) for j in range(len(hull))])
        mask=self._v6_color_mask(image);roi=np.zeros(mask.shape,np.uint8)
        lo=np.maximum(np.floor(uv.min(0)-32).astype(int),0);hi=np.minimum(np.ceil(uv.max(0)+32).astype(int),np.array([image.shape[1]-1,image.shape[0]-1]))
        cv2.rectangle(roi,tuple(lo),tuple(hi),255,-1);mask=cv2.bitwise_and(mask,roi);mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
        predicted_center,_=_project(center[None],camera,image.shape);predicted_area=max(float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32)))),1.)
        choices=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>max(12,predicted_area*.10)]
        if not choices:return None
        chosen=min(choices,key=lambda i:np.linalg.norm(centroids[i]-predicted_center[0])+8*abs(np.log(max(stats[i,cv2.CC_STAT_AREA],1)/predicted_area)))
        selected=(labels==chosen).astype(np.uint8)*255
        contours,_=cv2.findContours(selected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);filled=np.zeros_like(selected);cv2.drawContours(filled,contours,-1,255,-1)
        dt=cv2.distanceTransform(255-filled,cv2.DIST_L2,3)-cv2.distanceTransform(filled,cv2.DIST_L2,3)
        def geometry(parameters):return center+rotation@parameters[:3],rotation@Rotation.from_rotvec(parameters[3:]).as_matrix()
        def residual(parameters):
            c,r=geometry(parameters);pixels,z=_project(base_points@r.T+c,camera,image.shape);err,valid=_sample_map(dt,pixels)
            return np.r_[np.clip(err,-12,12),parameters[:3]/.015,parameters[3:]/.5]
        bounds=np.r_[np.full(3,translation_bound),np.full(3,rotation_bound)]
        seeds=[np.zeros(6)]
        if multistart:
            for axis in (0,1):
                for angle in (-.55,.55):
                    seed=np.zeros(6);seed[axis+3]=np.clip(angle,-rotation_bound+.01,rotation_bound-.01);seeds.append(seed)
        fits=[least_squares(residual,seed,bounds=(-bounds,bounds),diff_step=1e-3,max_nfev=28,loss='soft_l1',f_scale=.8) for seed in seeds]
        fit=min(fits,key=lambda f:f.cost);c,r=geometry(fit.x);pixels,z=_project(corners@r.T+c,camera,image.shape)
        model_mask=np.zeros_like(mask);cv2.fillConvexPoly(model_mask,np.rint(cv2.convexHull(pixels.astype(np.float32))).astype(np.int32),255)
        intersection=float(np.sum((model_mask>0)&(filled>0)));union=float(np.sum((model_mask>0)|(filled>0)));iou=intersection/max(union,1.)
        err=float(np.median(np.abs(residual(fit.x)[:-6])));area_ratio=float(np.sum(filled>0))/max(float(np.sum(model_mask>0)),1.)
        confidence=float(np.clip((iou-.45)/.45,0,1)*np.clip(1-err/3.,0,1))
        observed_uv=np.column_stack(np.where(filled>0)[::-1]);border=bool(np.any((observed_uv[:,0]<2)|(observed_uv[:,0]>image.shape[1]-3)|(observed_uv[:,1]<2)|(observed_uv[:,1]>image.shape[0]-3)))
        if border:confidence=min(confidence,.35)
        return {'center':c,'rotation':r,'confidence':confidence,'residual_px':err,'iou':iou,'area_ratio':area_ratio,'mask':filled,'corners':corners@r.T+c,'observed_pixels':observed_uv,'parameters':fit.x}

    def _v6_held_observation(self,observation,cameras,group,request,count,count_confidence):
        stage=str(request.get('stage','')).upper();unknown={'count':count,'count_confidence':float(count_confidence),'pose_confidence':0.,'center_tool':None,'rotation_tool':None,'width_m':None,'bounds_tool':None,'slip_m':None,'slip_rad':None,'following':None,'released':None,'reason':'held geometry not observable','measurement_step':self.step,'measurement_id':f'held:{self.selected_group_key}:{self.step}','geometry_method':self.v6_held_method or 'unobserved','geometry_generation':self.v6_held_generation}
        if stage not in ('TRIAL_LIFT','LIFT','TRANSPORT','PLACE','REPAIR_LIFT','REPAIR_DESCEND'):return unknown
        touch=np.asarray(observation['observation.force'],float)[12:14];eef_pos,eef_rotation=self._eef_rotation(observation)
        measured_width=.0053616+.085*float(np.asarray(observation['observation.state'])[7])
        pose_count=count
        if (pose_count is None or count_confidence<.6) and self.v6_held_previous is not None:
            previous_count=self.v6_held_previous.get('count')
            if previous_count is not None and abs(measured_width-int(previous_count)*2*self.cookie_half[1])<.0025:pose_count=previous_count
        if pose_count is None or min(touch)<.1:return unknown
        if count_confidence<.6 and self.v6_held_previous is None:return unknown
        n=int(pose_count)
        if abs(measured_width-n*2*self.cookie_half[1])>.0025:return unknown
        half=np.array([self.cookie_half[0],measured_width/2,self.cookie_half[2]])
        if self.v6_held_previous is not None and self.v6_held_previous.get('pose_confidence',0)>.5:
            relative_center=np.asarray(self.v6_held_previous['center_tool']);relative_rotation=np.asarray(self.v6_held_previous['rotation_tool'])
        else:
            reference=request.get('grasp_reference_eef');reference=np.asarray(reference,float) if reference is not None else None
            if reference is not None and reference.size>=7:
                q=reference[3:7];reference_rotation=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
            else:reference_rotation=eef_rotation
            relative_rotation=reference_rotation.T@np.asarray(group['rotation']);relative_center=np.array([0.,-.014676,0.])-relative_rotation[:,2]*.010
        center=eef_pos+eef_rotation@relative_center;rotation=eef_rotation@relative_rotation
        if center[2]-self.cookie_half[2] < self.scene.source_wall_base_z_m+self.scene.source_bin_wall_height_m+.004 and stage!='PLACE':
            unknown['reason']='trial lift still overlaps source neighbourhood';return unknown
        use_bevel=self.v6_use_bevel_held or (self.v6_wrist_target_complete and stage in ('TRANSPORT','PLACE'))
        method='beveled_CAD' if use_bevel else 'cuboid'
        fitted=self._v6_fit_colored_box(np.asarray(observation['observation.images.left_wrist']),cameras['left_wrist'],center,rotation,half,use_bevel=use_bevel)
        if fitted is None:return unknown
        ctool=eef_rotation.T@(fitted['center']-eef_pos);rtool=eef_rotation.T@fitted['rotation'];tool_corners=(fitted['corners']-eef_pos)@eef_rotation
        confidence=fitted['confidence'];reason='RGB group CAD silhouette fit'
        # A reliable fit must explain the full coloured object, not a cropped part.
        if not .70<fitted['area_ratio']<1.35:confidence=min(confidence,.4);reason='coloured object partly occluded or merged with background'
        reference_reset_reason=None
        if confidence>=.6:
            if self.v6_held_method is not None and self.v6_held_method!=method:
                self.v6_held_generation+=1;self.v6_held_reference=None;reference_reset_reason='geometry model refined; same-model slip reference restarted'
                self.v6_held_geometry_history.append({'step':self.step,'from':self.v6_held_method,'to':method,'generation':self.v6_held_generation,'reason':reference_reset_reason})
            self.v6_held_method=method;self.v6_use_bevel_held=use_bevel
            if self.v6_held_reference is None:self.v6_held_reference=(ctool.copy(),rtool.copy())
            slip=float(np.linalg.norm(ctool-self.v6_held_reference[0]));slip_rad=float(np.linalg.norm(Rotation.from_matrix(self.v6_held_reference[1].T@rtool).as_rotvec()))
        else:slip=None;slip_rad=None
        result={**unknown,'pose_confidence':float(confidence),'center_tool':ctool,'rotation_tool':rtool,'width_m':float(measured_width),
             'bounds_tool':np.column_stack([tool_corners.min(0),tool_corners.max(0)]),'pose_count_reference':n,'slip_m':slip,'slip_rad':slip_rad,
             'following':True if confidence>=.6 else None,'released':False if confidence>=.6 else None,'reason':reason,
             'geometry_method':method,'geometry_generation':self.v6_held_generation,'reference_reset_reason':reference_reset_reason,'residual_px':fitted['residual_px'],'iou':fitted['iou'],'uncertainty_m':max(.0015,fitted['residual_px']*.001)}
        if confidence>=.5:
            self.v6_held_previous={**result,'count':n}
        self.v6_held_debug=fitted
        return result

    def _v6_release_motion(self,observation,cameras,request,held):
        stage=str(request.get('stage','')).upper()
        if stage not in ('OPEN','RELEASE_LIFT','RETRACT','VERIFY','TARGET_OBSERVE','REPAIR_OPEN','REPAIR_RETRACT'):return held
        image=np.asarray(observation['observation.images.left_wrist']);gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);eef_pos,eef_rotation=self._eef_rotation(observation)
        track=getattr(self,'v6_release_track',None)
        if track is None:
            previous=self.v6_held_previous
            if previous is None or previous.get('pose_confidence',0)<.5:return held
            c=eef_pos+eef_rotation@np.asarray(previous['center_tool']);r=eef_rotation@np.asarray(previous['rotation_tool']);half=np.array([self.cookie_half[0],float(previous['width_m'])/2,self.cookie_half[2]])
            corners=np.array([[x,y,z] for x in(-half[0],half[0]) for y in(-half[1],half[1]) for z in(-half[2],half[2])]);uv,z=_project(corners@r.T+c,cameras['left_wrist'],image.shape)
            roi=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(roi,np.rint(cv2.convexHull(uv.astype(np.float32))).astype(int),255);roi=cv2.dilate(roi,np.ones((5,5),np.uint8));roi=cv2.bitwise_and(roi,self._v6_color_mask(image))
            points=cv2.goodFeaturesToTrack(gray,60,.01,3,mask=roi,blockSize=3)
            if points is None or len(points)<6:return held
            self.v6_release_track={'previous_gray':gray,'previous_points':points,'anchor_points':points.copy(),'anchor_eef':eef_pos.copy(),'anchor_rotation':eef_rotation.copy(),'measurement_step':self.step}
            return {**held,'motion_confidence':0.,'reason':'release feature tracks initialized'}
        new,status,error=cv2.calcOpticalFlowPyrLK(track['previous_gray'],gray,track['previous_points'],None,winSize=(19,19),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,20,.01))
        if new is None:return held
        backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray,track['previous_gray'],new,None,winSize=(19,19),maxLevel=2)
        if backward is None:return held
        valid=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(backward-track['previous_points'],axis=2).ravel()<.8)
        if np.sum(valid)<6:self.v6_release_track=None;return {**held,'motion_confidence':0.,'reason':'release tracks occluded'}
        points=new[valid];anchors=track['anchor_points'][valid];motion=float(np.median(np.linalg.norm(points-anchors,axis=2)))
        translation=float(np.linalg.norm(eef_pos-track['anchor_eef']));angle=float(np.linalg.norm(Rotation.from_matrix(track['anchor_rotation'].T@eef_rotation).as_rotvec()))
        track.update(previous_gray=gray,previous_points=points,anchor_points=anchors)
        touch=np.asarray(observation['observation.force'],float)[12:14];result={**held,'motion_confidence':0.,'image_motion_px':motion,'tcp_motion_m':translation,'measurement_step':self.step,'measurement_id':f'release:{self.selected_group_key}:{self.step}'}
        if translation>.010 or angle>.10:
            if motion>2.5 and max(touch)<.12:
                result.update(following=False,released=True,motion_confidence=.70,reason='coloured object moved relative to the withdrawing wrist; finger touch cleared')
            elif motion<1.0:
                result.update(following=True,released=False,motion_confidence=.65,reason='object features remain fixed relative to the moving wrist')
        return result

    def _v6_target_in_wrist(self,camera,shape,box):
        half=np.asarray(self.scene.target_bin_half_size_m)+self.scene.bin_wall_thickness_m
        corners=np.array([[x,y,z] for x in(-half[0],half[0]) for y in(-half[1],half[1]) for z in(0.,self.scene.target_bin_wall_height_m)])
        points=corners@_rot_z(float(box.get('yaw',0))).T+np.asarray(box['center']);uv,depth=_project(points,camera,shape)
        return bool(np.all(depth>0) and np.all(uv[:,0]>3) and np.all(uv[:,0]<shape[1]-4) and np.all(uv[:,1]>3) and np.all(uv[:,1]<shape[0]-4))

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
            filled=np.zeros_like(gray);cv2.drawContours(filled,[c for c in contours if cv2.contourArea(c)>15],-1,255,-1)
            distance[name]=cv2.distanceTransform(255-filled,cv2.DIST_L2,3)-cv2.distanceTransform(filled,cv2.DIST_L2,3)
        boxes={}
        for kind in ('source','target'):
            if kind=='source' and self.source_disturbed and self.v6_source_anchor is not None:
                boxes['source_box']={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in self.v6_source_anchor.items()};continue
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
            # The public wrist pose provides the metric world anchor; a drifted
            # overview camera otherwise drags the source fit by centimetres.
            if kind=='source' and any(n=='left_wrist' and len(p)>25 for n,p in cameras_used):
                cameras_used=[item for item in cameras_used if item[0]=='left_wrist']
            if kind=='target' and any(n=='left_wrist' for n,pts in cameras_used):
                if self._v6_target_in_wrist(cameras['left_wrist'],distance['left_wrist'].shape,{'center':np.r_[initial[:2],center_z],'yaw':initial[2]}):
                    cameras_used=[item for item in cameras_used if item[0]=='left_wrist']
            if not cameras_used:
                boxes[kind+'_box']={'center':np.r_[initial[:2],center_z],'yaw':float(initial[2]),'confidence':0.,'position_error_m':.025,'residual_px':None};continue
            def residual(p):
                pts=points@_rot_z(p[2]).T+np.r_[p[:2],center_z];res=[]
                for name,subset in cameras_used:
                    uv,z=_project(subset@_rot_z(p[2]).T+np.r_[p[:2],center_z],cameras[name],distance[name].shape)
                    err,v=_sample_map(distance[name],uv)
                    res.extend(np.clip(err,-15,15))
                return np.r_[res,(p[:2]-initial[:2])/.035,(p[2]-initial[2])/.12]
            lo=initial-bounds;hi=initial+bounds
            if kind=='target':
                nominal_xy=np.asarray(self.scene.target_bin_world_position_m[:2]);lo[:2],hi[:2]=nominal_xy-.045,nominal_xy+.045
                lo[2],hi[2]=-.085,.085;initial[2]=np.clip(initial[2],-.0849,.0849)
            else:
                nominal_xy=np.asarray(self.scene.source_bin_center_m);lo[:2],hi[:2]=nominal_xy-.025,nominal_xy+.025
            initial[:2]=np.clip(initial[:2],lo[:2]+1e-7,hi[:2]-1e-7)
            fit=least_squares(residual,initial,bounds=(lo,hi),diff_step=1e-3,max_nfev=35,loss='soft_l1',f_scale=1.)
            err=float(np.median(np.abs(residual(fit.x)[:-3])));confidence=float(np.clip(1-err/4,0,1))
            boxes[kind+'_box']={'center':np.r_[fit.x[:2],center_z],'yaw':float(fit.x[2]),'confidence':confidence,'position_error_m':max(.0035 if kind=='source' else .006,.0015+err*.0015),'residual_px':err,'pose_camera':'+'.join(n for n,_ in cameras_used)}
        candidate=boxes.get('target_box')
        if candidate is not None:
            supports=[]
            for name in ('front','left_wrist'):
                points=self._box_points('target');world=points@_rot_z(candidate['yaw']).T+candidate['center'];uv,z=_project(world,cameras[name],distance[name].shape)
                sampled,valid=_sample_map(masks[name].astype(float)/255.,uv,default=0.)
                if np.sum(valid)>30:supports.append(float(np.mean(sampled[valid]>.35)))
            rim_support=max(supports,default=0.);candidate['visible_rim_fraction']=rim_support;candidate['calibration_generation']=self.calibration_generation
            old=self.v6_target_anchor
            if old is None:self.v6_target_anchor={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in candidate.items()}
            else:
                movement=float(np.linalg.norm(np.asarray(candidate['center'])[:2]-np.asarray(old['center'])[:2]))
                # An occluded PLACE fit cannot overwrite a clear earlier box pose.
                # A large genuine displacement is accepted after a clear overview.
                precontact=self.v6_request_stage in ('SELECT','APPROACH','LOCAL_ALIGN','ALIGN_APPROACH','ALIGN_OBSERVE','SOURCE_OBSERVE','TRIAL_LIFT','LIFT','TRANSPORT','TARGET_OBSERVE')
                clear=candidate['confidence']>=.60 and rim_support>=.50
                # Before target contact a clear new view may correct an uncertain
                # initial metric pose by centimetres. PLACE does not re-anchor.
                consistent=precontact or movement<.004
                candidate['visible']=clear;candidate['occluded']=not clear
                if clear and consistent:self.v6_target_anchor={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in candidate.items()}
                else:
                    boxes['target_box']={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in old.items()}
                    boxes['target_box']['update_rejected']={'confidence':candidate['confidence'],'rim_fraction':rim_support,'movement_m':movement,'candidate_center':np.asarray(candidate['center']).tolist(),'candidate_yaw':candidate['yaw'],'reason':'occlusion or geometric inconsistency'}
        if not self.source_disturbed and 'source_box' in boxes and boxes['source_box']['confidence']>.5:
            self.v6_source_anchor={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in boxes['source_box'].items()}
        return boxes

    def _calibrate_front_from_source(self,observation,cameras,delta):
        """Anchor front extrinsics using source CAD positioned by the wrist view."""
        name='front';camera=cameras[name];im=np.asarray(observation['observation.images.'+name]);mx=im.max(2);mn=im.min(2)
        gray=((mx.astype(int)-mn.astype(int)<38)&(mx>65)&(mx<250)&(im[:,:,2].astype(int)+10>=im[:,:,0].astype(int))).astype(np.uint8)*255
        contours,_=cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);edge=np.zeros_like(gray);cv2.drawContours(edge,[c for c in contours if cv2.contourArea(c)>15],-1,255,1)
        filled=np.zeros_like(gray);cv2.drawContours(filled,[c for c in contours if cv2.contourArea(c)>15],-1,255,-1)
        dt=cv2.distanceTransform(255-filled,cv2.DIST_L2,3)-cv2.distanceTransform(filled,cv2.DIST_L2,3)
        half=np.asarray(self.scene.source_bin_half_size_m)+self.scene.bin_wall_thickness_m;zb=self.scene.source_wall_base_z_m;zt=zb+self.scene.source_bin_wall_height_m
        center=np.r_[np.asarray(self.scene.source_bin_center_m)+delta,0.]
        corners=np.array([[x,y,z] for z in (zb,zt) for x in (-half[0],half[0]) for y in (-half[1],half[1])])+center
        uv,z=_project(corners,camera,im.shape);hull=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel()
        points=np.concatenate([corners[hull[j]]+np.linspace(0,1,35,endpoint=False)[:,None]*(corners[hull[(j+1)%len(hull)]]-corners[hull[j]]) for j in range(len(hull))])
        def residual(p):
            cam=self._adjust_camera(camera,p);uv,z=_project(points,cam,im.shape);err,v=_sample_map(dt,uv)
            return np.r_[np.clip(err,-10,10),5*p[:3]/.007,5*p[3:6]/.03,5*p[6:]/2.]
        bound=np.array([.015,.015,.015,.06,.06,.06,3.])
        fit=least_squares(residual,np.zeros(7),bounds=(-bound,bound),diff_step=1e-3,max_nfev=30,loss='soft_l1',f_scale=.8)
        before=float(np.mean(residual(np.zeros(7))[:-7]**2));after=float(np.mean(residual(fit.x)[:-7]**2))
        if after<before*.85:
            self.camera_adjustments['front']=fit.x
            return {'method':'source_CAD_anchored_to_wrist_RGB','residual_px':float(np.sqrt(after)),'adjustment':fit.x.tolist()}
        return {'method':'nominal','residual_px':float(np.sqrt(before))}

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
        row_min=float(np.min(self.layout[:,1]));edge_weights=np.repeat(np.where(np.isclose(self.layout[indices,1],row_min),4.,1.),len(local))
        initial=np.asarray(source['center'][:2])-np.asarray(self.scene.source_bin_center_m)
        maps=[]
        for name in ('front','left_wrist'):
            im=np.asarray(observation['observation.images.'+name]);gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            edge=cv2.Canny(gray,35,85);dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
            uv,z=_project(points+np.r_[initial,0.],cameras[name],im.shape)
            valid=(z>0)&(uv[:,0]>2)&(uv[:,0]<im.shape[1]-3)&(uv[:,1]>2)&(uv[:,1]<im.shape[0]-3)
            if valid.sum()>100:maps.append((name,dt,points[valid],edge_weights[valid]))
        if any(item[0]=='left_wrist' for item in maps):maps=[item for item in maps if item[0]=='left_wrist']
        if not maps:return initial,source['confidence'],None
        def residual(delta):
            errs=[]
            for name,dt,pts,weights in maps:
                uv,z=_project(pts+np.r_[delta,0.],cameras[name],dt.shape);r,v=_sample_map(dt,uv)
                errs.extend(np.minimum(r,7.)*weights)
            return np.r_[errs,(delta-initial)/.004]
        candidates=[least_squares(residual,initial+[0.,offset],bounds=(initial-np.array([.003,.008]),initial+np.array([.003,.008])),diff_step=1e-3,max_nfev=22,loss='soft_l1',f_scale=.8) for offset in (-.004,0.,.004)]
        fit=min(candidates,key=lambda candidate:candidate.cost)
        err=float(np.median(residual(fit.x)[:-2]));confidence=float(np.clip(1-err/2.5,0,.8))
        return fit.x,confidence,err

    def _fit_selected_group(self,observation,cameras,centers,indices,source_error,allow_tilt=True):
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
        if any(item[0]=='left_wrist' for item in maps):maps=[item for item in maps if item[0]=='left_wrist']
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
        tilt_observed=allow_tilt and score < base_score*.70 and abs(p[2])>np.deg2rad(8.)
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
        if stage not in ('TRIAL_LIFT','LIFT','TRANSPORT','PLACE','REPAIR_LIFT','REPAIR_DESCEND'):return None,0.
        pose=np.asarray(observation['observation.eef_pose'],float)
        q=pose[3:7];rotation=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
        opening=float(np.asarray(observation['observation.state'])[7])
        gap=.0053616+.085*opening
        pad_center=pose[:3]+rotation@np.array([0.,-.014676,0.])
        up=-rotation[:,1];jaw=rotation[:,0];broad=rotation[:,2]
        center=pad_center-up*.010
        name='left_wrist';im=np.asarray(observation['observation.images.'+name]);camera=cameras[name]
        half_width=gap/2+.0015;hx=self.cookie_half[0]
        xs=np.linspace(-hx*.65,hx*.65,24);ys=np.linspace(-half_width,half_width,240)
        points=np.array([center+broad*x+jaw*y+up*self.cookie_half[2] for y in ys for x in xs])
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
        peaks,properties=find_peaks(signal,prominence=max(5.,float(np.std(signal))*.3),distance=max(5,int(.0045/(2*half_width)*240)))
        peaks=np.asarray([k for k in peaks if support[k]],dtype=int)
        if not len(peaks):return None,0.
        # Restrict to the contiguous colour-supported group nearest predicted centre.
        near=peaks[np.abs(ys[peaks])<gap/2+.001]
        if len(near)<1 or len(near)>12:return None,0.
        gaps=np.diff(ys[near]);regular= len(gaps)==0 or np.all((gaps>.0045)&(gaps<.0115))
        physical_count=int(round(gap/(2*self.cookie_half[1])))
        width_residual=abs(gap-physical_count*2*self.cookie_half[1])
        agrees=abs(len(near)-physical_count)==0 and width_residual<.0025
        # Width is a consistency check, never a replacement for RGB band count.
        confidence=.70 if regular and agrees and float(np.mean(sat[near]))>90 else .35
        return int(len(near)),confidence

    def _count_target(self,observation,cameras,request):
        """Estimate upright target occupancy from rectified coloured top bands.

        The metric is measured coloured length divided by fixed cookie thickness.
        No completed/expected batch count is used. Occluded or clipped views stay
        unknown and disagreement between views lowers confidence.
        """
        stage=str(request.get('stage','')).upper()
        if stage not in ('PLACE','OPEN','RETRACT','VERIFY'):return None,0.
        box=self.boxes['target_box'];rotation=_rot_z(box['yaw']);center=np.asarray(box['center'])
        slots=np.asarray(self.scene.target_slots_local_m);xs=np.unique(slots[:,0]);hy=float(self.scene.target_bin_half_size_m[1])-self.scene.bin_wall_thickness_m
        ztop=float(self.scene.target_floor_z_m)+self.scene.bin_wall_thickness_m/2+2*self.cookie_half[2]
        ys=np.linspace(-hy+.001,hy-.001,320);span=float(ys[-1]-ys[0]);thickness=2*self.cookie_half[1]
        estimates=[];diagnostics={}
        for name in ('front','left_wrist','right_wrist'):
            image=np.asarray(observation['observation.images.'+name]);counts=[];valid_view=True;metrics=[]
            for x in xs:
                xstrip=np.linspace(x-self.cookie_half[0]*.55,x+self.cookie_half[0]*.55,20)
                local=np.array([[xx,y,ztop] for y in ys for xx in xstrip]);world=local@rotation.T+center
                uv,depth=_project(world,cameras[name],image.shape)
                valid=(depth>0)&(uv[:,0]>1)&(uv[:,0]<image.shape[1]-2)&(uv[:,1]>1)&(uv[:,1]<image.shape[0]-2)
                if valid.mean()<.96:valid_view=False;break
                patch=cv2.remap(image,uv[:,0].astype(np.float32).reshape(320,20),uv[:,1].astype(np.float32).reshape(320,20),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
                black=float(np.mean(hsv[:,:,2]<28))
                if black>.12:valid_view=False;break
                colored=((hsv[:,:,1]>65)&(hsv[:,:,2]>35)).mean(1)
                support=(colored>.50).astype(np.uint8)
                support=cv2.medianBlur(support[:,None],5).ravel()
                occupied_length=float(support.mean()*span)
                count_float=occupied_length/thickness
                # Repeated bevel bands provide a second count cue; the minimum
                # physical separation pairs the two edges of one cookie.
                signal=np.mean(hsv[:,:,2].astype(float),axis=1)*np.clip(colored*2,0,1)
                signal=cv2.GaussianBlur(signal.astype(np.float32)[:,None],(1,5),0).ravel()
                from scipy.signal import find_peaks
                peaks,properties=find_peaks(signal,prominence=max(5.,float(np.std(signal))*.22),distance=max(5,int(.0047/span*320)))
                peaks=np.asarray([i for i in peaks if colored[i]>.40],dtype=int)
                peak_count=int(len(peaks))
                if peak_count==0 and occupied_length>thickness:
                    valid_view=False;break
                counts.append(float(peak_count));metrics.append({'occupied_length_m':occupied_length,'length_count':count_float,'count_float':float(peak_count),'black_fraction':black,'peaks':peak_count})
            if valid_view and len(counts)==len(xs):
                total=float(sum(counts));estimate=int(np.clip(round(total),0,len(slots)))
                estimates.append((name,estimate,total));diagnostics[name]=metrics
        self.target_count_diagnostics=diagnostics
        if not estimates:
            self.target_placement_valid=None;self.target_placement_confidence=0.
            return None,0.
        # Front is the stable overview; a nearby wrist supplies an independent
        # confirmation when the whole target rectangle is in view.
        preferred=next((e for e in estimates if e[0]=='front'),estimates[0])
        disagreement=any(abs(item[1]-preferred[1])>1 for item in estimates)
        quantized=abs(preferred[2]-preferred[1])<.30
        confidence=.65 if len(estimates)>1 and not disagreement and quantized and box['confidence']>.5 else .4
        if disagreement:confidence=.25
        self.target_placement_valid=None;self.target_placement_confidence=0.
        # RGB quantity alone does not establish upright, supported placement.
        # This audit stays unknown until multi-view height/pose evidence is fitted.
        return preferred[1],confidence

    def estimate(self,observation,request=None):
        request=request or {};self.step+=1
        self.v6_request_stage=str(request.get('stage','')).upper()
        touch=np.asarray(observation.get('observation.force',np.zeros(18)),float)[12:14]
        self.source_disturbed=self.source_disturbed or bool(np.max(touch)>.5)
        cameras=self._robot_cameras(observation)
        opening=float(np.asarray(observation['observation.state'])[7])
        if self.step==1 or (not self.source_disturbed and self.step<35 and self.step%5==0 and (self.wrist_last_opening is None or abs(opening-self.wrist_last_opening)>.08)):
            self._v6_calibrate_wrist(observation,cameras['left_wrist'])
        cameras['left_wrist']=self._v6_wrist_camera(cameras['left_wrist'],observation)
        if self.step==1:self.calibration={}
        cameras={n:self._adjust_camera(c,self.camera_adjustments.get(n,np.zeros(7))) for n,c in cameras.items()}
        key=(request.get('batch_index'),request.get('stage'))
        stage=str(request.get('stage',''))
        was_complete=self.v6_wrist_target_complete
        self.v6_wrist_target_complete=self.boxes is not None and self._v6_target_in_wrist(cameras['left_wrist'],np.asarray(observation['observation.images.left_wrist']).shape,self.boxes['target_box'])
        near_target_refresh=stage.upper()=='TRANSPORT' and self.v6_wrist_target_complete and (not was_complete or self.step-self.last_fit_step>=4)
        refit=near_target_refresh or bool(request.get('force_refit',False)) or self.boxes is None or (key!=self.last_key and any(s in stage.upper() for s in ('SELECT','APPROACH','ALIGN','VERIFY','OBSERVE','PLAN'))) or self.step-self.last_fit_step>=30
        if refit:
            self.boxes=self._fit_boxes(observation,cameras);self.last_fit_step=self.step
            self.cookie_delta,self.cookie_fit_confidence,self.cookie_fit_error=self._refine_cookie_layout(observation,cameras,self.boxes['source_box'],request)
            if self.step==1:
                self.calibration={'front':self._calibrate_front_from_source(observation,cameras,self.cookie_delta)}
                self.calibration_generation+=1;self.v6_target_anchor=None
                cameras['front']=self._adjust_camera(cameras['front'],self.camera_adjustments.get('front',np.zeros(7)))
                self.boxes=self._fit_boxes(observation,cameras)
        self.last_key=key
        self.last_cameras=cameras
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
            self.v6_held_previous=None;self.v6_held_reference=None;self.v6_held_method=None;self.v6_held_generation=0;self.v6_use_bevel_held=False;self.v6_release_track=None;self.v6_held_cache=None
        if self.selected_group is None or bool(request.get('force_refit',False)) or (refit and stage in ('SELECT','APPROACH','LOCAL_ALIGN','OBSERVE_SOURCE','DESCEND','CLOSE')):
            fitted=self._fit_selected_group(observation,cameras,centers,indices,source['position_error_m'],allow_tilt=self.source_disturbed or start>0)
            if fitted is not None:self.selected_group=fitted
        if self.selected_group is not None:
            fitted=self.selected_group;group.update({k:v for k,v in fitted.items() if k!='centers'})
            centers[indices]=fitted['centers'];rotations[indices]=fitted['rotation'];confidence[indices]=fitted['confidence']
        if stage in ('SELECT','APPROACH','LOCAL_ALIGN','OBSERVE_SOURCE','DESCEND','CLOSE'):
            self.held_anchor=(np.asarray(observation['observation.eef_pose'],float)[:3].copy(),np.asarray(group['center']).copy())
        if self.cookie_colors is None:self._v6_learn_cookie_colors(observation,cameras)
        local_align=self._v6_local_alignment(observation,cameras,group,request)
        held_count,held_confidence=self._count_visible_held(observation,cameras,group,request) if self.step%5==0 or key!=self.last_count_key else self.last_held_count
        self.last_count_key=key;self.last_held_count=(held_count,held_confidence)
        if self.step%5==0 or key!=self.last_target_key:
            self.last_target_count=self._count_target(observation,cameras,request);self.last_target_key=key
        target_count,target_confidence=self.last_target_count
        if near_target_refresh or self.step%4==0 or key!=getattr(self,'v6_held_key',None) or getattr(self,'v6_held_cache',None) is None:
            self.v6_held_cache=self._v6_held_observation(observation,cameras,group,request,held_count,held_confidence);self.v6_held_key=key
        held=self._v6_release_motion(observation,cameras,request,self.v6_held_cache)
        if self.placement_auditor is None:
            try:
                from placement_audit import PlacementAuditor
            except ModuleNotFoundError as exc:
                if exc.name!='placement_audit':raise
            else:self.placement_auditor=PlacementAuditor(self.config,color_mask=self._v6_color_mask)
        if self.placement_auditor is not None and (self.step%5==0 or key!=getattr(self,'v6_placement_key',None)):
            self.v6_placement_cache=self.placement_auditor.estimate(observation,cameras,self.boxes['target_box'],request,held);self.v6_placement_key=key
        placement=self.v6_placement_cache
        return {**self.boxes,'cookies':{'centers':centers,'rotations':rotations,'visible':np.isin(np.arange(len(centers)),indices)&(confidence>.6),'confidence':confidence},'group':group,'placement':placement,'target_occupancy':placement.get('occupied_bounds_by_column'),'held':held,'slip_m':held.get('slip_m'),'local_align':local_align,'wrist_calibration':self.wrist_calibration,'held_count':held_count,'held_count_confidence':held_confidence,'target_count':target_count,'target_count_confidence':target_confidence,'target_count_diagnostics':self.target_count_diagnostics,'target_placement_valid':self.target_placement_valid,'target_placement_confidence':self.target_placement_confidence,'camera_calibration':getattr(self,'calibration',{}),'source_tilt_deg':group['tilt_deg'],'position_error_m':source['position_error_m']}
