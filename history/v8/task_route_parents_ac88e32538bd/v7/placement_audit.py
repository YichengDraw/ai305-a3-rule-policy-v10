"""RGB-only multi-view placement audit for the fixed cookie CAD.

A freely fitted colored group envelope supplies position, tilt, packed span and
floor clearance. Planned count is used only after measurement for consistency.
Occluded, ambiguous or weak-depth fits remain unknown. No simulator is accepted.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _rz(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def _project(points,camera,shape):
    h,w=shape[:2]
    p=(np.asarray(points)-np.asarray(camera['pos']))@np.asarray(camera['rotation'])
    depth=-p[:,2]
    f=h/2/np.tan(np.deg2rad(camera['fovy'])/2)
    return np.column_stack(((w-1)/2+f*p[:,0]/np.maximum(depth,1e-5),
                            (h-1)/2-f*p[:,1]/np.maximum(depth,1e-5))),depth


def _corners(half):
    return np.asarray([[x,y,z] for x in (-half[0],half[0])
                       for y in (-half[1],half[1]) for z in (-half[2],half[2])])


def _hull(points):
    return cv2.convexHull(np.asarray(points,np.float32)).reshape(-1,2).astype(float)


def _sample_polygon(polygon,n=32):
    polygon=np.asarray(polygon,float)
    delta=np.roll(polygon,-1,axis=0)-polygon
    length=np.linalg.norm(delta,axis=1)
    perimeter=float(length.sum())
    if len(polygon)<3 or perimeter<1e-6:
        return None
    sample=(np.arange(n)+.5)*perimeter/n
    ends=np.cumsum(length)
    edge=np.searchsorted(ends,sample)
    begin=np.r_[0.,ends[:-1]]
    return polygon[edge]+delta[edge]*((sample-begin[edge])/np.maximum(length[edge],1e-6))[:,None]


def _distances(points,polygon):
    a=np.asarray(polygon,float);d=np.roll(a,-1,axis=0)-a
    v=np.asarray(points)[:,None,:]-a[None,:,:]
    u=np.clip(np.sum(v*d[None,:,:],axis=2)/np.maximum(np.sum(d*d,axis=1),1e-6),0,1)
    return np.sqrt(np.min(np.sum((v-u[:,:,None]*d[None,:,:])**2,axis=2),axis=1)+1e-8)


class PlacementAuditor:
    def __init__(self,config,color_mask=None):
        self.config=config
        self.scene=getattr(config,'cookie_transfer',config)
        self.half=np.asarray(self.scene.cookie_half_size_m,float)
        self.thickness=float(2*self.half[1])
        self.bin_half=np.asarray(self.scene.target_bin_half_size_m,float)
        self.wall=float(self.scene.bin_wall_thickness_m)
        self.floor=float(self.scene.target_floor_z_m)+self.wall/2
        self.slots=np.asarray(self.scene.target_slots_local_m,float)
        self.columns=np.unique(self.slots[:,0])
        self.color_mask=color_mask or self._color_mask
        units=[];normals=[]
        for axis in range(3):
            others=[i for i in range(3) if i!=axis]
            for sign in (-1.,1.):
                for a in np.linspace(-1,1,17 if others[0]==2 else 5):
                    for b in np.linspace(-1,1,17 if others[1]==2 else 5):
                        point=np.zeros(3);point[axis]=sign;point[others]=[a,b]
                        normal=np.zeros(3);normal[axis]=sign
                        units.append(point);normals.append(normal)
        self.surface_units=np.asarray(units);self.surface_normals=np.asarray(normals)
        hx,hy=self.bin_half;h=float(self.scene.target_bin_wall_height_m);t=self.wall
        self.occluders=[]
        for x in (-hx,hx):
            self.occluders.append((np.array([x-t/2,-hy-t/2,0.]),np.array([x+t/2,hy+t/2,h])))
        for y in (-hy,hy):
            self.occluders.append((np.array([-hx-t/2,y-t/2,0.]),np.array([hx+t/2,y+t/2,h])))
        self.reset()

    def reset(self,context=None):
        self.history={}
        self.last={}
        self.debug={}
        self.view_history={}
        self.held_reference={}
        self.episode_chrom=None

    def _learn_color(self,observation,cameras,held):
        if self.episode_chrom is not None or not held:
            return
        if (float(held.get('pose_confidence',0.) or 0.)<.6 or held.get('center_tool') is None
                or held.get('rotation_tool') is None or held.get('width_m') is None):
            return
        eef=np.asarray(observation['observation.eef_pose'],float)[:7]
        tool=Rotation.from_quat(np.r_[eef[4:7],eef[3]]).as_matrix()
        center=eef[:3]+tool@np.asarray(held['center_tool'])
        rotation=tool@np.asarray(held['rotation_tool'])
        half=np.array([self.half[0],float(held['width_m'])/2,self.half[2]])
        corners=_corners(half)@rotation.T+center
        values=[]
        for name,camera in cameras.items():
            image=np.asarray(observation['observation.images.'+name],np.uint8)
            uv,z=_project(corners,camera,image.shape)
            if np.any(z<.02):
                continue
            roi=np.zeros(image.shape[:2],np.uint8)
            cv2.fillConvexPoly(roi,np.rint(_hull(uv)).astype(np.int32),255)
            hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
            selected=(roi>0)&(hsv[:,:,1]>70)&(hsv[:,:,2]>40)
            if np.sum(selected)>10:
                values.append(image[selected].astype(float))
        if not values:
            return
        rgb=np.concatenate(values);chrom=rgb/np.maximum(rgb.sum(1,keepdims=True),1.)
        references=np.asarray([[.90,.62,.12],[.38,.18,.08],[.40,.60,.20],[.78,.30,.38]],float)
        references/=references.sum(1,keepdims=True)
        distances=np.linalg.norm(chrom[:,None,:]-references[None,:,:],axis=2)
        labels=np.argmin(distances,axis=1);near=np.min(distances,axis=1)<.075
        score=np.bincount(labels[near],weights=np.sqrt(rgb[near].max(1)),minlength=4)
        if score.sum()>0:
            pixels=chrom[near&(labels==int(np.argmax(score)))]
            if len(pixels)>=20:
                self.episode_chrom=np.median(pixels,axis=0)
                self.last.clear()
                self.view_history.clear()

    @staticmethod
    def _color_mask(image):
        rgb=np.asarray(image,float)
        chrom=rgb/np.maximum(rgb.sum(2,keepdims=True),1.)
        ref=np.asarray([[.90,.62,.12],[.38,.18,.08],[.40,.60,.20],[.78,.30,.38]],float)
        ref/=ref.sum(1,keepdims=True)
        distance=np.min(np.linalg.norm(chrom[:,:,None,:]-ref[None,None,:,:],axis=3),axis=2)
        hsv=cv2.cvtColor(np.asarray(image,np.uint8),cv2.COLOR_RGB2HSV)
        return ((distance<.075)&(hsv[:,:,1]>60)&(hsv[:,:,2]>35)).astype(np.uint8)*255

    def _views(self,observation,cameras,box,request):
        box_position=np.asarray(box['center'],float)
        box_rotation=_rz(float(box.get('yaw',0.)))
        column=int(np.clip(request.get('target_column',0),0,len(self.columns)-1))
        expected=request.get('batch_expected_local_center')
        center=np.asarray(expected if expected is not None else [self.columns[column],0.,self.floor+self.half[2]],float)
        roi_center=np.array([self.columns[column],0.,self.floor+.025])
        roi_half=np.array([self.half[0]+.010,self.bin_half[1]+.012,.035])
        roi_world=(roi_center+_corners(roi_half))@box_rotation.T+box_position
        data=[]
        for name,camera in cameras.items():
            key='observation.images.'+name
            if key not in observation:
                continue
            image=np.asarray(observation[key],np.uint8)
            roi_uv,depth=_project(roi_world,camera,image.shape)
            visible_depth=depth>.025
            if np.sum(visible_depth)<4:
                continue
            roi=np.zeros(image.shape[:2],np.uint8)
            cv2.fillConvexPoly(roi,np.rint(_hull(roi_uv[visible_depth])).astype(np.int32),255)
            colors=self.color_mask(image)
            if self.episode_chrom is not None:
                rgb=image.astype(float);chrom=rgb/np.maximum(rgb.sum(2,keepdims=True),1.)
                hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                colors=((np.linalg.norm(chrom-self.episode_chrom,axis=2)<.055)
                        &(hsv[:,:,1]>60)&(hsv[:,:,2]>35)).astype(np.uint8)*255
            mask=cv2.bitwise_and(colors,roi)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            number,labels,stats,centroids=cv2.connectedComponentsWithStats(mask)
            expected_uv,z=_project((center@box_rotation.T+box_position)[None],camera,image.shape)
            if z[0]<=0:
                continue
            candidates=[]
            for component in range(1,number):
                area=int(stats[component,cv2.CC_STAT_AREA])
                if area<7:
                    continue
                contour=cv2.findContours((labels==component).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
                if not contour:
                    continue
                polygon=_hull(np.vstack(contour).reshape(-1,2))
                if len(polygon)<3:
                    continue
                distance=float(_distances(expected_uv,polygon)[0])
                if cv2.pointPolygonTest(polygon.astype(np.float32),tuple(expected_uv[0]),False)>=0:
                    distance=0.
                # Association uses the requested target region; object size and
                # count are measured independently from the selected pixels.
                score=distance+.025*np.linalg.norm(centroids[component]-expected_uv[0])
                candidates.append((score,component,polygon,area))
            if not candidates:
                continue
            score,component,polygon,area=min(candidates,key=lambda entry:entry[0])
            if score>50:
                continue
            # A component cropped by the image cannot establish a full extent.
            inside=(polygon[:,0]>1)&(polygon[:,0]<image.shape[1]-2)&(polygon[:,1]>1)&(polygon[:,1]<image.shape[0]-2)
            if not np.all(inside):
                continue
            sample=_sample_polygon(polygon)
            if sample is None:
                continue
            data.append(dict(name=name,camera=camera,shape=image.shape,polygon=polygon,
                             sample=sample,area=area,mask=(labels==component).astype(np.uint8)*255))
        return data,center,box_position,box_rotation

    def _visible_hull(self,p,view,box_position,box_rotation):
        rotation=Rotation.from_euler('xyz',p[3:6]).as_matrix()
        half=np.array([self.half[0],p[6]/2,self.half[2]])
        points=(self.surface_units*half)@rotation.T+p[:3]
        normals=self.surface_normals@rotation.T
        camera=(np.asarray(view['camera']['pos'])-box_position)@box_rotation
        front=np.sum(normals*(camera-p[:3]),axis=1)>0
        points=points[front]
        ray=points-camera
        safe=np.where(np.abs(ray)>.0000001,ray,np.where(ray>=0,1e-7,-1e-7))
        occluded=np.zeros(len(points),bool)
        for low,high in self.occluders:
            a=(low-camera)/safe;b=(high-camera)/safe
            enter=np.max(np.minimum(a,b),axis=1);exit=np.min(np.maximum(a,b),axis=1)
            occluded|=(exit>=np.maximum(enter,0.))&(enter<.999)&(exit>0.)
        points=points[~occluded]
        if len(points)<3:
            return None
        uv,z=_project(points@box_rotation.T+box_position,view['camera'],view['shape'])
        uv=uv[z>.015]
        return _hull(uv) if len(uv)>=3 else None

    def _fit(self,views,initial,box_position,box_rotation,previous=None,reference=None):
        low=np.array([initial[0]-.018,-self.bin_half[1]-.018,self.floor-.005,-1.25,-1.25,-.55,.0025])
        high=np.array([initial[0]+.018,self.bin_half[1]+.018,self.floor+.090,1.25,1.25,.55,.096])
        initial=np.array([initial[0],initial[1],max(initial[2],self.floor+self.half[2]),0.,0.,0.,.035])
        if previous is not None:
            initial=np.asarray(previous,float).copy()
        elif reference is not None:
            initial[:3]=(reference['center']-box_position)@box_rotation
            initial[3:6]=Rotation.from_matrix(box_rotation.T@reference['rotation']).as_euler('xyz')
            initial[6]=reference['width']
        initial=np.clip(initial,low+1e-6,high-1e-6)

        def world_corners(p):
            rotation=Rotation.from_euler('xyz',p[3:6]).as_matrix()
            half=np.array([self.half[0],p[6]/2,self.half[2]])
            local=_corners(half)@rotation.T+p[:3]
            return local@box_rotation.T+box_position

        def image_residual(p):
            values=[]
            for view in views:
                predicted=self._visible_hull(p,view,box_position,box_rotation)
                sampled=None if predicted is None else _sample_polygon(predicted)
                if sampled is None:
                    values.extend(np.full(64,30.));continue
                values.extend(_distances(sampled,view['polygon']))
                values.extend(_distances(view['sample'],predicted))
            return np.asarray(values)
        def residual(p):
            values=image_residual(p)
            if reference is not None and reference['count']==1:
                # This width came from an earlier independent RGB held-object
                # measurement, not the plan. A released single slab remains one
                # object while its currently visible contour is associated.
                values=np.r_[values,(p[6]-reference['width'])/.0006]
            return values
        fit=least_squares(residual,initial,bounds=(low,high),max_nfev=45,
                          diff_step=1e-3,loss='soft_l1',f_scale=1.)
        residuals=image_residual(fit.x)
        rmse=float(np.sqrt(np.mean(np.minimum(residuals,8.)**2)))
        sigma=np.sqrt(np.maximum(np.diag(np.linalg.pinv(fit.jac.T@fit.jac)),0.))*max(.75,rmse)
        singular=np.linalg.svd(fit.jac,compute_uv=False)
        if len(singular)<7 or singular[-1]<singular[0]*1e-6:
            sigma[:]=np.inf
        return fit.x,rmse,sigma,world_corners(fit.x)

    def estimate(self,observation,cameras,box,request,held=None):
        unknown=dict(state='unknown',confidence=0.,count=None,count_confidence=0.,count_scope='batch',
                     center=None,rotation=None,width_m=None,target_local_bounds=None,
                     repairable=False,reason='target geometry not observed',
                     evidence_source='RGB multi-view freely fitted group envelope')
        if not box or box.get('center') is None:
            return unknown
        self._learn_color(observation,cameras,held)
        views,initial,box_position,box_rotation=self._views(observation,cameras,box,request)
        key=(int(request.get('batch_index',0)),int(request.get('target_column',0)))
        now=float(observation.get('time',0.))
        held=held or {}
        if (held.get('count') is not None and float(held.get('count_confidence',0.) or 0.)>=.6
                and float(held.get('pose_confidence',0.) or 0.)>=.6
                and held.get('center_tool') is not None and held.get('rotation_tool') is not None):
            eef=np.asarray(observation['observation.eef_pose'],float)[:7]
            tool=Rotation.from_quat(np.r_[eef[4:7],eef[3]]).as_matrix()
            width=held.get('width_m')
            if width is not None:
                self.held_reference[key]=dict(count=int(held['count']),width=float(width),time=now,
                                             center=eef[:3]+tool@np.asarray(held['center_tool']),
                                             rotation=tool@np.asarray(held['rotation_tool']))
        force=np.asarray(observation.get('observation.force',np.zeros(18)),float)
        fingers_free=bool(len(force)>=14 and max(force[12:14])<.12)
        # A moving wrist supplies metric extra views after release. Retain its
        # actual observed contour/camera pair, never a duplicated fixed overview.
        history_views=self.view_history.setdefault(key,[])
        if fingers_free:
            for view in views:
                if view['name']!='front':
                    history_views.append((now,view))
            self.view_history[key]=history_views=[entry for entry in history_views if now-entry[0]<1.25][-8:]
            for timestamp,view in reversed(history_views):
                if now-timestamp<.12:
                    continue
                distinct=all(np.linalg.norm(np.asarray(view['camera']['pos'])-np.asarray(other['camera']['pos']))>.006
                             or Rotation.from_matrix(np.asarray(view['camera']['rotation']).T@np.asarray(other['camera']['rotation'])).magnitude()>.04
                             for other in views)
                if distinct:
                    views.append(dict(view,name=view['name']+'@earlier_released_view'))
                if len(views)>=3:
                    break
        if not views:
            return dict(unknown,reason='no usable colored group contour in target views',views=[])
        previous=self.last.get(key)
        reference=self.held_reference.get(key)
        p,error,sigma,corners=self._fit(views,initial,box_position,box_rotation,previous,reference)
        self.debug=dict(views=views,corners=corners,parameters=p,fit_error_px=error)
        if error>2.4 or not np.all(np.isfinite(p)):
            return dict(unknown,reason='multi-view shape fit is inconsistent or occluded',fit_error_px=error,
                        views=[view['name'] for view in views])
        self.last[key]=p.copy()
        local_rotation=Rotation.from_euler('xyz',p[3:6]).as_matrix()
        rotation=box_rotation@local_rotation
        center=box_position+box_rotation@p[:3]
        half=np.array([self.half[0],p[6]/2,self.half[2]])
        extent=np.abs(local_rotation)@half
        bounds=np.column_stack((p[:3]-extent,p[:3]+extent))
        bottom=float(bounds[2,0]);tilt=float(np.degrees(np.arccos(np.clip(abs(local_rotation[2,2]),0,1))))
        confidence=float(np.clip(np.exp(-error/3.5),0.,1.))
        center_uncertainty=float(np.max(sigma[:3]))
        width_uncertainty=float(sigma[6])
        if center_uncertainty>.008 or width_uncertainty>.008:
            confidence=min(confidence,.45)
        if self.episode_chrom is None:
            confidence=min(confidence,.45)
        measured_count=int(np.clip(np.rint(p[6]/self.thickness),1,15))
        count_error=abs(p[6]-measured_count*self.thickness)
        count_confidence=confidence if count_error<.0018 and width_uncertainty<.0025 else min(.45,confidence)
        observed_count=measured_count if count_confidence>=.6 else None
        # A prior independently observed held count is a consistency check; it
        # never replaces the current RGB envelope measurement.
        held=held or {}
        held_count=held.get('count')
        held_count_conf=float(held.get('count_confidence',0.) or 0.)
        if held_count is not None and held_count_conf>=.6 and measured_count!=int(held_count):
            count_confidence=min(count_confidence,.45)
            observed_count=None
        history=self.history.setdefault(key,[])
        history.append((now,center.copy(),rotation.copy()))
        self.history[key]=history=history[-5:]
        stable=False
        if len(history)>=2:
            before=next((item for item in history if now-item[0]>=.15),None)
            if before is not None:
                angle=float(Rotation.from_matrix(before[2].T@rotation).magnitude())
                stable=np.linalg.norm(center-before[1])<.0025 and angle<np.deg2rad(6.)
        following=held.get('following')
        following_conf=float(held.get('motion_confidence',held.get('pose_confidence',0.)) or 0.)
        inside=bool(np.all(bounds[:2,0]>=-self.bin_half+self.wall/2-.003)
                    and np.all(bounds[:2,1]<=self.bin_half-self.wall/2+.003))
        seated=abs(bottom-self.floor)<.004
        upright=tilt<15.
        state='unknown';reason='waiting for independent depth/count/stability evidence'
        geometric_confident=confidence>=.6 and center_uncertainty<.0045
        if len(views)==1:
            # Known broad-face CAD can determine a single-view pose only when
            # silhouette conditioning, size and count evidence are sufficient.
            geometric_confident=bool(geometric_confident and views[0]['area']>=50
                                    and len(views[0]['polygon'])>=6
                                    and center_uncertainty<.003 and np.max(sigma[3:6])<.14)
        if geometric_confident:
            if following is True and following_conf>=.6 and fingers_free:
                state='invalid';reason='group follows the retracting gripper in RGB'
            elif not inside:
                state='invalid';reason='fitted group envelope crosses the inner target boundary'
            elif tilt>22.:
                state='invalid';reason='fitted group tilt exceeds the upright tolerance'
            elif bottom-self.floor>.008:
                state='invalid';reason='fitted group bottom remains above the support floor'
            elif seated and upright and stable and fingers_free and observed_count is not None:
                state='valid';reason='RGB envelope is contained, upright, seated, stationary, counted and finger contact is clear'
        prior_bounds=[entry.get('bounds') for entry in request.get('placed_bounds_by_batch',[])
            if int(entry.get('target_column',-1))==key[1] and int(entry.get('batch_index',-1))!=key[0]
            and entry.get('bounds') is not None]
        spans_prior=any(np.asarray(previous).shape==(3,2)
            and min(bounds[1,1],np.asarray(previous)[1,1])-max(bounds[1,0],np.asarray(previous)[1,0])>.002
            for previous in prior_bounds)
        count_scope='column' if spans_prior else 'batch'
        if spans_prior and state=='valid':
            state='unknown';reason='column contour includes an earlier group; current batch count association is unresolved'
        result=dict(unknown,state=state,confidence=confidence,count=observed_count,count_scope=count_scope,
                    count_confidence=float(count_confidence),center=center.tolist(),rotation=rotation.tolist(),
                    width_m=float(p[6]),target_local_bounds=bounds.tolist(),
                    repairable=bool(state=='invalid' and geometric_confident and observed_count is not None),
                    reason=reason,fit_error_px=error,
                    center_uncertainty_m=center_uncertainty if np.isfinite(center_uncertainty) else None,
                    width_uncertainty_m=width_uncertainty if np.isfinite(width_uncertainty) else None,
                    tilt_deg=tilt,bottom_clearance_m=bottom-self.floor,
                    upright=upright,seated=seated,inside=inside,stationary=stable,fingers_free=fingers_free,
                    count_basis='RGB fitted packed span divided by fixed CAD thickness; plan count excluded from fit',
                    measured_count_candidate=measured_count,packed_span_residual_m=float(count_error),
                    views=[view['name'] for view in views])
        self.debug=dict(views=views,corners=corners,result=result)
        return result
