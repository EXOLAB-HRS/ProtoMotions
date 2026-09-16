"""v2 fixed anatomical axes; MyoHub/myo_sim eb327ac (Apache-2.0).
Knee coupled translations/rotations are deliberately not transferred.
"""
from pathlib import Path
import json, hashlib, copy
import xml.etree.ElementTree as ET
import numpy as np
from ...common.paths import WORKSPACE_ROOT, PACKAGE_ROOT
ROOT=PACKAGE_ROOT/'human_model_v2'
SOURCE=ROOT/'assets/myoleg_joint_reference.json'
SOURCE_SHA='da7c4a13eec2abfedc683210972ea423d630461a56909bee2884e181e1bf1d97'

def configure_force_profile(profile):
    """Freeze joint geometry; use published sagittal strength surfaces by default.

    Seven men/seven women age18-25: equal average of source cohort curves,
    each with its published weight*height normalization. No individual scaling.
    Other axes retain explicit static proxies, not invented strength surfaces.
    """
    strength=copy.deepcopy(json.loads((PACKAGE_ROOT/'human_model_v1/profiles/candidate_parameters.json').read_text())['strength'])
    strength['status']='candidate; default v2 sagittal force law'
    strength['scope']='Bilateral hip flexion/extension, knee flexion/extension, ankle plantar/dorsiflexion only. Source dynamometer coordinates transferred to oblique joint coordinates as a proxy. Out-of-domain boundary hold is unvalidated.'
    profile['active_strength_model']=strength
    for side in ('L','R'):
        for suffix,pos in [('Hip_x','abduction' if side=='L' else 'adduction'),('Hip_z','external rotation' if side=='L' else 'internal rotation')]:
            row=profile['joints'][side+'_'+suffix]
            row['note']=row['note'].replace('Positive clinical direction: abduction.',f'Positive clinical direction: {pos}.').replace('Positive clinical direction: external rotation;',f'Positive clinical direction: {pos};')
        for suffix in ['Hip_y','Knee_y','Ankle_y']:
            row=profile['joints'][side+'_'+suffix]
            row['evidence']['active']='ANDERSON2007_DYNAMIC'
    profile['sources']['ANDERSON2007_DYNAMIC']={'url':'https://doi.org/10.1016/j.jbiomech.2007.03.022','location':'Eq9 Table2/3 ages18-25','scope':strength['scope']}
    return profile

def build():
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    ref=json.loads(SOURCE.read_text());assert ref['source_sha256']==SOURCE_SHA
    source=ET.Element('reference')
    for name,axis in ref['joint_axes'].items():ET.SubElement(source,'joint',name=name,axis=axis)
    for name,pos in ref['body_positions'].items():ET.SubElement(source,'body',name=name,pos=pos)
    reference={j.get('name'):j for j in source.iter('joint')}
    v1=PACKAGE_ROOT/'human_model_v1';profile=json.loads((v1/'profiles/population_profile_candidate.json').read_text())
    profile['id']='human_model_v2';profile['force_calibration_origin']='v1 population candidate: Kaneda knee, Nguyen wrist damping, pooled toe stiffness; same-cohort calibration, not independent validation';profile['sources']['MYOLEG_AXES']='MyoHub/myo_sim eb327ac '+SOURCE_SHA
    xml=ET.parse(v1/'assets/smpl_humanoid.xml');tree=xml.getroot();bodies={b.get('name'):b for b in tree.iter('body')}
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float);provenance={};offsets={}
    for side,ss in [('L','l'),('R','r')]:
        mapping=[('Hip_y','hip_flexion',-1),('Hip_x','hip_adduction',-1 if side=='L' else 1),('Hip_z','hip_rotation',-1 if side=='L' else 1),('Knee_y','knee_angle',1),('Ankle_y','ankle_angle',-1),('Subtalar_x','subtalar_angle',-1 if side=='L' else 1),('Toe_y','mtp_angle',1)]
        for suffix,src,sign in mapping:
            name=side+'_'+suffix;a=np.fromstring(reference[src+'_'+ss].get('axis'),sep=' ');axis=C@a*sign;axis/=np.linalg.norm(axis)
            provenance[name]=dict(source_joint=src+'_'+ss,source_axis=a.tolist(),axis_local=axis.tolist(),coordinate_sign=sign,source_to_model_rotation=C.tolist(),source_sha256=SOURCE_SHA)
        ankle=bodies[side+'_Ankle'];knee=bodies[side+'_Knee'];toe=bodies[side+'_Toe']
        scale=np.linalg.norm(np.fromstring(toe.get('pos'),sep=' '))/np.linalg.norm(np.fromstring(source.find('.//body[@name="toes_'+ss+'"]').get('pos'),sep=' '))
        delta=C@np.fromstring(source.find('.//body[@name="calcn_'+ss+'"]').get('pos'),sep=' ')*scale;offsets[side]=delta
        talus=ET.Element('body',name=side+'_Talus',pos=ankle.get('pos'));knee.remove(ankle);knee.append(talus);talus.append(ankle)
        ankle.set('pos',' '.join(map(str,delta)));toe.set('pos',' '.join(map(str,np.fromstring(toe.get('pos'),sep=' ')-delta)))
        for g in ankle.findall('geom'):g.set('pos',' '.join(map(str,np.fromstring(g.get('pos'),sep=' ')-delta)))
        for part in ['Hip','Knee','Ankle','Toe']:
            b=bodies[side+'_'+part]
            for j in b.findall('joint'):b.remove(j)
        for suffix,_,_ in mapping:
            name=side+'_'+suffix;b=talus if suffix=='Ankle_y' else ankle if suffix=='Subtalar_x' else bodies[side+'_'+suffix.rsplit('_',1)[0]]
            if suffix=='Subtalar_x':
                row=copy.deepcopy(profile['joints'][side+'_Ankle_x']);row['rom_deg']=[-20,20];row['evidence']['rom']='MYOLEG_AXES';row['note']='MyoLeg ROM; v1 ankle inversion/eversion torque proxy, not measured subtalar strength.';profile['joints'][name]=row
            ET.SubElement(b,'joint',name=name,type='hinge',pos='0 0 0',axis=' '.join(map(str,provenance[name]['axis_local'])),armature='0.02',stiffness='0',damping='0',range=' '.join(map(str,profile['joints'][name]['rom_deg'])),limited='true')
    names=[j.get('name') for j in tree.findall('.//joint') if j.get('type')=='hinge'];profile['joints']={n:profile['joints'][n] for n in names}
    profile['transfer_scope']='Upper body v1; lower force laws transferred as proxies to fixed anatomical axes, not independently calibrated.'
    for j in tree.findall('.//joint'):
        if j.get('name') in profile['joints']:j.set('stiffness','0');j.set('damping','0');j.set('range',' '.join(map(str,profile['joints'][j.get('name')]['rom_deg'])))
    actuator=tree.find('actuator');actuator.clear()
    for n in names:ET.SubElement(actuator,'motor',name=n,joint=n,gear='1',ctrllimited='true',ctrlrange='-5000 5000')
    original=Usd.Stage.Open(str(v1/'assets/smpl_humanoid_healthy_adult_v1.usda'));stage=Usd.Stage.CreateInMemory();stage.GetRootLayer().ImportFromString(original.GetRootLayer().ExportToString())
    prefix='/smpl_humanoid/bodies';jointroot='/smpl_humanoid/joints'
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):stage.RemovePrim(prim.GetPath())
    positions={n:np.array(stage.GetPrimAtPath(prefix+'/'+n).GetAttribute('xformOp:transform').Get().ExtractTranslation()) for n in bodies};mass_audit=[]
    for side,delta in offsets.items():
        name=side+'_Ankle';prim=stage.GetPrimAtPath(prefix+'/'+name);mass=UsdPhysics.MassAPI(prim)
        m=float(mass.GetMassAttr().Get());com=np.array(mass.GetCenterOfMassAttr().Get());I=np.array(mass.GetDiagonalInertiaAttr().Get());mt=.1*m
        talusname=side+'_Talus';positions[talusname]=positions[name].copy()
        talus=UsdGeom.Xform.Define(stage,prefix+'/'+talusname);talus.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*positions[name])))
        UsdPhysics.RigidBodyAPI.Apply(talus.GetPrim());tm=UsdPhysics.MassAPI.Apply(talus.GetPrim());tm.CreateMassAttr(mt);tm.CreateCenterOfMassAttr(Gf.Vec3f(0));tm.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5));tm.CreatePrincipalAxesAttr(Gf.Quatf(1))
        mc=m-mt;cc=m*com/mc;P=lambda v:np.dot(v,v)*np.eye(3)-np.outer(v,v)
        Ic=np.diag(I)+m*P(com)-mc*P(cc)-np.eye(3)*1e-5;eig,R=np.linalg.eigh(Ic)
        if np.linalg.det(R)<0:R[:,0]*=-1
        assert min(eig)>0
        mass.GetMassAttr().Set(mc);mass.GetCenterOfMassAttr().Set(Gf.Vec3f(*(cc-delta)));mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*eig));mass.GetPrincipalAxesAttr().Set(Gf.Quatf(Gf.Matrix3d(*R.T.reshape(-1).tolist()).ExtractRotation().GetQuat()))
        positions[name]+=delta;prim.GetAttribute('xformOp:transform').Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*positions[name])))
        for folder in prim.GetChildren():
            if folder.GetName() in ('collisions','visuals'):UsdGeom.Xformable(folder).AddTranslateOp().Set(Gf.Vec3d(*(-delta)))
        mass_audit.append(dict(side=side,original_mass=m,talus_mass=mt,calcaneus_mass=mc,inertia_min=float(min(eig)),assumption='10% mass split preserving neutral aggregate mass/COM/inertia; not measured talus inertia'))
        ET.SubElement(tree.find('.//body[@name="'+talusname+'"]'),'inertial',pos='0 0 0',mass=str(mt),diaginertia='1e-5 1e-5 1e-5')
        ET.SubElement(bodies[name],'inertial',pos=' '.join(map(str,cc-delta)),mass=str(mc),fullinertia=' '.join(map(str,[Ic[0,0],Ic[1,1],Ic[2,2],Ic[0,1],Ic[0,2],Ic[1,2]])))
    parents={child.get('name'):parent.get('name') for parent in tree.iter('body') for child in parent.findall('body')};rows=[]
    for b in tree.iter('body'):
        name=b.get('name');joints=[j for j in b.findall('joint') if j.get('type')=='hinge']
        if not joints:continue
        chain=[parents[name]]
        for i in range(len(joints)-1):
            fn='_joint_frame_'+name+'_'+str(i);chain.append(fn);positions[fn]=positions[name].copy()
            f=UsdGeom.Xform.Define(stage,prefix+'/'+fn);f.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*positions[fn])))
            UsdPhysics.RigidBodyAPI.Apply(f.GetPrim());ma=UsdPhysics.MassAPI.Apply(f.GetPrim());ma.CreateMassAttr(1e-6);ma.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9));ma.CreateCenterOfMassAttr(Gf.Vec3f(0));ma.CreatePrincipalAxesAttr(Gf.Quatf(1))
        chain.append(name)
        for i,j in enumerate(joints):
            n=j.get('name');axis=np.fromstring(j.get('axis'),sep=' ');axis/=np.linalg.norm(axis);hinge=UsdPhysics.RevoluteJoint.Define(stage,jointroot+'/'+n)
            hinge.CreateAxisAttr('X');hinge.CreateBody0Rel().SetTargets([prefix+'/'+chain[i]]);hinge.CreateBody1Rel().SetTargets([prefix+'/'+chain[i+1]])
            local=positions[name]-positions[chain[i]];quat=Gf.Quatf(Gf.Rotation(Gf.Vec3d(1,0,0),Gf.Vec3d(*axis)).GetQuat())
            hinge.CreateLocalPos0Attr(Gf.Vec3f(*local));hinge.CreateLocalPos1Attr(Gf.Vec3f(0));hinge.CreateLocalRot0Attr(quat);hinge.CreateLocalRot1Attr(quat)
            lo,hi=profile['joints'][n]['rom_deg'];hinge.CreateLowerLimitAttr(lo);hinge.CreateUpperLimitAttr(hi)
            drive=UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular');drive.CreateStiffnessAttr(0);drive.CreateDampingAttr(0);drive.CreateTypeAttr('force')
            rows.append(dict(name=n,parent=chain[i],child=chain[i+1],axis_local=axis.tolist(),offset_parent=local.tolist(),source=provenance.get(n,{'source':'v1 upper-body retained'})))
    configure_force_profile(profile)
    (ROOT/'profiles/healthy_adult_v2.json').write_text(json.dumps(profile,indent=2)+'\n');ET.indent(xml);xml.write(ROOT/'assets/human_model_v2.xml',encoding='unicode');stage.GetRootLayer().Export(str(ROOT/'assets/human_model_v2.usda'))
    (ROOT/'joint_map.json').write_text(json.dumps(dict(model_id='human_model_v2',dof_count=len(rows),lower_dofs=14,joints=rows,mass_split=mass_audit,source_sha256=SOURCE_SHA,neutral_body_positions={k:v.tolist() for k,v in positions.items()},scope='MyoLeg fixed axes on SMPL morphology; no knee coupling or muscle geometry'),indent=2)+'\n')
    return len(rows)
if __name__=='__main__':print('Built v2 DOFs',build())


def validate_assets(profile):
    """Fail closed on profile/USD/coordinate-map mismatch at simulator creation."""
    from pxr import Usd,UsdPhysics,Gf
    mapping=json.loads((ROOT/'joint_map.json').read_text());stage=Usd.Stage.Open(str(ROOT/'assets/human_model_v2.usda'))
    joints={p.GetName():UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    if set(joints)!=set(profile['joints']):raise ValueError('v2 USD/profile joint mismatch')
    maximum_axis=0.;maximum_center=0.
    for row in mapping['joints']:
        name=row['name'];j=joints[name]
        if not np.allclose([j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()],profile['joints'][name]['rom_deg'],atol=1e-5,rtol=0):raise ValueError('v2 ROM mismatch: '+name)
        axis=np.array(Gf.Rotation(Gf.Quatd(j.GetLocalRot0Attr().Get())).TransformDir(Gf.Vec3d(1,0,0)))
        error=float(np.linalg.norm(axis-row['axis_local']));center=float(np.linalg.norm(np.array(j.GetLocalPos0Attr().Get())-row['offset_parent']))
        if error>1e-6 or center>1e-6 or j.GetLocalRot0Attr().Get()!=j.GetLocalRot1Attr().Get():raise ValueError('v2 joint frame mismatch: '+name)
        maximum_axis=max(maximum_axis,error);maximum_center=max(maximum_center,center)
    return dict(joints=len(joints),axis_vector_error_max=maximum_axis,center_error_max_m=maximum_center)
