#!/usr/bin/env python3
"""Versioned, coordinate-preserving CCD audit/materialization for SurfNA.

Never edits the source dataset, caches, surfaces, or training configuration.
Chemical identity/quality decisions do not consume model performance metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment

RDLogger.DisableLog('rdApp.warning')
BASE = Path('/public/home/luoyuxuan/SurfNA_V2')
SNAP = BASE / 'data/na_dataset_v2/snapshot_20260825_rcsb'
FROZEN = BASE / 'reports/na_dataset_v2_frozen_nldock139_20260826'
ALLOWED = {'A','C','G','U','T','DA','DC','DG','DT','DU','ADE','CYT','GUA','THY','URA'}
BOND_TYPES = {'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,
              'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in rows for k in r})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def cif_rows(d, prefix):
    fields=[k for k in d if k.startswith(prefix+'.')]
    if not fields:return []
    vals={k:(d[k] if isinstance(d[k],list) else [d[k]]) for k in fields}
    return [{k.split('.',1)[1]:vals[k][i] for k in fields} for i in range(len(vals[fields[0]]))]


class CCD:
    def __init__(self, root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def fetch(self, code):
        if not re.fullmatch('[A-Z0-9]{1,8}',code):raise ValueError('invalid CCD identifier')
        p=self.root/(code+'.cif')
        if not p.exists():
            url='https://files.rcsb.org/ligands/download/'+code+'.cif'
            with urllib.request.urlopen(url,timeout=30) as r: payload=r.read()
            d=MMCIF2Dict(io.StringIO(payload.decode()))
            identity=d.get('_chem_comp.id',[''])[0]
            if identity!=code:raise ValueError('CCD identity mismatch')
            temp=p.with_suffix('.download');temp.write_bytes(payload);temp.replace(p)
        return p

    @lru_cache(maxsize=1024)
    def definition(self,code):
        path=self.root/(code+'.cif')
        d=MMCIF2Dict(str(path))
        atoms={a['atom_id']:a for a in cif_rows(d,'_chem_comp_atom')}
        aliases={a.get('alt_atom_id'):a['atom_id'] for a in atoms.values() if a.get('alt_atom_id') not in (None,'.','?')}
        bonds=cif_rows(d,'_chem_comp_bond')
        return atoms,bonds,aliases

    def atom_name(self,code,name):
        atoms,_,aliases=self.definition(code)
        if name in atoms:return name
        if name in aliases:return aliases[name]
        raise ValueError('atom absent from CCD: '+code+':'+name)

    @lru_cache(maxsize=1024)
    def template(self,code):
        atoms,bonds,_=self.definition(code)
        rw=Chem.RWMol();indices={}
        for name,a in atoms.items():
            if a['type_symbol'] in {'H','D'}:continue
            aa=Chem.Atom(a['type_symbol'].capitalize());aa.SetFormalCharge(int(a['charge']))
            indices[name]=rw.AddAtom(aa)
        for b in bonds:
            if b['atom_id_1'] in indices and b['atom_id_2'] in indices:
                rw.AddBond(indices[b['atom_id_1']],indices[b['atom_id_2']],BOND_TYPES[b['value_order'].upper()])
        m=rw.GetMol();Chem.SanitizeMol(m)
        return m


def chemistry(m):
    return {'heavy_atoms':m.GetNumHeavyAtoms(),'charge':Chem.GetFormalCharge(m),
            'radicals':sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()),
            'aromatic_atoms':sum(a.GetIsAromatic() for a in m.GetAtoms()),
            'fragments':len(Chem.GetMolFrags(m)),
            'smiles':Chem.MolToSmiles(m)}


def read_mol(path):
    m=Chem.SDMolSupplier(str(path),removeHs=False,sanitize=True)[0]
    if m is None:raise ValueError('cannot parse source SDF')
    return Chem.RemoveAllHs(m)


def mol2_labels(path, mol):
    data=path.read_text().split('@<TRIPOS>ATOM')[1].split('@<TRIPOS>')[0]
    rows=[l.split() for l in data.splitlines() if l.strip()]
    rows=[r for r in rows if r[5].split('.')[0].upper() not in {'H','D'}]
    if len(rows)!=mol.GetNumAtoms():raise ValueError('MOL2/source heavy atom counts differ')
    xyz=np.asarray([[float(x) for x in r[2:5]] for r in rows])
    cost=np.linalg.norm(mol.GetConformer().GetPositions()[:,None,:]-xyz[None,:,:],axis=2)
    for i,a in enumerate(mol.GetAtoms()):
        for j,r in enumerate(rows):
            if a.GetSymbol().upper()!=r[5].split('.')[0].upper():cost[i,j]+=1e6
    ii,jj=linear_sum_assignment(cost)
    if len(ii)!=len(rows) or float(cost[ii,jj].max())>0.001:
        raise ValueError('MOL2/source coordinates or elements do not agree')
    labels=[None]*len(rows)
    for i,j in zip(ii,jj):labels[int(i)]=(rows[j][6],rows[j][7],rows[j][1])
    return labels,float(cost[ii,jj].max())


def source_crosslinks(pdb):
    d=MMCIF2Dict(str(SNAP/'raw_snapshot/cif'/(pdb.lower()+'.cif')))
    result=set()
    for r in cif_rows(d,'_struct_conn'):
        if r.get('conn_type_id') not in {'covale','disulf'}:continue
        pair=[]
        for p in ('ptnr1','ptnr2'):
            pair.append((r[p+'_label_comp_id'],r[p+'_label_atom_id']))
        result.add(frozenset(pair))
    return result


def named_rebuild(old, labels, ccd, crosslinks):
    rw=Chem.RWMol();indices={};resolved=[]
    for i,(group,code,name) in enumerate(labels):
        name=ccd.atom_name(code,name)
        a=ccd.definition(code)[0][name]
        if a['type_symbol'].upper()!=old.GetAtomWithIdx(i).GetSymbol().upper():
            raise ValueError('element mismatch in named mapping')
        key=(group,code,name)
        if key in indices:raise ValueError('duplicate atom name within component instance')
        aa=Chem.Atom(a['type_symbol'].capitalize());aa.SetFormalCharge(int(a['charge']))
        indices[key]=rw.AddAtom(aa);resolved.append(key)
    groups={(g,c) for g,c,_ in resolved}
    for group,code in groups:
        atoms,bonds,_=ccd.definition(code)
        expected={n for n,a in atoms.items() if a['type_symbol'] not in {'H','D'}}
        present={n for g,c,n in resolved if (g,c)==(group,code)}
        missing=expected-present
        if missing:
            if len(groups)==1 or any(atoms[n].get('pdbx_leaving_atom_flag')!='Y' for n in missing):
                raise ValueError('missing non-leaving CCD atoms: '+code+':'+','.join(sorted(missing)))
        for b in bonds:
            k1=(group,code,b['atom_id_1']);k2=(group,code,b['atom_id_2'])
            if k1 in indices and k2 in indices:
                rw.AddBond(indices[k1],indices[k2],BOND_TYPES[b['value_order'].upper()])
    for b in old.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();a1,a2=resolved[i],resolved[j]
        if a1[:2]==a2[:2]:continue
        if frozenset(((a1[1],a1[2]),(a2[1],a2[2]))) not in crosslinks:
            raise ValueError('inter-component bond not supported by deposited struct_conn')
        if b.GetBondType()!=Chem.BondType.SINGLE:
            raise ValueError('non-single inter-component bond needs manual review')
        rw.AddBond(i,j,Chem.BondType.SINGLE)
    new=rw.GetMol();new.AddConformer(Chem.Conformer(old.GetConformer()))
    Chem.SanitizeMol(new)
    Chem.AssignStereochemistryFrom3D(new,replaceExistingTags=True)
    return new,resolved


def bond_map(m):
    return {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):str(b.GetBondType()) for b in m.GetBonds()}


def validate_repair(old,new):
    if [a.GetAtomicNum() for a in old.GetAtoms()]!=[a.GetAtomicNum() for a in new.GetAtoms()]:
        raise ValueError('atom count/order changed')
    xyz=old.GetConformer().GetPositions()
    error=float(np.abs(xyz-new.GetConformer().GetPositions()).max())
    if error>1e-9:raise ValueError('coordinates changed')
    if chemistry(new)['radicals']:raise ValueError('radicals remain after CCD reconstruction')
    if chemistry(new)['fragments']!=chemistry(old)['fragments']:
        raise ValueError('fragment count changed')
    before,after=bond_map(old),bond_map(new)
    # A new heavy-atom connectivity cannot be accepted merely because sanitization succeeds.
    if set(before)!=set(after):raise ValueError('heavy connectivity differs from deposited/prepared ligand')
    return {'coordinate_max_abs_change_A':error,
            'bond_order_changes':sum(before[k]!=after[k] for k in before),
            'formal_charge_atom_changes':sum(a.GetFormalCharge()!=b.GetFormalCharge() for a,b in zip(old.GetAtoms(),new.GetAtoms()))}


def save_record(root,name,old,new,source,receptor,provenance):
    out=root/'records'/name;out.mkdir(parents=True,exist_ok=False)
    target=out/'ligand.sdf'
    new.SetProp('surfna_chemistry_policy','ccd_reference_state_v1_not_physiological_pH')
    Chem.MolToMolFile(new,str(target))
    check=read_mol(target)
    if not np.array_equal(check.GetConformer().GetPositions(),old.GetConformer().GetPositions()):
        raise ValueError('serialized coordinates changed')
    validate_repair(new,check)
    (out/'receptor.pdb').symlink_to(Path(receptor).resolve())
    provenance.update({'source_ligand':str(source),'source_ligand_sha256':digest(source),'output_ligand_sha256':digest(target),
                       'heavy_atom_coordinates_and_order_unchanged':True,'source_receptor':str(receptor)})
    write_json(out/'provenance.json',provenance)
    return str(out)


def prepare_jobs():
    meta={r['record_id']:r for r in read_csv(SNAP/'manifests/dataset_v2_final.csv')}
    excluded=read_csv(FROZEN/'na_strict/excluded_from_trainval.csv')
    disallowed={r['record_id'] for r in excluded if r['reason']=='same_parent_pdb_as_test'}
    curated=[r for r in read_csv(FROZEN/'curation_audit.csv') if r['curation_status']=='eligible_core' and r['record_id'] not in disallowed]
    assert len(curated)==3527
    testnames=(BASE/'data/splits/nldock139_local30_graph_ready/nldock139_graph_ready.txt').read_text().split()
    assert len(testnames)==133
    jobs=[]
    for name in testnames:
        p=BASE/'data/derived_v2/nldock139_g22_local30'/name
        prov=json.loads((p/'nldock139_provenance.json').read_text())
        jobs.append({'name':name,'kind':'test','pdb_id':name.upper(),'source':p/(name+'_ligand.sdf'),
                     'receptor':p/(name+'_protein_processed.pdb'),'mol2':Path(prov['jiang_processed_ligand_mol2']),
                     'provenance':prov,'codes':sorted({x.rsplit('_',2)[0] for x in json.loads(prov['ligand_residues_json'])})})
    for r in curated:
        m=meta[r['record_id']];p=Path(m['record_dir'])
        jobs.append({'name':r['record_id'],'kind':'candidate','pdb_id':r['pdb_id'],'source':p/'ligand.sdf',
                     'receptor':p/'receptor.pdb','codes':[r['comp_id']],'metadata':r})
    return jobs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['test','test_review','candidates'],required=True)
    args=ap.parse_args();root=args.output.resolve()
    if root.parent!=BASE/'data/na_dataset_v3':raise ValueError('output must be a direct child of scoped data/na_dataset_v3')
    root.mkdir(parents=True,exist_ok=True)
    phase_done=root/('CHEMISTRY_'+args.phase.upper()+'.json')
    if phase_done.exists():raise ValueError('phase is immutable; choose a new version to rerun')
    ccd=CCD(root/'ccd_reference')
    jobs=[j for j in prepare_jobs() if (j['kind']=='test')==args.phase.startswith('test')]
    original_test_rows=[]
    if args.phase=='test_review':
        original_test_rows=read_csv(root/'reports/chemistry_test.csv')
        pending={r['name'] for r in original_test_rows if r['status']=='review_required'}
        jobs=[j for j in jobs if j['name'] in pending]
    codes=sorted({c for j in jobs for c in j['codes']})
    downloads=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(ccd.fetch,c):c for c in codes}
        for idx,f in enumerate(as_completed(futures),1):
            code=futures[f]
            try:p=f.result();downloads.append({'code':code,'status':'ok','sha256':digest(p)})
            except Exception as e:downloads.append({'code':code,'status':'error','error':str(e)})
            if idx%25==0:print('CCD',idx,'/',len(codes),flush=True)
    write_csv(root/'reports'/('ccd_download_'+args.phase+'.csv'),downloads)
    failed={r['code'] for r in downloads if r['status']=='error'}
    rows=[]
    for idx,job in enumerate(jobs,1):
        row={'name':job['name'],'pdb_id':job['pdb_id'],'kind':job['kind'],'codes':';'.join(job['codes'])}
        try:
            if failed.intersection(job['codes']):raise ValueError('CCD download unavailable')
            old=read_mol(job['source']);row.update({'before_'+k:v for k,v in chemistry(old).items()})
            if job['kind']=='test':
                try:
                    labels,maperror=mol2_labels(job['mol2'],old)
                    row['mol2_mapping_max_distance_A']=maperror
                    actual_codes={l[1] for l in labels}
                    if actual_codes!=set(job['codes']):raise ValueError('MOL2/provenance component codes differ')
                    links=source_crosslinks(job['pdb_id']) if len({l[0] for l in labels})>1 else set()
                    new,mapping=named_rebuild(old,labels,ccd,links)
                    row['mapping_method']='MOL2_atom_names_and_CCD; deposited_crosslinks_for_composites'
                    map_data=[{'source_heavy_index':i,'component_instance':g,'ccd':c,'ccd_atom':a} for i,(g,c,a) in enumerate(mapping)]
                except Exception as named_error:
                    if args.phase!='test_review' or len(job['codes'])!=1:raise
                    template=ccd.template(job['codes'][0])
                    if template.GetNumAtoms()!=old.GetNumAtoms():raise ValueError(str(named_error)+'; full CCD atom count also differs')
                    new=AllChem.AssignBondOrdersFromTemplate(template,old)
                    for atom in new.GetAtoms():atom.SetNumRadicalElectrons(0)
                    Chem.SanitizeMol(new)
                    Chem.AssignStereochemistryFrom3D(new,replaceExistingTags=True)
                    row['mapping_method']='full_heavy_graph_isomorphism_CCD_template'
                    row['named_mapping_unavailable_reason']=str(named_error)
                    map_data={'heavy_index_order':'unchanged','template_ccd':job['codes'][0]}
            else:
                template=ccd.template(job['codes'][0])
                if template.GetNumAtoms()!=old.GetNumAtoms():raise ValueError('source/CCD heavy counts differ')
                new=AllChem.AssignBondOrdersFromTemplate(template,old)
                Chem.AssignStereochemistryFrom3D(new,replaceExistingTags=True)
                row['mapping_method']='full_heavy_graph_isomorphism_CCD_template'
                map_data={'heavy_index_order':'unchanged','template_ccd':job['codes'][0]}
            row.update(validate_repair(old,new));row.update({'after_'+k:v for k,v in chemistry(new).items()})
            row['record_dir']=save_record(root,job['name'],old,new,job['source'],job['receptor'],
                {'input_kind':job['kind'],'original_provenance':job.get('provenance',{}),'audit':dict(row),'atom_mapping':map_data,
                 'ccd_files':{c:digest(root/'ccd_reference'/(c+'.cif')) for c in job['codes']}})
            row['status']='verified_coordinate_preserving'
        except Exception as e:
            row['status']='review_required';row['reason']=type(e).__name__+': '+str(e)
        if job['kind']=='candidate':
            m=job['metadata']
            for k in ['na_family_080','scaffold_key','selected_exact_pose_key','selected_na_sequence_signature','pocket_exact_fingerprint']:
                row[k]=m.get(k,'')
            row['protein_contact_fraction']=int(m['protein_contacting_ligand_atoms'])/int(m['ligand_heavy_atoms_cif'])
            row['na_contact_fraction']=float(m['na_contact_ligand_fraction'])
            row['original_receptor']=str(job['receptor'])
        rows.append(row)
        if idx%25==0:print('CHEMISTRY',args.phase,idx,'/',len(jobs),dict(Counter(r['status'] for r in rows)),flush=True)
    write_csv(root/'reports'/('chemistry_'+args.phase+'.csv'),rows)
    if args.phase=='test_review':
        revised={r['name']:r for r in rows}
        effective=[revised.get(r['name'],r) for r in original_test_rows]
        write_csv(root/'reports/chemistry_test_effective.csv',effective)
    summary={'phase':args.phase,'total':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
        'before_radical_records':sum(r.get('before_radicals',0)>0 for r in rows),
        'verified_after_radical_records':sum(r.get('after_radicals',0)>0 for r in rows if r['status']=='verified_coordinate_preserving'),
        'review':[{k:r.get(k) for k in ['name','codes','reason']} for r in rows if r['status']!='verified_coordinate_preserving'],
        'policy':'CCD reference formal charge state; not a pH prediction; preserve heavy coordinates and order',
        'training_ready':False,'reason':'new graph caches, context gates, split and scaler compatibility not yet finalized',
        'script_sha256':digest(__file__)}
    write_json(phase_done,summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
