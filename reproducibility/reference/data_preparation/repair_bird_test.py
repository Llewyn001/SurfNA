"""Coordinate-preserving BIRD composite reconstruction for two fixed test entries."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from build_target_domain_ccd import (BASE,CCD,BOND_TYPES,cif_rows,read_csv,write_csv,
    write_json,read_mol,chemistry,validate_repair,save_record,digest)


def reconstruct(old,entry,bird,ccd):
    chain='C'
    atoms=[a for a in cif_rows(entry,'_atom_site') if a['label_asym_id']==chain
           and a['pdbx_PDB_model_num']=='1' and a['type_symbol'] not in {'H','D'}]
    if len(atoms)!=old.GetNumAtoms():raise ValueError('deposited/source heavy atom counts differ')
    xyz=np.asarray([[float(a['Cartn_'+x]) for x in ('x','y','z')] for a in atoms])
    cost=np.linalg.norm(old.GetConformer().GetPositions()[:,None,:]-xyz[None,:,:],axis=2)
    for i,a in enumerate(old.GetAtoms()):
        for j,b in enumerate(atoms):
            if a.GetSymbol().upper()!=b['type_symbol'].upper():cost[i,j]+=1e6
    ii,jj=linear_sum_assignment(cost)
    if float(cost[ii,jj].max())>.001:raise ValueError('source/deposited atom mapping mismatch')
    ordered=[None]*len(atoms)
    for i,j in zip(ii,jj):ordered[i]=atoms[j]
    seq={r['num']:r['mon_id'] for r in cif_rows(bird,'_pdbx_reference_entity_poly_seq')}
    if {(a['label_seq_id'],a['label_comp_id']) for a in atoms}!=set(seq.items()):
        raise ValueError('BIRD residue sequence mismatch')
    for code in set(seq.values()):ccd.fetch(code)
    rw=Chem.RWMol();indices={};mapping=[]
    for i,a in enumerate(ordered):
        code=a['label_comp_id'];name=ccd.atom_name(code,a['label_atom_id'])
        key=(a['label_seq_id'],code,name)
        if key in indices:raise ValueError('duplicate atom label')
        definition=ccd.definition(code)[0][name]
        if definition['type_symbol'].upper()!=a['type_symbol'].upper():raise ValueError('element mismatch')
        atom=Chem.Atom(a['type_symbol'].capitalize());atom.SetFormalCharge(int(definition['charge']))
        indices[key]=rw.AddAtom(atom);mapping.append(key)
    for pos,code in seq.items():
        aa,bb,_=ccd.definition(code)
        present={name for p,c,name in indices if p==pos and c==code}
        expected={name for name,a in aa.items() if a['type_symbol'] not in {'H','D'}}
        if any(aa[name].get('pdbx_leaving_atom_flag')!='Y' for name in expected-present):
            raise ValueError('non-leaving heavy atom missing from BIRD component '+pos+':'+code)
        for b in bb:
            k1=(pos,code,b['atom_id_1']);k2=(pos,code,b['atom_id_2'])
            if k1 in indices and k2 in indices:rw.AddBond(indices[k1],indices[k2],BOND_TYPES[b['value_order'].upper()])
    deposited=set()
    for r in cif_rows(entry,'_struct_conn'):
        if r['conn_type_id']!='covale':continue
        if any(r.get(p+'_label_asym_id')!=chain for p in ('ptnr1','ptnr2')):continue
        deposited.add(frozenset((r[p+'_label_seq_id'],r[p+'_label_comp_id'],r[p+'_label_atom_id']) for p in ('ptnr1','ptnr2')))
    for b in cif_rows(bird,'_pdbx_reference_entity_poly_link'):
        k1=(b['entity_seq_num_1'],b['comp_id_1'],b['atom_id_1'])
        k2=(b['entity_seq_num_2'],b['comp_id_2'],b['atom_id_2'])
        if frozenset((k1,k2)) not in deposited:raise ValueError('BIRD link not confirmed by deposited struct_conn')
        rw.AddBond(indices[k1],indices[k2],BOND_TYPES[b['value_order'].upper()])
    new=rw.GetMol();new.AddConformer(Chem.Conformer(old.GetConformer()))
    Chem.SanitizeMol(new);Chem.AssignStereochemistryFrom3D(new,replaceExistingTags=True)
    # Heavy composition explicitly checked against the BIRD molecular formula.
    reference=cif_rows(bird,'_pdbx_reference_molecule')[0]
    import re
    expected={e:int(n or 1) for e,n in re.findall(r'([A-Z][a-z]?)(\d*)',reference['formula']) if e!='H'}
    if dict(Counter(a.GetSymbol() for a in new.GetAtoms()))!=expected:raise ValueError('BIRD heavy formula mismatch')
    return new,mapping,float(cost[ii,jj].max())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve();assert root.parent==BASE/'data/na_dataset_v3'
    assert not (root/'CHEMISTRY_TEST_BIRD_V2.json').exists()
    rows=read_csv(root/'reports/chemistry_test_effective.csv');results=[]
    ccd=CCD(root/'bird_reference/components')
    for name,prd in [('2d55','PRD_000001'),('316d','PRD_000006')]:
        r=dict(next(r for r in rows if r['name']==name))
        try:
            original=BASE/'data/derived_v2/nldock139_g22_local30'/name
            source=original/(name+'_ligand.sdf');old=read_mol(source)
            ep=root/'bird_reference'/(name.upper()+'.cif');bp=root/'bird_reference'/(prd+'.cif')
            entry=MMCIF2Dict(str(ep));bird=MMCIF2Dict(str(bp))
            assert any(x['asym_id']=='C' and x['prd_id']==prd for x in cif_rows(entry,'_pdbx_molecule'))
            new,mapping,error=reconstruct(old,entry,bird,ccd)
            metrics=validate_repair(old,new)
            provenance={'method':'BIRD_sequence_links_and_CCD_atom_templates','BIRD_ID':prd,
                'entry_sha256':digest(ep),'BIRD_sha256':digest(bp),'atom_mapping':mapping,
                'template_coordinate_mapping_error_A':error,**metrics,
                'CCD_component_sha256':{code:digest(ccd.root/(code+'.cif')) for _,code,_ in mapping}}
            target=save_record(root,name,old,new,source,original/(name+'_protein_processed.pdb'),provenance)
            r.update({'status':'verified_coordinate_preserving','error':'','reason':'','record_dir':target,
                      'mapping_method':provenance['method'],**metrics})
            r.update({'after_'+k:v for k,v in chemistry(new).items()})
        except Exception as exc:r.update({'status':'review_required','error':str(exc)})
        results.append(r)
    write_csv(root/'reports/chemistry_test_bird_v2.csv',results)
    write_json(root/'CHEMISTRY_TEST_BIRD_V2.json',{'counts':dict(Counter(r['status'] for r in results)),
        'results':results,'test_membership_unchanged':True})
    print([(r['name'],r['status'],r.get('error')) for r in results])

if __name__=='__main__':main()
