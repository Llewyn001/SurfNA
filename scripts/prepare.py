#!/usr/bin/env python3
"""Prepare one user-supplied known-site receptor, ligand and full8 surface."""
import argparse, json, pickle, random, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receptor',type=Path,required=True,help='Nucleic-acid pocket PDB in surface coordinate frame')
    p.add_argument('--ligand',type=Path,required=True,help='Single 3D ligand SDF; chemistry and starting conformer')
    p.add_argument('--surface',type=Path,required=True,help='Surface-v2 full8 PLY, raw unnormalized properties')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260903)
    a=p.parse_args()
    for x in (a.receptor,a.ligand,a.surface):
        if not x.is_file():p.error('Input does not exist: '+str(x))
    if a.output.exists():p.error('Use a new preparation output directory')
    from surfna.runtime import activate,generator_config
    activate('generator')
    import numpy as np,torch
    from rdkit import Chem
    from plyfile import PlyData
    from datasets.pdbbind import PDBBind
    from utils.surface_v2_features import feature_names
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    cfg=generator_config()
    surface=PlyData.read(str(a.surface))
    fields=('x','y','z')+feature_names('v2_full8')
    for name in fields:
        if name not in surface['vertex'].data.dtype.names:raise ValueError('Missing PLY property: '+name)
        if not np.isfinite(surface['vertex'][name]).all():raise ValueError('Nonfinite PLY property: '+name)
    if 'face' not in surface or len(surface['face'])==0:raise ValueError('PLY requires mesh faces')
    molecules=list(Chem.SDMolSupplier(str(a.ligand),removeHs=False))
    if len(molecules)!=1 or molecules[0] is None:raise ValueError('Supply exactly one valid ligand in SDF')
    mol=molecules[0]
    if mol.GetNumConformers()!=1 or not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError('Ligand requires one finite 3D conformer')
    if not mol.GetConformer().Is3D():raise ValueError('Use a 3D ligand SDF, not a 2D depiction')
    if not mol.GetNumHeavyAtoms():raise ValueError('Empty ligand')
    a.output.mkdir(parents=True)
    # Use a generic identifier: no identity, reference RMSD or affinity is a model input.
    name='query';inp=a.output/'input'/name;surf=a.output/'surfaces'/name
    inp.mkdir(parents=True);surf.mkdir(parents=True)
    shutil.copyfile(a.receptor,inp/'query_protein_processed.pdb')
    shutil.copyfile(a.receptor,surf/'query_pocket.pdb')
    shutil.copyfile(a.surface,surf/'query.ply')
    with Chem.SDWriter(str(inp/'query_ligand.sdf')) as w:w.write(mol)
    names=a.output/'names.txt';names.write_text(name+'\n')
    dataset=PDBBind(root=str(a.output/'input'),split_path=str(names),
        cache_path=str(a.output/'cache_work'),surface_path=str(a.output/'surfaces'),
        receptor_radius=cfg.receptor_radius,c_alpha_max_neighbors=cfg.c_alpha_max_neighbors,
        matching=False,keep_original=True,remove_hs=True,num_conformers=1,
        num_workers=1,require_ligand=True,all_atoms=False,
        max_surface_vertices=cfg.max_surface_vertices,
        surface_feature_schema=cfg.surface_feature_schema,surface_scaler_json=cfg.surface_scaler_json)
    if len(dataset.complex_graphs)!=1 or len(dataset.rdkit_ligands)!=1:
        raise RuntimeError('Input graph preparation failed; inspect the preceding parser error')
    g=dataset.complex_graphs[0]
    for field in ('nucleic_feat','nucleic_anchor_pos','nucleic_anchor_type','nucleic_anchor_parent_index'):
        if field not in g['receptor']:raise ValueError('Missing NuSurf input: '+field)
    for kind in ('ligand','receptor','surface'):
        if not torch.isfinite(g[kind].pos).all():raise ValueError('Nonfinite '+kind+' coordinates')
    # Warn about common coordinate-frame mistakes instead of translating structures silently.
    dist=torch.cdist(g['ligand'].pos.float(),g['surface'].pos.float()).min().item()
    if dist>15:raise ValueError('Ligand/surface frames differ or the supplied ligand is outside the known site (>15 A)')
    cache=a.output/'cache';cache.mkdir()
    for file in ('heterographs.pkl','rdkit_ligands.pkl'):
        shutil.copyfile(Path(dataset.full_cache_path)/file,cache/file)
    report={'name':name,'atoms':int(g['ligand'].num_nodes),'surface_vertices':int(g['surface'].num_nodes),
            'starting_conformer':'user-supplied; rigid-body pose and torsions randomized before diffusion',
            'matching_to_crystal':False,'seed':a.seed}
    (a.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
