"""
caoduanhua : we should to implemented a parapllel version of evaluate.py for a large dataset
"""

import copy
import csv
import hashlib
import json
import os
import random
import torch
import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import wandb
from biopandas.pdb import PandasPdb
from rdkit import RDLogger
from rdkit.Chem import RemoveHs,AllChem
from datasets.process_mols import write_mol_with_coords, generate_conformer
from torch_geometric.loader import DataLoader
from datasets.pdbbind import PDBBind, read_mol
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, get_symmetry_rmsd, remove_all_hs, read_strings_from_txt, ExponentialMovingAverage
from utils.visualise import PDBFile
from tqdm import tqdm
from loguru import logger
torch.multiprocessing.set_sharing_strategy('file_system')
RDLogger.DisableLog('rdApp.*')
import yaml
cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default=None, help='Checkpoint to use inside the folder')
parser.add_argument('--confidence_model_label', type=str, default='mdn1', help='Label used when saving confidence scores')
parser.add_argument('--confidence_model_dir_2', type=str, default=None, help='Optional second confidence model directory')
parser.add_argument('--confidence_ckpt_2', type=str, default=None, help='Checkpoint for the optional second confidence model')
parser.add_argument('--confidence_model_label_2', type=str, default='mdn2', help='Label for the optional second confidence model')
parser.add_argument('--model_version', type=str, default='version3', help='version of mdn model')
# save docking result or not
parser.add_argument('--save_docking_result', action='store_true', default=True, help='Whether to save docking result')
parser.add_argument(
    '--no_save_docking_result',
    action='store_false',
    dest='save_docking_result',
    help='Write the pose-level metric manifest without materializing SDF files (analysis-only).',
)
# put ligand to pocket center
parser.add_argument('--ligand_to_pocket_center', action='store_true', default=False, help='Whether to put ligand on pocket center')
# use_noise_to_rank
parser.add_argument('--use_noise_to_rank', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')
parser.add_argument('--run_name', type=str, default='test_ns_48_nv_10_layer_62023-06-25_07-54-08_model', help='')
parser.add_argument('--project', type=str, default='ligbind_inf_test_mdn', help='')
parser.add_argument('--surface_path', type=str, default='~/PDBBind_processed_8A_surface/', help='test dataset surface path')
parser.add_argument('--esm_embeddings_path', type=str, default=None, help='test dataset esmbedding path')
parser.add_argument('--out_dir', type=str, default='~/diffScreen/test_workdir/mdn_result_40', help='Where to save results to')
parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel')
parser.add_argument('--cache_path', type=str, default='~/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/cache_PDBBIND_pocket_8A', help='Folder from where to load/restore cached dataset')
parser.add_argument('--data_dir', type=str, default='~/DeepLearningForDock/datasets/equibind_and_diffdock_dataset/PDBBIND/PDBBind_pocket_8A/', help='Folder containing original structures')
parser.add_argument('--split_path', type=str, default='~/DeepLearningForDock/DiffDockForScreen/diffScreen/data/splits/timesplit_test', help='Path of file defining the split')
parser.add_argument('--no_overlap_names_path', type=str, default='~/DeepLearningForDock/DiffDockForScreen/diffScreen/data/splits/timesplit_test_no_rec_overlap', help='Path text file with the folder names in the test set that have no receptor overlap with the train set')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')
parser.add_argument('--wandb', action='store_true', default=False, help='')
parser.add_argument('--wandb_dir', type=str, default='~/diffScreen/test_workdir', help='Folder in which to save wandb logs')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--multi_seed_conformer', action='store_true', default=False, help='Whether to use multi_seed_conformer in inference steps')
parser.add_argument('--sampling_seed', type=int, default=1024, help='Deterministic diffusion sampling seed')
parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Whether to save visualizations')
parser.add_argument('--samples_per_complex', type=int, default=40, help='Number of poses to sample for each complex')
parser.add_argument('--actual_steps', type=int, default=None, help='')
parser.add_argument('--mdn_dist_threshold_test', type=float, default=None, help='mdn_dist_threshold_test')
parser.add_argument('--decoy_manifest_out', type=str, default=None, help='Optional TSV path for generated decoy poses with RMSD labels')
parser.add_argument(
    '--frozen_cache_contract',
    type=str,
    required=True,
    help='Required JSON contract for the immutable validation graph cache and split identities.',
)
parser.add_argument(
    '--subset_names_path',
    type=str,
    default=None,
    help=(
        'Optional exact subset of identities to sample after loading the cache for --split_path. '
        'This preserves the frozen full-split cache contract while allowing resumable array shards.'
    ),
)
parser.add_argument(
    '--intervention_contract',
    type=str,
    required=True,
    help='Frozen JSON contract for the generation-level Surface intervention.',
)
parser.add_argument(
    '--analysis_surface_gate_alpha',
    type=float,
    choices=(0.0, 1.0),
    required=True,
    help='Generation-time Surface residual gate; only the preregistered endpoints 0 or 1 are allowed.',
)
parser.add_argument(
    '--chemistry_condition',
    choices=('true_chemistry', 'joint_shuffled_chemistry'),
    required=True,
    help='Use true surface chemistry or one frozen joint permutation of chemistry channels.',
)
parser.add_argument(
    '--chemistry_shuffle_seed',
    type=int,
    default=None,
    help='Frozen permutation repeat. Required only for joint_shuffled_chemistry.',
)
parser.add_argument(
    '--shuffle_permutation_manifest',
    type=str,
    required=True,
    help='Frozen per-complex joint chemistry permutations.',
)
# force_minimized param
parser.add_argument('--force_optimize', action='store_true', default=False, help='')
args = parser.parse_args()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _deny_cache_rebuild(*_args, **_kwargs):
    raise RuntimeError(
        'Frozen validation cache is missing or mismatched; automatic preprocessing/rebuild is forbidden'
    )


def _load_and_precheck_cache_contract():
    with open(args.frozen_cache_contract) as handle:
        contract = json.load(handle)
    if contract.get('automatic_rebuild_allowed') is not False:
        raise RuntimeError('invalid frozen-cache contract: automatic rebuild must be explicitly forbidden')
    if int(contract.get('complex_count', -1)) != 89:
        raise RuntimeError(f"invalid frozen-cache cohort size: {contract.get('complex_count')}")
    expected_split = os.path.realpath(contract['split_path'])
    observed_split = os.path.realpath(args.split_path)
    if observed_split != expected_split:
        raise RuntimeError(f'frozen split path mismatch: {observed_split} != {expected_split}')
    expected_cache_root = os.path.realpath(contract['cache_root'])
    observed_cache_root = os.path.realpath(args.cache_path)
    if observed_cache_root != expected_cache_root:
        raise RuntimeError(f'frozen cache root mismatch: {observed_cache_root} != {expected_cache_root}')
    required_files = {
        contract['split_path']: contract['split_sha256'],
        contract['heterographs_path']: contract['heterographs_sha256'],
        contract['rdkit_ligands_path']: contract['rdkit_ligands_sha256'],
    }
    for path, expected_sha256 in required_files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'frozen validation artifact is missing: {path}')
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f'frozen validation artifact hash mismatch: {path} '
                f'expected={expected_sha256} observed={observed_sha256}'
            )
    PDBBind.preprocessing = _deny_cache_rebuild
    PDBBind.inference_preprocessing = _deny_cache_rebuild
    return contract


def _load_and_precheck_intervention_contract():
    with open(args.intervention_contract) as handle:
        contract = json.load(handle)
    if contract.get('schema_version') != 'surfna-generation-intervention-supplement-v1':
        raise RuntimeError('unexpected generation-intervention contract schema')
    if contract.get('validation_complexes') != 89:
        raise RuntimeError('generation-intervention contract does not freeze val89')
    if contract.get('samples_per_complex') != 10 or contract.get('inference_steps') != 20:
        raise RuntimeError('generation-intervention contract does not freeze K10/20-step sampling')
    if contract.get('sampling_seed') != 20260826:
        raise RuntimeError('generation-intervention contract has the wrong sampling seed')
    if args.samples_per_complex != 10 or args.inference_steps != 20 or args.sampling_seed != 20260826:
        raise RuntimeError('runtime sampling settings differ from the frozen intervention contract')
    expected_manifest = os.path.realpath(contract['shuffle_permutation_manifest'])
    observed_manifest = os.path.realpath(args.shuffle_permutation_manifest)
    if observed_manifest != expected_manifest:
        raise RuntimeError(f'permutation manifest path mismatch: {observed_manifest} != {expected_manifest}')
    if _sha256_file(observed_manifest) != contract['shuffle_permutation_manifest_sha256']:
        raise RuntimeError('permutation manifest hash mismatch')
    allowed_seeds = tuple(int(value) for value in contract['shuffle_seeds'])
    if args.chemistry_condition == 'true_chemistry':
        if args.chemistry_shuffle_seed is not None:
            raise RuntimeError('true_chemistry must not specify a shuffle seed')
    elif args.chemistry_shuffle_seed not in allowed_seeds:
        raise RuntimeError(
            f'joint_shuffled_chemistry requires one frozen shuffle seed; observed={args.chemistry_shuffle_seed}'
        )
    if args.chemistry_condition == 'joint_shuffled_chemistry' and args.analysis_surface_gate_alpha != 1.0:
        raise RuntimeError('chemistry-shuffle conditions are preregistered only at gate alpha=1')
    return contract


def _apply_generation_intervention(dataset, contract):
    chemistry_channels = tuple(int(value) for value in contract['chemistry_channels'])
    if chemistry_channels != (0, 1, 2, 4, 5, 6):
        raise RuntimeError(f'unexpected frozen chemistry channels: {chemistry_channels}')
    if args.chemistry_condition == 'true_chemistry':
        return
    with open(args.shuffle_permutation_manifest) as handle:
        manifest = json.load(handle)
    if tuple(int(value) for value in manifest['chemistry_channels']) != chemistry_channels:
        raise RuntimeError('chemistry channels differ between contract and permutation manifest')
    seed_key = str(args.chemistry_shuffle_seed)
    observed_names = []
    for graph in dataset.complex_graphs:
        name = _complex_name(graph)
        observed_names.append(name)
        try:
            record = manifest['permutations'][name][seed_key]
        except KeyError as exc:
            raise RuntimeError(f'missing frozen chemistry permutation for {name}/{seed_key}') from exc
        features = graph['surface'].x
        if features.ndim != 2 or features.shape[1] != 8:
            raise RuntimeError(f'expected v2_full8 surface features for {name}, observed {tuple(features.shape)}')
        permutation = torch.as_tensor(record['permutation'], dtype=torch.long, device=features.device)
        if permutation.numel() != features.shape[0] or int(record['num_vertices']) != features.shape[0]:
            raise RuntimeError(f'frozen chemistry permutation length mismatch for {name}')
        if sorted(permutation.detach().cpu().tolist()) != list(range(features.shape[0])):
            raise RuntimeError(f'invalid frozen chemistry permutation for {name}')
        shuffled = features.clone()
        channel_index = torch.as_tensor(chemistry_channels, dtype=torch.long, device=features.device)
        shuffled[:, channel_index] = features[permutation][:, channel_index]
        graph['surface'].x = shuffled
    if len(observed_names) != len(set(observed_names)):
        raise RuntimeError('duplicate complex identity while applying the generation intervention')
    logger.info(
        f'Applied frozen joint chemistry shuffle seed={args.chemistry_shuffle_seed} '
        f'to {len(observed_names)} cached complexes'
    )


def _stable_complex_sampling_seed(base_seed, complex_name):
    token = f'surfna-lb200-task-trajectory-v1|{int(base_seed)}|{complex_name}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little', signed=False)


def _reset_complex_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_single(value):
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else value
    return value


def _complex_name(graph):
    name = getattr(graph, "name", None)
    if name is None:
        name = graph["name"]
    name = _unwrap_single(name)
    if isinstance(name, (list, tuple)):
        name = name[0]
    return str(name)


def _center_numpy(graph):
    center = _unwrap_single(getattr(graph, "original_center"))
    if torch.is_tensor(center):
        center = center.detach().cpu().numpy()
    else:
        center = np.asarray(center)
    return center.reshape(-1, 3)[0].astype(np.float32)


def _center_tensor(graph, *, device=None, dtype=None):
    center = _unwrap_single(getattr(graph, "original_center"))
    if not torch.is_tensor(center):
        center = torch.as_tensor(center)
    if device is not None or dtype is not None:
        center = center.to(device=device, dtype=dtype)
    return center.reshape(-1, 3)[0]


def _confidence_numpy(confidence):
    if confidence is None:
        return None
    if isinstance(confidence, (list, tuple)) and len(confidence) == 0:
        return None
    if torch.is_tensor(confidence):
        confidence = confidence.detach().cpu().numpy()
    else:
        confidence = np.asarray(confidence)
    if confidence.size == 0:
        return None
    if confidence.ndim > 1 and confidence.shape[-1] == 1:
        confidence = confidence.reshape(-1)
    return confidence


def _safe_label(label):
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(label))


def _resolve_receptor_path(data_dir, complex_name):
    """Return the exact canonical receptor frozen for the L2 validation set."""
    receptor = os.path.join(
        data_dir, complex_name, f'{complex_name}_protein_processed.pdb'
    )
    if not os.path.isfile(receptor):
        raise FileNotFoundError(
            f'Canonical L2 receptor PDB is missing for {complex_name}: {receptor}'
        )
    return receptor


def _load_confidence_args(model_dir, score_model_args):
    with open(f'{model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))
    confidence_args.transfer_weights = False
    confidence_args.original_model_dir = None
    confidence_args.mdn_dist_threshold_test = args.mdn_dist_threshold_test if args.mdn_dist_threshold_test is not None else 5.0
    if not hasattr(confidence_args, 'mdn_dist_threshold_train'):
        confidence_args.mdn_dist_threshold_train = 7.0

    confidence_esm_path = getattr(confidence_args, 'esm_embeddings_path', None)
    score_max_surface = int(getattr(score_model_args, 'max_surface_vertices', 0) or 0)
    confidence_max_surface = int(getattr(confidence_args, 'max_surface_vertices', 0) or 0)
    same_lm_graph = bool(args.esm_embeddings_path) == bool(confidence_esm_path)
    same_surface_cap = score_max_surface == confidence_max_surface
    confidence_args.use_original_model_cache = same_lm_graph and same_surface_cap
    if not confidence_args.use_original_model_cache:
        logger.info(
            'HAPPENING | confidence model uses its own cache '
            f'(score_lm={bool(args.esm_embeddings_path)}, confidence_lm={bool(confidence_esm_path)}, '
            f'score_surfmax={score_max_surface}, confidence_surfmax={confidence_max_surface})'
        )
    return confidence_args


def _build_confidence_specs(score_model_args):
    specs = []
    for model_dir, ckpt, label in [
        (args.confidence_model_dir, args.confidence_ckpt, args.confidence_model_label),
        (args.confidence_model_dir_2, args.confidence_ckpt_2, args.confidence_model_label_2),
    ]:
        if model_dir is None:
            continue
        specs.append({
            'label': _safe_label(label),
            'model_dir': model_dir,
            'ckpt': ckpt or 'best_model.pt',
            'args': _load_confidence_args(model_dir, score_model_args),
            'model_args': None,
            'model': None,
            'complex_dict': None,
        })
    return specs


def _score_confidence(data_list, confidence_model, confidence_model_args, confidence_data_list=None):
    with torch.no_grad():
        loader = DataLoader(data_list, batch_size=args.batch_size)
        confidence_loader = iter(DataLoader(confidence_data_list, batch_size=args.batch_size)) \
            if confidence_data_list is not None else None
        confidence = []
        for complex_graph_batch in loader:
            complex_graph_batch = complex_graph_batch.to(device)
            if confidence_loader is not None:
                confidence_complex_graph_batch = next(confidence_loader).to(device)
                confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                b = confidence_complex_graph_batch.num_graphs
                set_time(confidence_complex_graph_batch, 0, 0, 0, b, confidence_model_args.all_atoms, device)
                confidence.append(confidence_model(confidence_complex_graph_batch)[-1])
            else:
                b = complex_graph_batch.num_graphs
                set_time(complex_graph_batch, 0, 0, 0, b, confidence_model_args.all_atoms, device)
                confidence.append(confidence_model(complex_graph_batch)[-1])
        if not confidence:
            return None
        return torch.cat(confidence, dim=0)


def main_function():
    frozen_cache_contract = _load_and_precheck_cache_contract()
    intervention_contract = _load_and_precheck_intervention_contract()
    if accelerator.is_local_main_process:
        if args.wandb:
            wandb.login(key = 'yourkey')
            run = wandb.init(
                entity='SurfDock',
                settings=wandb.Settings(start_method="fork"),
                project=args.project,
                name=args.run_name,
                dir = args.wandb_dir,
                config=args
            )
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    if args.out_dir is None: args.out_dir = f'inference_out_dir_not_specified/{args.run_name}'
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    if args.esm_embeddings_path is None and getattr(score_model_args, 'esm_embeddings_path', None):
        args.esm_embeddings_path = score_model_args.esm_embeddings_path

    confidence_specs = _build_confidence_specs(score_model_args)

    if args.force_optimize:
        logger.info('Using ForceField for energy minimized!')
    test_dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                        receptor_radius=score_model_args.receptor_radius,
                        cache_path=args.cache_path, split_path=args.split_path,
                        remove_hs=score_model_args.remove_hs, max_lig_size=None,
                        c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                        matching=not score_model_args.no_torsion, keep_original=True,
                        popsize=score_model_args.matching_popsize,
                        maxiter=score_model_args.matching_maxiter,
                        all_atoms=score_model_args.all_atoms,
                        atom_radius=score_model_args.atom_radius,
                        atom_max_neighbors=score_model_args.atom_max_neighbors,
                        esm_embeddings_path=args.esm_embeddings_path,
                        require_ligand=True,
                        num_workers=args.num_workers,surface_path = args.surface_path,
                        per_complex_timeout_sec=getattr(score_model_args, 'per_complex_timeout_sec', 180),
                        max_surface_vertices=getattr(score_model_args, 'max_surface_vertices', 0),
                        surface_feature_schema=getattr(score_model_args, 'surface_feature_schema', 'legacy4'),
                        surface_scaler_json=getattr(score_model_args, 'surface_scaler_json', None))

    observed_full_cache_path = os.path.realpath(test_dataset.full_cache_path)
    expected_full_cache_path = os.path.realpath(frozen_cache_contract['full_cache_path'])
    if observed_full_cache_path != expected_full_cache_path:
        raise RuntimeError(
            f'frozen full-cache path mismatch: {observed_full_cache_path} != {expected_full_cache_path}'
        )
    observed_names = [_complex_name(graph) for graph in test_dataset.complex_graphs]
    expected_names = list(frozen_cache_contract['complex_names_in_order'])
    if len(observed_names) != 89 or observed_names != expected_names:
        raise RuntimeError(
            'frozen validation cache identity/order mismatch: '
            f'observed={len(observed_names)} expected={len(expected_names)}'
        )

    if args.subset_names_path is not None:
        subset_names = read_strings_from_txt(args.subset_names_path)
        if not subset_names:
            raise ValueError(f'--subset_names_path is empty: {args.subset_names_path}')
        if len(subset_names) != len(set(subset_names)):
            raise ValueError(f'--subset_names_path contains duplicate identities: {args.subset_names_path}')
        wanted = set(subset_names)
        kept_graphs = []
        kept_ligands = []
        observed = []
        for graph, ligand in zip(test_dataset.complex_graphs, test_dataset.rdkit_ligands):
            name = _complex_name(graph)
            if name in wanted:
                kept_graphs.append(graph)
                kept_ligands.append(ligand)
                observed.append(name)
        missing = sorted(wanted - set(observed))
        unexpected = sorted(set(observed) - wanted)
        if missing or unexpected or len(observed) != len(subset_names):
            raise ValueError(
                '--subset_names_path does not map one-to-one onto the cached graph cohort; '
                f'missing={missing[:10]} unexpected={unexpected[:10]} '
                f'expected={len(subset_names)} observed={len(observed)}'
            )
        # Preserve split-file order, which is also the deterministic sampling order.
        index_by_name = {name: index for index, name in enumerate(observed)}
        ordered_indices = [index_by_name[name] for name in subset_names]
        test_dataset.complex_graphs = [kept_graphs[index] for index in ordered_indices]
        test_dataset.rdkit_ligands = [kept_ligands[index] for index in ordered_indices]
        logger.info(
            f'Using {len(test_dataset.complex_graphs)} cached complexes selected by '
            f'--subset_names_path={args.subset_names_path}'
        )

    _apply_generation_intervention(test_dataset, intervention_contract)

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for spec in confidence_specs:
        confidence_args = spec['args']
        if not (confidence_args.use_original_model_cache or confidence_args.transfer_weights):
            # if the confidence model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            logger.info(f"HAPPENING | confidence model {spec['label']} uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now.")
            confidence_test_dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                                    receptor_radius=confidence_args.receptor_radius,
                                cache_path=args.cache_path, split_path=args.split_path,
                                remove_hs=confidence_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                                matching=not confidence_args.no_torsion, keep_original=True,
                                popsize=confidence_args.matching_popsize,
                                maxiter=confidence_args.matching_maxiter,
                                all_atoms=confidence_args.all_atoms,
                                atom_radius=confidence_args.atom_radius,
                                atom_max_neighbors=confidence_args.atom_max_neighbors,
                                esm_embeddings_path=getattr(confidence_args, 'esm_embeddings_path', None), require_ligand=True,
                                num_workers=args.num_workers,surface_path = args.surface_path,
                                per_complex_timeout_sec=getattr(confidence_args, 'per_complex_timeout_sec', 180),
                                max_surface_vertices=getattr(confidence_args, 'max_surface_vertices', 0),
                                surface_feature_schema=getattr(confidence_args, 'surface_feature_schema', 'legacy4'),
                                surface_scaler_json=getattr(confidence_args, 'surface_scaler_json', None))
            spec['complex_dict'] = {d.name: d for d in confidence_test_dataset}

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    if not args.no_model:
        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,model_type = score_model_args.model_type)
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
        if args.ckpt == 'last_model.pt':
            model_state_dict = state_dict['model']
            ema_weights_state = state_dict['ema_weights']
            model.load_state_dict(model_state_dict, strict=True)
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=score_model_args.ema_rate)
            ema_weights.load_state_dict(ema_weights_state, device=device)
            ema_weights.copy_to(model.parameters())
        else:
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            if not isinstance(state_dict, dict):
                raise TypeError(f'checkpoint is not a state dict: {args.model_dir}/{args.ckpt}')
            state_dict = {
                (key[7:] if key.startswith('module.') else key): value
                for key, value in state_dict.items()
            }
            if len(state_dict) != 463:
                raise RuntimeError(
                    f'checkpoint tensor-count mismatch: observed={len(state_dict)} expected=463'
                )
            incompatible = model.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    'strict score checkpoint load unexpectedly reported incompatible keys: '
                    f'missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}'
                )
            model = model.to(device)
            model.eval()
            logger.info('loaded model weight for score model')
        if not hasattr(model, 'analysis_surface_gate_alpha'):
            raise RuntimeError('analysis Surface-gate overlay was not imported')
        model.analysis_surface_gate_alpha = float(args.analysis_surface_gate_alpha)
        logger.info(
            f'Generation intervention: gate_alpha={args.analysis_surface_gate_alpha} '
            f'chemistry_condition={args.chemistry_condition} '
            f'shuffle_seed={args.chemistry_shuffle_seed}'
        )
        for spec in confidence_specs:
            confidence_args = spec['args']
            if confidence_args.transfer_weights:
                with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
                    spec['model_args'] = Namespace(**yaml.full_load(f))
            else:
                spec['model_args'] = confidence_args

            confidence_model = get_model(spec['model_args'], device, t_to_sigma=t_to_sigma, no_parallel=True,
                                        model_type = spec['model_args'].model_type)
            state_dict = torch.load(f"{spec['model_dir']}/{spec['ckpt']}", map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            spec['model'] = confidence_model.to(device)
            spec['model'].eval()

    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    logger.info('t schedule', tr_schedule)

    rmsds_list, obrmsds, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, names_list = [], [], [], 0, 0, [], [], []
    confidence_labels = [spec['label'] for spec in confidence_specs]
    confidences_by_label_list = {label: [] for label in confidence_labels}
    run_times, min_self_distances_list, without_rec_overlap_list = [], [], []
    N = args.samples_per_complex
    decoy_manifest_rows = []
    names_no_rec_overlap = read_strings_from_txt(args.no_overlap_names_path)
    names_test_all = read_strings_from_txt(args.split_path)
    name_map_idx = {name: idx for idx, name in enumerate(names_test_all)}
    idx_map_name = {idx: name for idx, name in enumerate(names_test_all)}

    logger.info('Size of test dataset: ', len(test_dataset))

    model = accelerator.prepare(model)
    test_loader= accelerator.prepare(test_loader)
    for spec in confidence_specs:
        spec['model'] = accelerator.prepare(spec['model'])

    for idx, orig_complex_graph in tqdm(enumerate(test_loader),total = len(test_loader),disable= not accelerator.is_local_main_process):
        complex_name = _complex_name(orig_complex_graph)
        complex_sampling_seed = _stable_complex_sampling_seed(args.sampling_seed, complex_name)
        _reset_complex_rng(complex_sampling_seed)
        original_center_np = _center_numpy(orig_complex_graph)
        missing_confidence = [
            spec['label'] for spec in confidence_specs
            if spec['complex_dict'] is not None and complex_name not in spec['complex_dict']
        ]
        if missing_confidence:
            skipped += 1
            logger.info(f"HAPPENING | The confidence dataset did not contain {complex_name} for {missing_confidence}. We are skipping this complex.")
            continue
        success = 0
        while not success:
            try:
                success = 1
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                
                if args.multi_seed_conformer:
                    # random multi seed conformers
                    if test_dataset.require_ligand:
                        for data in data_list:
                            mol_rdkit = copy.deepcopy(data.mol[0])
                            mol_rdkit.RemoveAllConformers()
                            mol_rdkit = AllChem.AddHs(mol_rdkit)
                            generate_conformer(mol_rdkit)
                            mol_rdkit = RemoveHs(mol_rdkit, sanitize=True)
                            data.mol = [mol_rdkit]
                randomize_position(data_list, score_model_args.no_torsion, args.no_random, score_model_args.tr_sigma_max,ligand_to_pocket_center = args.ligand_to_pocket_center)
                pdb = None
                if args.save_visualisation:
                    visualization_list = []
                    for idx, graph in enumerate(data_list):
                        # raw pose
                        lig = read_mol(args.data_dir, complex_name, remove_hs=score_model_args.remove_hs)
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        # pose rdkit matching
                        orig_center_tensor = _center_tensor(
                            orig_complex_graph,
                            device=orig_complex_graph['ligand'].pos.device,
                            dtype=orig_complex_graph['ligand'].pos.dtype)
                        pdb.add((orig_complex_graph['ligand'].pos + orig_center_tensor).detach().cpu(), 1, 0)
                        # logger.info(orig_complex_graph['ligand'].pos.shape,orig_complex_graph.original_center.shape)
                        # logger.info(graph['ligand'].pos.device,graph.original_center.device)
                        # random rdkit matching
                        graph_center_tensor = _center_tensor(
                            graph,
                            device=graph['ligand'].pos.device,
                            dtype=graph['ligand'].pos.dtype)
                        pdb.add((graph['ligand'].pos + graph_center_tensor).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None
                rec_path = _resolve_receptor_path(args.data_dir, complex_name)
                logger.info(f"Resolved receptor file path: {rec_path}")
                rec = PandasPdb().read_pdb(rec_path)
                rec_df = rec.df['ATOM']
                receptor_pos = rec_df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(
                    np.float32) - original_center_np
                receptor_pos = np.tile(receptor_pos, (N, 1, 1))
                start_time = time.time()
                confidence_by_label = {}
                if not args.no_model:
                    sampling_result = sampling(input_data_list=data_list, model=model,
                                                    inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                    tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                    tor_schedule=tor_schedule,
                                                    device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                    no_random=args.no_random,
                                                    ode=args.ode, visualization_list=visualization_list,
                                                    confidence_model=None,
                                                    confidence_data_list=None,
                                                    confidence_model_args=None,
                                                    batch_size=args.batch_size,
                                                    no_final_step_noise=args.no_final_step_noise,args = args)
                    if sampling_result is None:
                        raise RuntimeError("sampling returned None after an internal exception")
                    data_list, _ = sampling_result

                for spec in confidence_specs:
                    if spec['complex_dict'] is not None:
                        confidence_data_list = [
                            copy.deepcopy(spec['complex_dict'][complex_name]) for _ in range(N)
                        ]
                    else:
                        confidence_data_list = None
                    confidence_by_label[spec['label']] = _confidence_numpy(
                        _score_confidence(
                            data_list=data_list,
                            confidence_model=spec['model'],
                            confidence_model_args=spec['model_args'],
                            confidence_data_list=confidence_data_list,
                        )
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                confidence = confidence_by_label[confidence_labels[0]] if confidence_labels else None

                run_times.append(time.time() - start_time)
                if score_model_args.no_torsion: orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + original_center_np)
                filterHs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()
                if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                    orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
                ligand_pos = np.asarray(
                    [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])
                orig_ligand_pos = np.expand_dims(
                    orig_complex_graph['ligand'].orig_pos[filterHs],
                    axis=0) # - orig_complex_graph.original_center.cpu().numpy() ,since get_idx have done this function!
                try:
                    mol = remove_all_hs(orig_complex_graph.mol[0])
                    rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[0], [l for l in ligand_pos])
                    rmsd_source_method = 'symmetry_aware'
                except Exception as e:
                    logger.info("Using non corrected RMSD because of the error", e)
                    rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
                    # Keep this source label explicit. The formal pose manifest
                    # will recompute a symmetry-aware RMSD from the written SDF
                    # and must never treat this fallback value as a gold label.
                    rmsd_source_method = 'coordinate_rmsd_fallback'
                rmsds_list.append(rmsd)
                centroid_distance = np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos.mean(axis=1), axis=1)
                # if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                #     confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = np.array(confidence)#.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    # logger.info(confidence, re_order,rmsd)
                    logger.info(
                        f"{complex_name} rmsd {np.around(rmsd, 1)[re_order]} centroid distance {np.around(centroid_distance, 1)[re_order]} confidences {np.around(confidence, 4)[re_order]}"
                                )
                    for label, confidence_values in confidence_by_label.items():
                        if confidence_values is not None:
                            confidences_by_label_list[label].append(np.array(confidence_values))

                else:
                    logger.info(complex_name, ' rmsd', np.around(rmsd, 1), ' centroid distance',
                        np.around(centroid_distance, 1))
                logger.info("316")

                cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                min_cross_distances = np.min(cross_distances, axis=(1, 2))
                self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
                min_self_distances = np.min(self_distances, axis=(1, 2))
                base_cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - orig_ligand_pos[:, None, :, :], axis=-1)
                base_min_cross_distances = np.min(base_cross_distances, axis=(1, 2))
                    
                """ add a save command by caoduanhua to save the last state of ligand"""
                ########################################################################
                if args.save_docking_result:
                    ligand_pos_add_center = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + original_center_np for complex_graph in data_list])
                    lig = orig_complex_graph.mol[0]
                    # save predictions
                    
                    write_dir = f'{args.out_dir}/docking_result_{os.path.basename(args.split_path)}/{complex_name}'
                    os.makedirs(write_dir, exist_ok=True)
                    docking_scores = confidence if confidence is not None else np.full(len(ligand_pos_add_center), np.nan)
                    for sample_idx, (pos,rmsd_i,score) in enumerate(zip(ligand_pos_add_center,rmsd,docking_scores)):
                        mol_pred = copy.deepcopy(lig)
                        # The graph is built from the all-hydrogen-stripped ligand.  RDKit's
                        # default RemoveHs keeps isotopic and degree-zero hydrogens, which
                        # makes the SDF molecule longer than the predicted coordinate array
                        # for a small number of repaired PDB ligands.  Use the same exhaustive
                        # hydrogen policy as the graph/RMSD path so atom indices stay aligned.
                        if score_model_args.remove_hs: mol_pred = remove_all_hs(mol_pred)
                        # if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                        pose_path = os.path.join(write_dir, f'{complex_name}_sample_{sample_idx}_rmsd_{rmsd_i}_confidence_{score}.sdf')
                        write_mol_with_coords(mol_pred, pos, pose_path)
                        if args.decoy_manifest_out is not None:
                            decoy_manifest_rows.append({
                                'complex_name': complex_name,
                                'sample_idx': sample_idx,
                                'pose_sdf': pose_path,
                                'rmsd': float(rmsd_i),
                                'rmsd_source_method': rmsd_source_method,
                                'centroid_distance': float(centroid_distance[sample_idx]),
                                'min_cross_distance': float(min_cross_distances[sample_idx]),
                                'confidence': float(score) if np.isfinite(score) else '',
                                'split_path': args.split_path,
                                'complex_sampling_seed': complex_sampling_seed,
                                'analysis_surface_gate_alpha': float(args.analysis_surface_gate_alpha),
                                'chemistry_condition': args.chemistry_condition,
                                'chemistry_shuffle_seed': '' if args.chemistry_shuffle_seed is None else int(args.chemistry_shuffle_seed),
                            })
                elif args.decoy_manifest_out is not None:
                    docking_scores = confidence if confidence is not None else np.full(len(rmsd), np.nan)
                    for sample_idx, (rmsd_i, score) in enumerate(zip(rmsd, docking_scores)):
                        decoy_manifest_rows.append({
                            'complex_name': complex_name,
                            'sample_idx': sample_idx,
                            'pose_sdf': '',
                            'rmsd': float(rmsd_i),
                            'rmsd_source_method': rmsd_source_method,
                            'centroid_distance': float(centroid_distance[sample_idx]),
                            'min_cross_distance': float(min_cross_distances[sample_idx]),
                            'confidence': float(score) if np.isfinite(score) else '',
                            'split_path': args.split_path,
                            'complex_sampling_seed': complex_sampling_seed,
                            'analysis_surface_gate_alpha': float(args.analysis_surface_gate_alpha),
                            'chemistry_condition': args.chemistry_condition,
                            'chemistry_shuffle_seed': '' if args.chemistry_shuffle_seed is None else int(args.chemistry_shuffle_seed),
                        })
                ########################################################################

                centroid_distances_list.append(centroid_distance)

                min_cross_distances_list.append(min_cross_distances)
                min_self_distances_list.append(min_self_distances)
                base_min_cross_distances_list.append(base_min_cross_distances)

                if args.save_visualisation:
                    write_dir_vis = f'{args.out_dir}/docking_result_{os.path.basename(args.split_path)}/{complex_name}'
                    os.makedirs(write_dir, exist_ok=True)
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            try:
                                visualization_list[batch_idx].write(
                                    f'{write_dir_vis}/{complex_name}_{rank + 1}_{rmsd[batch_idx]:.1f}_{(confidence)[batch_idx]:.1f}.pdb')
                            except:
                                continue
                    else:
                        for rank, batch_idx in enumerate(np.argsort(rmsd)):
                            try:
                                visualization_list[batch_idx].write(
                                    f'{write_dir_vis}/{complex_name}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb')
                            except:
                                continue
                without_rec_overlap_list.append(1 if complex_name in names_no_rec_overlap else 0)
                names_list.append(name_map_idx[complex_name])
            except Exception as e:
                logger.info(f"Failed on {complex_name}: {e}")
                failures += 1
                # Any failure aborts the job. Retrying or skipping would alter the
                # common-random-number stream and break the paired nine-model comparison.
                raise
                
    accelerator.wait_for_everyone()
    if args.decoy_manifest_out is not None:
        if accelerator.is_local_main_process:
            os.makedirs(os.path.dirname(args.decoy_manifest_out), exist_ok=True)
            fieldnames = ['complex_name', 'sample_idx', 'pose_sdf', 'rmsd', 'rmsd_source_method', 'centroid_distance', 'min_cross_distance', 'confidence', 'split_path', 'complex_sampling_seed', 'analysis_surface_gate_alpha', 'chemistry_condition', 'chemistry_shuffle_seed']
            with open(args.decoy_manifest_out, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                writer.writerows(decoy_manifest_rows)
            logger.info(f'Wrote decoy manifest with {len(decoy_manifest_rows)} rows to {args.decoy_manifest_out}')
            logger.info(f'Decoy generation skipped performance summary; failures={failures}, skipped={skipped}')
        return
    rmsds_list, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list =\
         accelerator.gather(torch.tensor(rmsds_list).to(device)), accelerator.gather(torch.tensor(centroid_distances_list).to(device)),accelerator.gather(torch.tensor(failures).to(device)),accelerator.gather(torch.tensor(skipped).to(device)),\
         accelerator.gather(torch.tensor(min_cross_distances_list).to(device)), accelerator.gather(torch.tensor(base_min_cross_distances_list).to(device))
    gathered_confidences_by_label = {}
    for label in confidence_labels:
        gathered_confidences_by_label[label] = accelerator.gather(
            torch.tensor(confidences_by_label_list[label]).to(device)
        )
    run_times, min_self_distances_list, without_rec_overlap_list = accelerator.gather(torch.tensor(run_times).to(device)), accelerator.gather(torch.tensor(min_self_distances_list).to(device)),accelerator.gather(torch.tensor(without_rec_overlap_list).to(device))
    rmsds_list, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list = \
    rmsds_list.cpu().detach().numpy(), centroid_distances_list.cpu().detach().numpy(), failures.cpu().detach().numpy(), skipped.cpu().detach().numpy(), min_cross_distances_list.cpu().detach().numpy(), base_min_cross_distances_list.cpu().detach().numpy()
    gathered_confidences_by_label = {
        label: tensor.cpu().detach().numpy()
        for label, tensor in gathered_confidences_by_label.items()
    }
    run_times, min_self_distances_list, without_rec_overlap_list = \
        run_times.cpu().detach().numpy(), min_self_distances_list.cpu().detach().numpy(), without_rec_overlap_list.cpu().detach().numpy()
    names_list = accelerator.gather(torch.tensor(names_list).to(device))
    names_list = names_list.cpu().detach().numpy()
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:

        logger.info('Performance without hydrogens included in the loss')
        logger.info(f"Failures: {failures}, failures due to exceptions")
        logger.info(f"{skipped},skipped because complex was not in confidence dataset")
        performance_metrics = {}
        all_confidences_by_label = {
            label: np.array(values)
            for label, values in gathered_confidences_by_label.items()
        }
        all_confidences = all_confidences_by_label[confidence_labels[0]] if confidence_labels else np.array([])
        has_confidence_scores = all_confidences.size > 0 and all_confidences.ndim >= 2
        for overlap in ['', 'no_overlap_']:
            if 'no_overlap_' == overlap:
                without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
                if without_rec_overlap.sum() == 0: continue
                rmsds = np.array(rmsds_list)[without_rec_overlap]
                min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
                centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
                # if confidence_model is not None:
                confidences = all_confidences[without_rec_overlap] if has_confidence_scores else None
                confidences_by_label = {
                    label: values[without_rec_overlap]
                    for label, values in all_confidences_by_label.items()
                    if values.size > 0 and values.ndim >= 2
                }
                # else:
                #     confidences = None
                min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
                base_min_cross_distances = np.array(base_min_cross_distances_list)[without_rec_overlap]
                names = np.array(names_list)[without_rec_overlap]
            else:
                rmsds = np.array(rmsds_list)
                min_self_distances = np.array(min_self_distances_list)
                centroid_distances = np.array(centroid_distances_list)
                # if confidence_model is not None:
                confidences = all_confidences if has_confidence_scores else None
                confidences_by_label = {
                    label: values
                    for label, values in all_confidences_by_label.items()
                    if values.size > 0 and values.ndim >= 2
                }
                # else:
                #     confidences = None
                min_cross_distances = np.array(min_cross_distances_list)
                base_min_cross_distances = np.array(base_min_cross_distances_list)
                names = np.array(names_list)
            names = np.array([idx_map_name[idx] for idx in names])

            run_times = np.array(run_times)
            np.save(f'{args.out_dir}/{overlap}min_cross_distances.npy', min_cross_distances)
            np.save(f'{args.out_dir}/{overlap}min_self_distances.npy', min_self_distances)
            np.save(f'{args.out_dir}/{overlap}base_min_cross_distances.npy', base_min_cross_distances)
            np.save(f'{args.out_dir}/{overlap}rmsds.npy', rmsds)
            np.save(f'{args.out_dir}/{overlap}centroid_distances.npy', centroid_distances)
            np.save(f'{args.out_dir}/{overlap}confidences.npy', confidences if confidences is not None else np.array([]))
            for label, label_confidences in confidences_by_label.items():
                np.save(f'{args.out_dir}/{overlap}confidences_{label}.npy', label_confidences)
            np.save(f'{args.out_dir}/{overlap}run_times.npy', run_times)
            np.save(f'{args.out_dir}/{overlap}complex_names.npy', np.array(names))

            # Top-1 Success Rate (first prediction only) - for comparison with benchmark papers
            top1_rmsds = rmsds[:, 0]
            top1_centroid_distances = centroid_distances[:, 0]
            top1_min_cross_distances = min_cross_distances[:, 0]
            top1_min_self_distances = min_self_distances[:, 0]
            
            performance_metrics.update({
                f'{overlap}run_times_std': run_times.std().__round__(2),
                f'{overlap}run_times_mean': run_times.mean().__round__(2),
                f'{overlap}steric_clash_fraction': (
                            100 * (min_cross_distances < 0.4).sum() / len(min_cross_distances) / N).__round__(2),
                f'{overlap}self_intersect_fraction': (
                            100 * (min_self_distances < 0.4).sum() / len(min_self_distances) / N).__round__(2),
                f'{overlap}mean_rmsd': rmsds.mean(),
                f'{overlap}rmsds_below_1': (100 * (rmsds < 1).sum() / len(rmsds) / N),
                f'{overlap}rmsds_below_2': (100 * (rmsds < 2).sum() / len(rmsds) / N),
                f'{overlap}rmsds_below_5': (100 * (rmsds < 5).sum() / len(rmsds) / N),
                f'{overlap}rmsds_percentile_25': np.percentile(rmsds, 25).round(2),
                f'{overlap}rmsds_percentile_50': np.percentile(rmsds, 50).round(2),
                f'{overlap}rmsds_percentile_75': np.percentile(rmsds, 75).round(2),

                f'{overlap}mean_centroid': centroid_distances.mean().__round__(2),
                f'{overlap}centroid_below_2': (100 * (centroid_distances < 2).sum() / len(centroid_distances) / N).__round__(2),
                f'{overlap}centroid_below_5': (100 * (centroid_distances < 5).sum() / len(centroid_distances) / N).__round__(2),
                f'{overlap}centroid_percentile_25': np.percentile(centroid_distances, 25).round(2),
                f'{overlap}centroid_percentile_50': np.percentile(centroid_distances, 50).round(2),
                f'{overlap}centroid_percentile_75': np.percentile(centroid_distances, 75).round(2),
                
                # Top-1 Success Rate (first prediction only)
                f'{overlap}top1_steric_clash_fraction': (
                            100 * (top1_min_cross_distances < 0.4).sum() / len(top1_min_cross_distances)).__round__(2),
                f'{overlap}top1_self_intersect_fraction': (
                            100 * (top1_min_self_distances < 0.4).sum() / len(top1_min_self_distances)).__round__(2),
                f'{overlap}top1_rmsds_below_1': (100 * (top1_rmsds < 1).sum() / len(top1_rmsds)).__round__(2),
                f'{overlap}top1_rmsds_below_2': (100 * (top1_rmsds < 2).sum() / len(top1_rmsds)).__round__(2),
                f'{overlap}top1_rmsds_below_5': (100 * (top1_rmsds < 5).sum() / len(top1_rmsds)).__round__(2),
                f'{overlap}top1_rmsds_percentile_25': np.percentile(top1_rmsds, 25).round(2),
                f'{overlap}top1_rmsds_percentile_50': np.percentile(top1_rmsds, 50).round(2),
                f'{overlap}top1_rmsds_percentile_75': np.percentile(top1_rmsds, 75).round(2),
                f'{overlap}top1_centroid_below_2': (100 * (top1_centroid_distances < 2).sum() / len(top1_centroid_distances)).__round__(2),
                f'{overlap}top1_centroid_below_5': (100 * (top1_centroid_distances < 5).sum() / len(top1_centroid_distances)).__round__(2),
                f'{overlap}top1_centroid_percentile_25': np.percentile(top1_centroid_distances, 25).round(2),
                f'{overlap}top1_centroid_percentile_50': np.percentile(top1_centroid_distances, 50).round(2),
                f'{overlap}top1_centroid_percentile_75': np.percentile(top1_centroid_distances, 75).round(2),
            })

            if N >= 5:
                top5_rmsds = np.min(rmsds[:, :5], axis=1)
                top5_order = np.argsort(rmsds[:, :5], axis=1)
                top5_centroid_distances = np.take_along_axis(centroid_distances[:, :5], top5_order, axis=1)[:, 0]
                top5_min_cross_distances = np.take_along_axis(min_cross_distances[:, :5], top5_order, axis=1)[:, 0]
                top5_min_self_distances = np.take_along_axis(min_self_distances[:, :5], top5_order, axis=1)[:, 0]
                performance_metrics.update({
                    f'{overlap}top5_steric_clash_fraction': (
                                100 * (top5_min_cross_distances < 0.4).sum() / len(top5_min_cross_distances)).__round__(2),
                    f'{overlap}top5_self_intersect_fraction': (
                                100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)).__round__(2),
                    f'{overlap}top5_rmsds_below_1': (100 * (top5_rmsds < 1).sum() / len(top5_rmsds)).__round__(2),
                    f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),
                    f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
                    f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),
                    f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2),
                    f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

                    f'{overlap}top5_centroid_below_2': (
                                100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),
                    f'{overlap}top5_centroid_below_5': (
                                100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
                    f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),
                    f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
                    f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
                })

            if N >= 10:
                top10_rmsds = np.min(rmsds[:, :10], axis=1)
                top10_order = np.argsort(rmsds[:, :10], axis=1)
                top10_centroid_distances = np.take_along_axis(centroid_distances[:, :10], top10_order, axis=1)[:, 0]
                top10_min_cross_distances = np.take_along_axis(min_cross_distances[:, :10], top10_order, axis=1)[:, 0]
                top10_min_self_distances = np.take_along_axis(min_self_distances[:, :10], top10_order, axis=1)[:, 0]
                performance_metrics.update({
                    f'{overlap}top10_steric_clash_fraction': (
                                100 * (top10_min_cross_distances < 0.4).sum() / len(top10_min_cross_distances)).__round__(2),
                    f'{overlap}top10_self_intersect_fraction': (
                                100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)).__round__(2),
                    f'{overlap}top10_rmsds_below_1': (100 * (top10_rmsds < 1).sum() / len(top10_rmsds)).__round__(2),
                    f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
                    f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
                    f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),
                    f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
                    f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),

                    f'{overlap}top10_centroid_below_2': (
                                100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),
                    f'{overlap}top10_centroid_below_5': (
                                100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
                    f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),
                    f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
                    f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
                })

            # if confidence_model is not None:
            if confidences is not None:
                confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]

                filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
                filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
                filtered_min_cross_distances = min_cross_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:,
                                            0]
                filtered_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]
                performance_metrics.update({
                    f'{overlap}filtered_self_intersect_fraction': (
                                100 * (filtered_min_self_distances < 0.4).sum() / len(filtered_min_self_distances)).__round__(
                        2),
                    f'{overlap}filtered_steric_clash_fraction': (
                                100 * (filtered_min_cross_distances < 0.4).sum() / len(filtered_min_cross_distances)).__round__(
                        2),
                    f'{overlap}filtered_rmsds_below_1': (100 * (filtered_rmsds < 1).sum() / len(filtered_rmsds)).__round__(2),
                    f'{overlap}filtered_rmsds_below_2': (100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)).__round__(2),
                    f'{overlap}filtered_rmsds_below_5': (100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)).__round__(2),
                    f'{overlap}filtered_rmsds_percentile_25': np.percentile(filtered_rmsds, 25).round(2),
                    f'{overlap}filtered_rmsds_percentile_50': np.percentile(filtered_rmsds, 50).round(2),
                    f'{overlap}filtered_rmsds_percentile_75': np.percentile(filtered_rmsds, 75).round(2),

                    f'{overlap}filtered_centroid_below_2': (
                                100 * (filtered_centroid_distances < 2).sum() / len(filtered_centroid_distances)).__round__(2),
                    f'{overlap}filtered_centroid_below_5': (
                                100 * (filtered_centroid_distances < 5).sum() / len(filtered_centroid_distances)).__round__(2),
                    f'{overlap}filtered_centroid_percentile_25': np.percentile(filtered_centroid_distances, 25).round(2),
                    f'{overlap}filtered_centroid_percentile_50': np.percentile(filtered_centroid_distances, 50).round(2),
                    f'{overlap}filtered_centroid_percentile_75': np.percentile(filtered_centroid_distances, 75).round(2),
                })

                if N >= 5:
                    top5_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)
                    top5_filtered_centroid_distances = \
                    centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                    top5_filtered_min_cross_distances = \
                    min_cross_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                    top5_filtered_min_self_distances = \
                    min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]
                    performance_metrics.update({
                        f'{overlap}top5_filtered_self_intersect_fraction': (
                                    100 * (top5_filtered_min_cross_distances < 0.4).sum() / len(
                                top5_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top5_filtered_steric_clash_fraction': (
                                    100 * (top5_filtered_min_cross_distances < 0.4).sum() / len(
                                top5_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_below_1': (
                                    100 * (top5_filtered_rmsds < 1).sum() / len(top5_filtered_rmsds)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_below_2': (
                                    100 * (top5_filtered_rmsds < 2).sum() / len(top5_filtered_rmsds)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_below_5': (
                                    100 * (top5_filtered_rmsds < 5).sum() / len(top5_filtered_rmsds)).__round__(2),
                        f'{overlap}top5_filtered_rmsds_percentile_25': np.percentile(top5_filtered_rmsds, 25).round(2),
                        f'{overlap}top5_filtered_rmsds_percentile_50': np.percentile(top5_filtered_rmsds, 50).round(2),
                        f'{overlap}top5_filtered_rmsds_percentile_75': np.percentile(top5_filtered_rmsds, 75).round(2),

                        f'{overlap}top5_filtered_centroid_below_2': (100 * (top5_filtered_centroid_distances < 2).sum() / len(
                            top5_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top5_filtered_centroid_below_5': (100 * (top5_filtered_centroid_distances < 5).sum() / len(
                            top5_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top5_filtered_centroid_percentile_25': np.percentile(top5_filtered_centroid_distances,
                                                                                        25).round(2),
                        f'{overlap}top5_filtered_centroid_percentile_50': np.percentile(top5_filtered_centroid_distances,
                                                                                        50).round(2),
                        f'{overlap}top5_filtered_centroid_percentile_75': np.percentile(top5_filtered_centroid_distances,
                                                                                        75).round(2),
                    })
                if N >= 10:
                    top10_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10],
                                                axis=1)
                    top10_filtered_centroid_distances = \
                    centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                    top10_filtered_min_cross_distances = \
                    min_cross_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                    top10_filtered_min_self_distances = \
                    min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                        np.arange(rmsds.shape[0])[:, None], np.argsort(
                            rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]
                    performance_metrics.update({
                        f'{overlap}top10_filtered_self_intersect_fraction': (
                                    100 * (top10_filtered_min_cross_distances < 0.4).sum() / len(
                                top10_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top10_filtered_steric_clash_fraction': (
                                    100 * (top10_filtered_min_cross_distances < 0.4).sum() / len(
                                top10_filtered_min_cross_distances)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_below_1': (
                                    100 * (top10_filtered_rmsds < 1).sum() / len(top10_filtered_rmsds)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_below_2': (
                                    100 * (top10_filtered_rmsds < 2).sum() / len(top10_filtered_rmsds)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_below_5': (
                                    100 * (top10_filtered_rmsds < 5).sum() / len(top10_filtered_rmsds)).__round__(2),
                        f'{overlap}top10_filtered_rmsds_percentile_25': np.percentile(top10_filtered_rmsds, 25).round(2),
                        f'{overlap}top10_filtered_rmsds_percentile_50': np.percentile(top10_filtered_rmsds, 50).round(2),
                        f'{overlap}top10_filtered_rmsds_percentile_75': np.percentile(top10_filtered_rmsds, 75).round(2),

                        f'{overlap}top10_filtered_centroid_below_2': (100 * (top10_filtered_centroid_distances < 2).sum() / len(
                            top10_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top10_filtered_centroid_below_5': (100 * (top10_filtered_centroid_distances < 5).sum() / len(
                            top10_filtered_centroid_distances)).__round__(2),
                        f'{overlap}top10_filtered_centroid_percentile_25': np.percentile(top10_filtered_centroid_distances,
                                                                                        25).round(2),
                        f'{overlap}top10_filtered_centroid_percentile_50': np.percentile(top10_filtered_centroid_distances,
                                                                                        50).round(2),
                        f'{overlap}top10_filtered_centroid_percentile_75': np.percentile(top10_filtered_centroid_distances,
                                                                                        75).round(2),
                    })

        for k in performance_metrics:
            logger.info(f"{k}: {performance_metrics[k]}")

        if args.wandb:
            wandb.log(performance_metrics)
            histogram_metrics_list = [('top1_rmsd', top1_rmsds),
                                    ('top1_centroid_distance', top1_centroid_distances),
                                    ('mean_rmsd', rmsds.mean(axis=1)),
                                    ('mean_centroid_distance', centroid_distances.mean(axis=1))]
            if N >= 5:
                histogram_metrics_list.append(('top5_rmsds', top5_rmsds))
                histogram_metrics_list.append(('top5_centroid_distances', top5_centroid_distances))
            if N >= 10:
                histogram_metrics_list.append(('top10_rmsds', top10_rmsds))
                histogram_metrics_list.append(('top10_centroid_distances', top10_centroid_distances))
            # if confidence_model is not None:
            if confidences is not None:
                histogram_metrics_list.append(('filtered_rmsd', filtered_rmsds))
                histogram_metrics_list.append(('filtered_centroid_distance', filtered_centroid_distances))
                if N >= 5:
                    histogram_metrics_list.append(('top5_filtered_rmsds', top5_filtered_rmsds))
                    histogram_metrics_list.append(('top5_filtered_centroid_distances', top5_filtered_centroid_distances))
                if N >= 10:
                    histogram_metrics_list.append(('top10_filtered_rmsds', top10_filtered_rmsds))
                    histogram_metrics_list.append(('top10_filtered_centroid_distances', top10_filtered_centroid_distances))
        if args.decoy_manifest_out is not None:
            manifest_path = args.decoy_manifest_out
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            fieldnames = ['complex_name', 'sample_idx', 'pose_sdf', 'rmsd', 'centroid_distance', 'min_cross_distance', 'confidence', 'split_path']
            with open(manifest_path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='	')
                writer.writeheader()
                writer.writerows(decoy_manifest_rows)
            logger.info(f'Wrote decoy manifest with {len(decoy_manifest_rows)} rows to {manifest_path}')

        if args.wandb:
            wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    from accelerate.utils import set_seed
    device = accelerator.device
    set_seed(args.sampling_seed)
    logger.info(f'device {str(accelerator.device)} is used!')
    main_function()
    # sys.exit()
