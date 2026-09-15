
from argparse import ArgumentParser,FileType

def parse_train_args():
    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--log_dir', type=str, default='workdir', help='Folder in which to save model and logs')
    parser.add_argument('--wandb_dir', type=str, default='wandb', help='Folder in which to save wandb logs')
    parser.add_argument('--restart_dir', type=str, help='Folder of previous training model from which to restart')
    parser.add_argument('--restart_file', type=str, default=None,
                        help='Checkpoint file to load from restart_dir, or an absolute checkpoint path. Defaults are script-specific.')
    parser.add_argument('--restart_key_filter', type=str, default='all',
                        choices=['all', 'surface'],
                        help='Load all compatible checkpoint weights, or only surface-related weights for transfer.')
    parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')
    parser.add_argument('--data_dir', type=str, default='data/proteindata/', help='Folder containing original structures')
    parser.add_argument('--split_train', type=str, default='data/splits/timesplit_no_lig_cpu().detach()overlap_train', help='Path of file defining the split')
    parser.add_argument('--split_val', type=str, default='data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
    parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')
    parser.add_argument('--test_sigma_intervals', action='store_true', default=False, help='Whether to log loss per noise interval')
    parser.add_argument('--val_inference_freq', type=int, default=5, help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--skip_inference_freq', type=int, default=0, help='skip inference epochs')
    
    parser.add_argument('--train_inference_freq', type=int, default=None, help='Frequency of epochs for which to run expensive inference on train data')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps for inference on val')
    

    parser.add_argument('--num_inference_complexes', type=int, default=100, help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--inference_earlystop_metric', type=str, default='valinf_rmsds_lt2', help='This is the metric that is addionally used when val_inference_freq is not None')
    parser.add_argument('--inference_earlystop_goal', type=str, default='max', help='Whether to maximize or minimize metric')
    parser.add_argument('--wandb', action='store_true', default=False, help='')
    parser.add_argument('--project', type=str, default='SurfDock_train', help='')
    parser.add_argument('--run_name', type=str, default='', help='')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global experiment seed used by Python/NumPy/Torch and DataLoader generators.')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='CUDA optimization parameter for faster training')
    parser.add_argument('--num_dataloader_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='pin_memory arg of dataloader')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate this many MDN mini-batches before an optimizer step. '
                             'This lets single-GPU runs match larger effective batch sizes.')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=20, help='Patience of the LR scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--restart_lr', type=float, default=None, help='If this is not none, the lr of the optimizer will be overwritten with this value when restarting from a checkpoint.')
    parser.add_argument('--w_decay', type=float, default=0.0, help='Weight decay added to loss')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
    parser.add_argument('--use_ema', action='store_true', default=False, help='Whether or not to use ema for the model weights')
    parser.add_argument('--ema_rate', type=float, default=0.999, help='decay rate for the exponential moving average model parameters ')
    #parser.add_argument('--use_receptor_graph', action='store_true', default=False, help='Whether to use the receptor graph')
    # Dataset
    parser.add_argument('--limit_complexes', type=int, default=0, help='If positive, the number of training and validation complexes is capped')
    parser.add_argument('--all_atoms', action='store_true', default=False, help='Whether to use the all atoms model')
    parser.add_argument('--receptor_radius', type=float, default=30, help='Cutoff on distances for receptor edges')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='Maximum number of neighbors for each residue')
    parser.add_argument('--atom_radius', type=float, default=5, help='Cutoff on distances for atom connections')
    parser.add_argument('--atom_max_neighbors', type=int, default=8, help='Maximum number of atom neighbours for receptor')
    parser.add_argument('--matching', action='store_false', default=True, help='matching or not in data processing')
    parser.add_argument('--matching_popsize', type=int, default=20, help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20, help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
    parser.add_argument('--num_conformers', type=int, default=1, help='Number of conformers to match to each ligand')
    parser.add_argument('--esm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')
    parser.add_argument('--lm_embedding_type', type=str, default='auto',
                        choices=['auto', 'esm', 'rnafm', 'custom'],
                        help='Type of receptor language-model embeddings. "auto" keeps legacy ESM behavior when --esm_embeddings_path is set.')
    parser.add_argument('--lm_embedding_dim', type=int, default=0,
                        help='Dimension of receptor LM embeddings. Required for custom embeddings; RNA-FM uses 640.')
    parser.add_argument('--surface_path', type=str, default='data/proteinsurface', help='surface information path')
    parser.add_argument('--surface_feature_dim', type=int, default=4,
                        help='Number of scalar features in data[surface].x. SurfNA 4-feature surfaces use hbond,hphob,charge,si.')
    parser.add_argument('--surface_feature_schema', type=str, default='legacy4',
                        choices=['legacy4', 'v2_shared4', 'v2_full8'],
                        help='Surface feature schema. V2 schemas require one shared training-fitted scaler JSON.')
    parser.add_argument('--surface_scaler_json', type=str, default=None,
                        help='Training-only, domain-balanced Surface-v2 scaler JSON. Required by V2 schemas.')
    parser.add_argument('--max_surface_vertices', type=int, default=0,
                        help='If >0, keep only the ligand-proximal N surface vertices during preprocessing to control memory.')
    # Nucleic-acid specific features (optional, backward-compatible)
    parser.add_argument('--use_nucleic_feat_fusion', action='store_true', default=False,
                        help='If set, fuse `data[receptor].nucleic_feat` into receptor embeddings before Residue->Surface projection')
    parser.add_argument('--nucleic_feat_dim', type=int, default=12,
                        help='Dimensionality of `data[receptor].nucleic_feat` (default: 12)')
    parser.add_argument('--use_ligand_patch_tokenizer', action='store_true', default=False,
                        help='Enable ligand-anchored soft surface-patch tokens (SurfNA V2).')
    parser.add_argument('--patch_temperature', type=float, default=2.5,
                        help='Length scale (Angstrom) for ligand-anchored surface-patch assignments.')
    parser.add_argument('--use_nusurf_fusion', action='store_true', default=False,
                        help='Enable bidirectional base/sugar/phosphate-anchor fusion for nucleic-acid receptors.')
    parser.add_argument('--nusurf_distance_scale', type=float, default=6.0,
                        help='Length scale (Angstrom) for NuSurf surface-anchor attention.')
    parser.add_argument('--nusurf_num_blocks', type=int, default=1,
                        help='Number of sequential NuSurf surface-anchor conditioning blocks. '
                             'Use 2 for the SurfNA V2 task-aligned architecture.')
    # Robust preprocessing
    parser.add_argument('--per_complex_timeout_sec', type=int, default=180,
                        help='If preprocessing a single complex takes longer than this (seconds), skip it. Default: 180s')
    # Adaptive weight transfer strategy
    parser.add_argument('--use_adaptive_transfer', action='store_true', default=False,
                        help='Enable adaptive weight transfer with Adapter layers and scalar gating (default: False)')
    parser.add_argument('--adapter_rank', type=int, default=8,
                        help='Rank of low-rank adapter layers (default: 8, smaller values reduce parameters)')
    parser.add_argument('--adapter_scope', type=str, default='surface',
                        choices=['surface', 'full'],
                        help='Scope of adapter layers: "surface" (only surface layers) or "full" (includes ligand, cross-attention, output heads)')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze surface-related backbone layers completely (default: False). If False, use small LR for surface layers instead.')
    parser.add_argument('--surface_lr_ratio', type=float, default=0.5,
                        help='Learning rate ratio for surface layers when freeze_backbone=False (default: 0.5, i.e., 50%% of base LR)')
    parser.add_argument('--unfreeze_layers', type=str, default='',
                        help='Comma-separated list of layer names to unfreeze initially (for progressive unfreezing). Example: "surface_conv_layers.0,surface_node_embedding"')
    parser.add_argument('--progressive_unfreeze', action='store_true', default=False,
                        help='Enable automatic progressive unfreezing during training (default: False)')
    parser.add_argument('--unfreeze_schedule', type=str, default=None,
                        help='Progressive unfreezing schedule. Format: "epoch1:layer1,layer2;epoch2:layer3". '
                             'Example: "20:surface_conv_layers.0;40:surface_conv_layers.1,lig_to_surface_conv_layers"')
    # Diffusion
    # transformStyle
    parser.add_argument('--transformStyle', type=str, default='diffdock', help='transformStyle',choices=['BERT','diffdock'])
    parser.add_argument('--tr_weight', type=float, default=0.33, help='Weight of translation loss')
    parser.add_argument('--rot_weight', type=float, default=0.33, help='Weight of rotation loss')
    parser.add_argument('--tor_weight', type=float, default=0.33, help='Weight of torsional loss')
    parser.add_argument('--rot_sigma_min', type=float, default=0.1, help='Minimum sigma for rotational component')
    parser.add_argument('--rot_sigma_max', type=float, default=1.65, help='Maximum sigma for rotational component')
    parser.add_argument('--tr_sigma_min', type=float, default=0.1, help='Minimum sigma for translational component')
    parser.add_argument('--tr_sigma_max', type=float, default=30, help='Maximum sigma for translational component')
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314, help='Minimum sigma for torsional component')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14, help='Maximum sigma for torsional component')
    parser.add_argument('--no_torsion', action='store_true', default=False, help='If set only rigid matching')
    # Model
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
    parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=32, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='Embeddings size for the cross distance')
    parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
    parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='Whether to use the dynamic distance cutoff')
    parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
    parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='Type of diffusion time embedding')
    parser.add_argument('--sigma_embed_dim', type=int, default=32, help='Size of the embedding of the diffusion time')
    parser.add_argument('--embedding_scale', type=int, default=1000, help='Parameter of the diffusion time embedding')
    # mdn mode
    # loss terms
    parser.add_argument('--ligand_distance_prediction', action='store_true', default=False, help='can been used in mdn scoring model traing')
    parser.add_argument('--atom_type_prediction', action='store_true', default=False, help='used in mdn scoring model traing')
    parser.add_argument('--bond_type_prediction', action='store_true', default=False, help='used in mdn scoring model traing')
    parser.add_argument('--residue_type_prediction', action='store_true', default=False, help='can been used in mdn scoring model traing')
    parser.add_argument('--mdn_aux_loss_weight', type=float, default=0.001,
                        help='Weight for atom/bond/residue auxiliary losses in MDN training.')
    parser.add_argument('--mdn_dist_threshold_train', type=float, default=7.0, help='mdn_dist_threshold_train')
    parser.add_argument('--mdn_dist_threshold_test', type=float, default=5.0, help='mdn_dist_threshold_test')
    parser.add_argument('--mdn_loss_mode', type=str, default='contact_only',
                        choices=['contact_only', 'weighted_all'],
                        help='MDN distance loss. contact_only is the original SurfScore-style loss; weighted_all keeps far-distance pairs with a small weight.')
    parser.add_argument('--mdn_far_weight', type=float, default=0.05,
                        help='Weight for distance pairs beyond mdn_dist_threshold_train when --mdn_loss_mode weighted_all is used.')
    parser.add_argument('--mdn_score_mode', type=str, default='sum',
                        choices=['sum', 'mean', 'topk_sum', 'topk_mean', 'soft_distance_sum', 'soft_distance_mean'],
                        help='Pose-level MDN confidence aggregation at evaluation time.')
    parser.add_argument('--mdn_score_topk', type=int, default=64,
                        help='Number of ligand-surface pair scores retained for topk MDN score aggregation.')
    parser.add_argument('--mdn_soft_threshold_scale', type=float, default=1.0,
                        help='Scale of the sigmoid distance gate for soft_distance_* MDN score aggregation.')
    parser.add_argument('--mdn_sigma_min', type=float, default=1.1,
                        help='Minimum sigma offset used by the MDN head. Original value is 1.1.')
    parser.add_argument('--mdn_mu_min', type=float, default=1.0,
                        help='Minimum mu offset used by the MDN head. Original value is 1.0.')
    parser.add_argument('--mdn_rerank_mode', type=str, default='native_mdn',
                        choices=['native_mdn', 'decoy_supervised'],
                        help='native_mdn keeps original SurfScore training; decoy_supervised trains a pose reranker from generated decoys.')
    parser.add_argument('--decoy_manifest_train', type=str, default=None,
                        help='TSV manifest of generated training decoy poses for --mdn_rerank_mode decoy_supervised.')
    parser.add_argument('--decoy_manifest_val', type=str, default=None,
                        help='TSV manifest of generated validation decoy poses for --mdn_rerank_mode decoy_supervised.')
    parser.add_argument('--pose_loss_weight', type=float, default=1.0,
                        help='Weight of pose-level BCE loss in decoy-supervised MDN reranking.')
    parser.add_argument('--pairwise_loss_weight', type=float, default=0.5,
                        help='Weight of same-complex pairwise ranking loss in decoy-supervised MDN reranking.')
    parser.add_argument('--mdn_nll_weight', type=float, default=0.2,
                        help='Weight of original MDN distance NLL auxiliary loss in decoy-supervised reranking.')
    parser.add_argument('--pose_label_mode', type=str, default='soft_rmsd', choices=['soft_rmsd'],
                        help='Pose label transform. soft_rmsd uses exp(-rmsd / 2.0).')
    parser.add_argument('--pairwise_margin', type=float, default=0.2,
                        help='Margin for pairwise same-complex ranking loss.')
    # mdn mode
    parser.add_argument('--model_type',  type=str, default='score_model', help='model type',choices=['score_model','mdn_model','pretrain_model','energy_score_model','surface_score_model','surface_pretrain_v2'])
    parser.add_argument('--pretrain_v2_mode', action='store_true', default=False,
                        help='Use contact-field, ligand-patch, anchor-contact and masked-surface reconstruction objectives.')
    parser.add_argument('--pretrain_contact_radius', type=float, default=4.5,
                        help='Angstrom contact definition used by SurfNA V2 pretraining.')
    parser.add_argument('--pretrain_patch_weight', type=float, default=0.5,
                        help='Weight of ligand-anchored patch-contact loss within interaction pretraining.')
    parser.add_argument('--pretrain_anchor_weight', type=float, default=0.5,
                        help='Weight of nucleotide-anchor contact loss within interaction pretraining.')
    parser.add_argument('--pretrain_contrastive_weight', type=float, default=0.25,
                        help='Weight of symmetric surface--ligand in-batch InfoNCE pairing loss.')
    parser.add_argument('--pretrain_contrastive_temperature', type=float, default=0.2,
                        help='Temperature of the V2 surface--ligand InfoNCE pairing loss.')
    parser.add_argument('--pretrain_contrastive_queue_size', type=int, default=256,
                        help='Number of detached prior-batch surface/ligand negatives retained for V2 InfoNCE.')
    parser.add_argument('--pretrain_reconstruction_weight', type=float, default=1.0,
                        help='Weight of masked Surface V2 feature reconstruction loss.')
    parser.add_argument('--task_aligned_aux', action='store_true', default=False,
                        help='Jointly optimize native surface--ligand contact/pairing losses with diffusion score matching.')
    parser.add_argument('--task_aux_weight', type=float, default=0.25,
                        help='Weight of the complete task-aligned interaction auxiliary added to diffusion loss.')
    parser.add_argument('--task_aux_contact_radius', type=float, default=4.5,
                        help='Native-pose contact radius in Angstrom for task-aligned surface and ligand targets.')
    parser.add_argument('--task_aux_ligand_weight', type=float, default=0.5,
                        help='Relative weight of ligand-atom contact prediction in the task-aligned auxiliary.')
    parser.add_argument('--task_aux_anchor_weight', type=float, default=0.5,
                        help='Relative weight of NA anchor contact prediction; it is zero for proteins without anchors.')
    parser.add_argument('--task_aux_contrastive_weight', type=float, default=0.1,
                        help='Relative weight of native surface--ligand in-batch pairing in the task-aligned auxiliary.')
    parser.add_argument('--task_aux_noise_decay', type=float, default=2.0,
                        help='exp(-decay*t) weighting so native contact supervision emphasizes lower-noise denoising states.')
    # ConfidenceCGScoreModelV3
    parser.add_argument('--model_version', type=str, default='version3', help='version of mdn model')
    parser.add_argument('--topN', type=int, default=1, help='topN atoms with the smallest distances with surface node for mdn calculate! ')
    # early_stop_patience 
    parser.add_argument('--mdn_early_stop_patience', type=int, default=30 ,help='early stop epochs for mdn')
    parser.add_argument('--mdn_dropout', type=float, default=0.1, help='dropout rate for mdn')
    parser.add_argument('--n_gaussians', type=int, default=20, help='dropout rate for mdn')
    # tor_sigma_min
    args = parser.parse_args()
    return args
