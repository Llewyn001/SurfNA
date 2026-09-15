import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm
from e3nn import o3
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims
from loguru import logger
from models.surfna_v2_modules import LigandAnchoredPatchTokenizer, NuSurfAnchorFusion
try:
    from models.adapter_layers import ScalarGatedAdapter, LinearAdapter
except ImportError:
    ScalarGatedAdapter = None
    LinearAdapter = None


# to use a graph to make massage passing ,between surface and residue,and then just use surface to update!
# compare with version 2 , this version use more layers to update surface nodes

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None, lm_embedding_dim=None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            elif self.lm_embedding_type == 'rnafm':
                self.lm_embedding_dim = 640
            elif self.lm_embedding_type == 'custom':
                self.lm_embedding_dim = int(lm_embedding_dim or 0)
                if self.lm_embedding_dim <= 0:
                    raise ValueError('custom LM embeddings require --lm_embedding_dim > 0')
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)
    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            scalar_x = x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features].float()
            scalar_x = torch.nan_to_num(scalar_x, nan=0.0, posinf=0.0, neginf=0.0)
            x_embedding += self.linear(scalar_x)
        if self.lm_embedding_type is not None:
            lm_x = x[:, -self.lm_embedding_dim:].float()
            lm_x = torch.nan_to_num(lm_x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50.0, 50.0)
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, lm_x], axis=1))
        return x_embedding


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:

            out = self.batch_norm(out)
        return out


class TensorProductScoreModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=10,in_rec_edge_features = 5, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False, num_confidence_outputs=1, args=None):
        super(TensorProductScoreModel, self).__init__()
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.timestep_emb_func = timestep_emb_func
        self.confidence_mode = confidence_mode
        self.pretrain_v2_mode = getattr(args, 'pretrain_v2_mode', False) if args is not None else False
        self.pretrain_contact_radius = getattr(args, 'pretrain_contact_radius', 4.5) if args is not None else 4.5
        self.pretrain_patch_weight = getattr(args, 'pretrain_patch_weight', 0.5) if args is not None else 0.5
        self.pretrain_anchor_weight = getattr(args, 'pretrain_anchor_weight', 0.5) if args is not None else 0.5
        self.pretrain_contrastive_weight = getattr(args, 'pretrain_contrastive_weight', 0.25) if args is not None else 0.25
        self.pretrain_contrastive_temperature = getattr(args, 'pretrain_contrastive_temperature', 0.2) if args is not None else 0.2
        self.pretrain_contrastive_queue_size = max(0, int(getattr(args, 'pretrain_contrastive_queue_size', 256) if args is not None else 256))
        self.pretrain_reconstruction_weight = getattr(args, 'pretrain_reconstruction_weight', 1.0) if args is not None else 1.0
        self.task_aligned_aux = getattr(args, 'task_aligned_aux', False) if args is not None else False
        self.task_aux_weight = getattr(args, 'task_aux_weight', 0.25) if args is not None else 0.25
        self.task_aux_contact_radius = getattr(args, 'task_aux_contact_radius', 4.5) if args is not None else 4.5
        self.task_aux_ligand_weight = getattr(args, 'task_aux_ligand_weight', 0.5) if args is not None else 0.5
        self.task_aux_anchor_weight = getattr(args, 'task_aux_anchor_weight', 0.5) if args is not None else 0.5
        self.task_aux_contrastive_weight = getattr(args, 'task_aux_contrastive_weight', 0.1) if args is not None else 0.1
        self.task_aux_noise_decay = getattr(args, 'task_aux_noise_decay', 2.0) if args is not None else 2.0
        # Ephemeral diagnostics populated by the latest V2 pretraining forward.
        # This is deliberately not a parameter or buffer, so checkpoint shape
        # and transfer compatibility remain unchanged.
        self.last_pretrain_metrics = {}
        self.last_task_aux_loss = None
        self.last_task_aux_metrics = {}
        self.num_conv_layers = num_conv_layers
        self.surface_feature_dim = getattr(args, 'surface_feature_dim', 4) if args is not None else 4
        self.use_ligand_patch_tokenizer = getattr(args, 'use_ligand_patch_tokenizer', False) if args is not None else False
        self.use_nusurf_fusion = getattr(args, 'use_nusurf_fusion', False) if args is not None else False
        self.nusurf_num_blocks = max(1, int(getattr(args, 'nusurf_num_blocks', 1))) if self.use_nusurf_fusion else 0
        self.register_buffer('pretrain_surface_queue', torch.zeros(self.pretrain_contrastive_queue_size, ns))
        self.register_buffer('pretrain_ligand_queue', torch.zeros(self.pretrain_contrastive_queue_size, ns))
        self.register_buffer('pretrain_contrastive_queue_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('pretrain_contrastive_queue_ptr', torch.zeros((), dtype=torch.long))

        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.rec_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_residue_feature_dims,
            sigma_embed_dim=0,
            lm_embedding_type=lm_embedding_type,
            lm_embedding_dim=getattr(args, 'lm_embedding_dim', None) if args is not None else None,
        )
        self.rec_edge_embedding = nn.Sequential(nn.Linear(in_rec_edge_features + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        # surface embeddings
        self.surface_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=[[], self.surface_feature_dim], sigma_embed_dim=sigma_embed_dim)
        self.ligand_patch_tokenizer = (LigandAnchoredPatchTokenizer(ns, getattr(args, 'patch_temperature', 2.5))
                                       if self.use_ligand_patch_tokenizer else None)
        self.nusurf_fusion = (NuSurfAnchorFusion(ns, getattr(args, 'nucleic_feat_dim', 12),
                                                 getattr(args, 'nusurf_distance_scale', 6.0))
                              if self.use_nusurf_fusion else None)
        self.nusurf_refine_blocks = nn.ModuleList([
            NuSurfAnchorFusion(ns, getattr(args, 'nucleic_feat_dim', 12),
                               getattr(args, 'nusurf_distance_scale', 6.0))
            for _ in range(max(0, self.nusurf_num_blocks - 1))
        ])
        self.surface_edge_embedding = nn.Sequential(nn.Linear(3 + sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.surface_rec_cross_edge_embedding = nn.Sequential(nn.Linear(cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.surface_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)

        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]
        lig_conv_layers= []
        # surface modules
        surface_conv_layers,lig_to_surface_conv_layers, surface_to_lig_conv_layers = [], [],[]
        residue_to_surface_conv_layers = []
        rec_conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }
            if i ==0:
                residue_to_surface_conv_layers.append(TensorProductConvLayer(** {
                'in_irreps': f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                'sh_irreps': self.sh_irreps,
                'out_irreps':  in_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }))
                rec_conv_layers.append(TensorProductConvLayer(** {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }))
                
            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)

            if i != num_conv_layers - 1:

                # surface layers
                surface_layer = TensorProductConvLayer(**parameters)
                surface_conv_layers.append(surface_layer)
                lig_to_surface_layer = TensorProductConvLayer(**parameters)
                lig_to_surface_conv_layers.append(lig_to_surface_layer)

   
            # surface layers
            surface_to_lig_layer = TensorProductConvLayer(**parameters)
            surface_to_lig_conv_layers.append(surface_to_lig_layer)

        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)

        # surface cross residue layer
        self.residue_to_surface_conv_layers = nn.ModuleList(residue_to_surface_conv_layers)
        # surface layers
        self.surface_conv_layers = nn.ModuleList(surface_conv_layers)
        self.lig_to_surface_conv_layers = nn.ModuleList(lig_to_surface_conv_layers)
        self.surface_to_lig_conv_layers = nn.ModuleList(surface_to_lig_conv_layers)

        # Optional nucleic-acid specific feature fusion (keeps receptor.x dims unchanged)
        self.use_nucleic_feat_fusion = getattr(args, 'use_nucleic_feat_fusion', False) if args is not None else False
        self.nucleic_feat_dim = getattr(args, 'nucleic_feat_dim', 12) if args is not None else 12
        if self.use_nucleic_feat_fusion:
            self.nuc_feat_proj = nn.Sequential(
                nn.Linear(self.nucleic_feat_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns),
            )
            self.nuc_feat_gate = nn.Sequential(
                nn.Linear(self.nucleic_feat_dim, ns),
                nn.Sigmoid(),
            )
        
        # Adaptive weight transfer: Adapter layers for surface-related modules
        self.use_adaptive_transfer = getattr(args, 'use_adaptive_transfer', False) if args is not None else False
        self.adapter_rank = getattr(args, 'adapter_rank', 8) if args is not None else 8
        self.adapter_scope = getattr(args, 'adapter_scope', 'surface') if args is not None else 'surface'
        
        if self.use_adaptive_transfer and ScalarGatedAdapter is not None:
            # Create adapters for surface-related layers
            # For embedding layers (Linear-like), we use simpler approach
            # Adapters will be applied in forward pass
            
            # Adapters for surface node and edge embeddings
            self.surface_node_adapter = LinearAdapter(
                in_dim=ns, out_dim=ns, adapter_rank=self.adapter_rank, dropout=dropout
            ) if LinearAdapter is not None else None
            
            self.surface_edge_adapter = LinearAdapter(
                in_dim=ns, out_dim=ns, adapter_rank=self.adapter_rank, dropout=dropout
            ) if LinearAdapter is not None else None
            
            # Get feature dimensions from irreps
            # For adapters, we need the dimension of the feature representation
            # We'll use the dimension after the first layer as a reference
            first_irrep_dim = o3.Irreps(irrep_seq[0]).dim if len(irrep_seq) > 0 else ns
            # For later layers, use max dimension from irrep_seq
            max_irrep_dim = max([o3.Irreps(irrep).dim for irrep in irrep_seq]) if len(irrep_seq) > 0 else ns
            
            # Adapters for surface convolution layers (E3NN layers)
            # Each adapter operates on updates from the corresponding layer
            self.surface_conv_adapters = nn.ModuleList([
                ScalarGatedAdapter(
                    feature_dim=max_irrep_dim,
                    adapter_rank=self.adapter_rank,
                    dropout=dropout
                ) for i in range(num_conv_layers - 1)
            ]) if ScalarGatedAdapter is not None else nn.ModuleList()
            
            self.lig_to_surface_adapters = nn.ModuleList([
                ScalarGatedAdapter(
                    feature_dim=max_irrep_dim,
                    adapter_rank=self.adapter_rank,
                    dropout=dropout
                ) for i in range(num_conv_layers - 1)
            ]) if ScalarGatedAdapter is not None else nn.ModuleList()
            
            self.surface_to_lig_adapters = nn.ModuleList([
                ScalarGatedAdapter(
                    feature_dim=max_irrep_dim,
                    adapter_rank=self.adapter_rank,
                    dropout=dropout
                ) for i in range(num_conv_layers)
            ]) if ScalarGatedAdapter is not None else nn.ModuleList()
            
            self.residue_to_surface_adapter = ScalarGatedAdapter(
                feature_dim=first_irrep_dim,
                adapter_rank=self.adapter_rank,
                dropout=dropout
            ) if ScalarGatedAdapter is not None else None
            
            # Extended adapters for 'full' scope: ligand layers, cross-attention, output heads
            if self.adapter_scope == 'full':
                # Adapters for ligand node and edge embeddings
                self.lig_node_adapter = LinearAdapter(
                    in_dim=ns, out_dim=ns, adapter_rank=self.adapter_rank, dropout=dropout
                ) if LinearAdapter is not None else None
                
                self.lig_edge_adapter = LinearAdapter(
                    in_dim=ns, out_dim=ns, adapter_rank=self.adapter_rank, dropout=dropout
                ) if LinearAdapter is not None else None
                
                # Adapters for ligand convolution layers
                self.lig_conv_adapters = nn.ModuleList([
                    ScalarGatedAdapter(
                        feature_dim=max_irrep_dim,
                        adapter_rank=self.adapter_rank,
                        dropout=dropout
                    ) for i in range(num_conv_layers)
                ]) if ScalarGatedAdapter is not None else nn.ModuleList()
                
                # Adapters for cross edge embedding
                self.cross_edge_adapter = LinearAdapter(
                    in_dim=ns, out_dim=ns, adapter_rank=self.adapter_rank, dropout=dropout
                ) if LinearAdapter is not None else None
                
                # Adapters for output heads (tr_final, rot_final, tor_final)
                self.tr_final_adapter = LinearAdapter(
                    in_dim=1, out_dim=1, adapter_rank=min(self.adapter_rank, 4), dropout=dropout
                ) if LinearAdapter is not None else None
                
                self.rot_final_adapter = LinearAdapter(
                    in_dim=1, out_dim=1, adapter_rank=min(self.adapter_rank, 4), dropout=dropout
                ) if LinearAdapter is not None else None
                
                self.tor_final_adapter = LinearAdapter(
                    in_dim=1, out_dim=1, adapter_rank=min(self.adapter_rank, 4), dropout=dropout
                ) if LinearAdapter is not None and not no_torsion else None
                
                logger.info(f"Full adapter scope enabled: ligand layers, cross-attention, output heads")
            else:
                self.lig_node_adapter = None
                self.lig_edge_adapter = None
                self.lig_conv_adapters = nn.ModuleList()
                self.cross_edge_adapter = None
                self.tr_final_adapter = None
                self.rot_final_adapter = None
                self.tor_final_adapter = None
        else:
            self.surface_node_adapter = None
            self.surface_edge_adapter = None
            self.surface_conv_adapters = nn.ModuleList()
            self.lig_to_surface_adapters = nn.ModuleList()
            self.surface_to_lig_adapters = nn.ModuleList()
            self.residue_to_surface_adapter = None
            self.lig_node_adapter = None
            self.lig_edge_adapter = None
            self.lig_conv_adapters = nn.ModuleList()
            self.cross_edge_adapter = None
            self.tr_final_adapter = None
            self.rot_final_adapter = None
            self.tor_final_adapter = None

        if self.confidence_mode:
            surface_hidden_dim = 2 * ns + 6 * nv
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2*self.ns if num_conv_layers >= 3 else self.ns,ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs)
            )
            self.surface_head = nn.Sequential(
                    nn.Linear(surface_hidden_dim, 2*ns, bias=False),
                    nn.BatchNorm1d(2*ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(2*ns, 3, bias=False)
                )
            if self.pretrain_v2_mode:
                # Surface is updated through all but the terminal ligand layer.
                # Its width follows the irrep schedule rather than a 6-layer
                # hard-code, keeping EMA/checkpoint initialization eager.
                terminal_surface_dim = o3.Irreps(irrep_seq[min(num_conv_layers - 1, len(irrep_seq) - 1)]).dim
                self.pretrain_surface_contact_head = nn.Linear(terminal_surface_dim, 1)
                self.pretrain_surface_reconstruction_head = nn.Linear(terminal_surface_dim, self.surface_feature_dim)
                self.pretrain_anchor_contact_head = nn.Linear(ns, 1)
        else:
            #for pretrain
            # self.surface_head = nn.Sequential(
            #         nn.Linear(2*ns+6*nv, 2*ns, bias=False),
            #         nn.BatchNorm1d(2*ns) if not confidence_no_batchnorm else nn.Identity(),
            #         nn.Tanh(),
            #         nn.Dropout(dropout),
            #         nn.Linear(2*ns, 3, bias=False)
            #     )
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

            if not no_torsion:
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )

                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")

                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.lig_conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )

                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

        if self.task_aligned_aux:
            terminal_surface_dim = o3.Irreps(
                irrep_seq[min(num_conv_layers - 1, len(irrep_seq) - 1)]
            ).dim
            terminal_ligand_dim = self.lig_conv_layers[-1].out_irreps.dim
            self.task_surface_contact_head = nn.Linear(terminal_surface_dim, 1)
            self.task_ligand_contact_head = nn.Linear(terminal_ligand_dim, 1)
            self.task_anchor_contact_head = nn.Linear(ns, 1)



    @staticmethod
    def _balanced_contact_loss(logits, target, sample_weight=None):
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(min=1.0, max=20.0)
        per_item = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction='none'
        )
        if sample_weight is None:
            return per_item.mean()
        sample_weight = sample_weight.to(per_item).clamp_min(1e-4)
        return (per_item * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)

    @staticmethod
    def _native_ligand_positions(data):
        native_pos = getattr(data['ligand'], 'orig_pos', None)
        if native_pos is None:
            return None
        if torch.is_tensor(native_pos):
            native_pos = native_pos.to(device=data['ligand'].pos.device, dtype=data['ligand'].pos.dtype)
        elif isinstance(native_pos, (list, tuple)):
            native_pos = torch.cat([
                torch.as_tensor(part, device=data['ligand'].pos.device, dtype=data['ligand'].pos.dtype)
                for part in native_pos
            ], dim=0)
        else:
            native_pos = torch.as_tensor(
                native_pos, device=data['ligand'].pos.device, dtype=data['ligand'].pos.dtype
            )
        native_pos = native_pos.reshape(-1, 3)
        if native_pos.shape[0] != data['ligand'].pos.shape[0]:
            return None
        return native_pos

    def _compute_task_aligned_auxiliary(self, data, surface_node_attr, lig_node_attr, anchor_state):
        native_pos = self._native_ligand_positions(data)
        if native_pos is None or native_pos.numel() == 0:
            self.last_task_aux_loss = surface_node_attr.sum() * 0.0
            self.last_task_aux_metrics = {'native_position_available': False}
            return

        radius = float(self.task_aux_contact_radius)
        surface_batch = data['surface'].batch
        ligand_batch = data['ligand'].batch
        distance = torch.cdist(data['surface'].pos, native_pos)
        same_complex = surface_batch[:, None] == ligand_batch[None, :]
        distance = distance.masked_fill(~same_complex, float('inf'))

        surface_target = (distance.min(dim=1).values <= radius).float()
        ligand_target = (distance.min(dim=0).values <= radius).float()
        surface_logits = self.task_surface_contact_head(surface_node_attr).squeeze(-1)
        ligand_logits = self.task_ligand_contact_head(lig_node_attr).squeeze(-1)

        graph_t = data.complex_t['tr'].to(surface_logits)
        decay = max(0.0, float(self.task_aux_noise_decay))
        surface_weight = torch.exp(-decay * graph_t[surface_batch])
        ligand_weight = torch.exp(-decay * graph_t[ligand_batch])
        surface_loss = self._balanced_contact_loss(surface_logits, surface_target, surface_weight)
        ligand_loss = self._balanced_contact_loss(ligand_logits, ligand_target, ligand_weight)

        anchor_loss = surface_loss.new_zeros(())
        anchor_positive_fraction = surface_loss.new_full((), -1.0)
        if anchor_state is not None and anchor_state.numel() > 0:
            anchor_batch = data['receptor'].batch[data['receptor'].nucleic_anchor_parent_index]
            anchor_distance = torch.cdist(data['receptor'].nucleic_anchor_pos, native_pos)
            anchor_same_complex = anchor_batch[:, None] == ligand_batch[None, :]
            anchor_distance = anchor_distance.masked_fill(~anchor_same_complex, float('inf'))
            anchor_target = (anchor_distance.min(dim=1).values <= radius).float()
            anchor_logits = self.task_anchor_contact_head(anchor_state).squeeze(-1)
            anchor_weight = torch.exp(-decay * graph_t[anchor_batch])
            anchor_loss = self._balanced_contact_loss(anchor_logits, anchor_target, anchor_weight)
            anchor_positive_fraction = anchor_target.mean()

        surface_pool = scatter_mean(surface_node_attr[:, :self.ns], surface_batch, dim=0)
        ligand_pool = scatter_mean(lig_node_attr[:, :self.ns], ligand_batch, dim=0)
        pair_count = min(surface_pool.shape[0], ligand_pool.shape[0])
        contrastive_loss = surface_loss.new_zeros(())
        if pair_count > 1:
            surface_pool = F.normalize(surface_pool[:pair_count], dim=-1)
            ligand_pool = F.normalize(ligand_pool[:pair_count], dim=-1)
            logits = surface_pool @ ligand_pool.transpose(0, 1)
            logits = logits / max(float(self.pretrain_contrastive_temperature), 1e-3)
            target = torch.arange(pair_count, device=logits.device)
            contrastive_loss = 0.5 * (
                F.cross_entropy(logits, target) + F.cross_entropy(logits.transpose(0, 1), target)
            )

        total = (
            surface_loss
            + float(self.task_aux_ligand_weight) * ligand_loss
            + float(self.task_aux_anchor_weight) * anchor_loss
            + float(self.task_aux_contrastive_weight) * contrastive_loss
        )
        self.last_task_aux_loss = total
        self.last_task_aux_metrics = {
            'native_position_available': True,
            'surface_loss': surface_loss.detach(),
            'ligand_loss': ligand_loss.detach(),
            'anchor_loss': anchor_loss.detach(),
            'contrastive_loss': contrastive_loss.detach(),
            'surface_positive_fraction': surface_target.mean().detach(),
            'ligand_positive_fraction': ligand_target.mean().detach(),
            'anchor_positive_fraction': anchor_positive_fraction.detach(),
            'mean_noise_weight': torch.exp(-decay * graph_t).mean().detach(),
            'pair_count': surface_loss.new_tensor(float(pair_count)),
        }

    def forward(self, data):
        #logger.info(f'forward data: {data.name}')
        self.last_task_aux_loss = None
        self.last_task_aux_metrics = {}
        
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        # Apply ligand node adapter if enabled (full scope)
        if self.use_adaptive_transfer and self.lig_node_adapter is not None:
            lig_node_attr = self.lig_node_adapter(lig_node_attr, lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        # Apply ligand edge adapter if enabled (full scope)
        if self.use_adaptive_transfer and self.lig_edge_adapter is not None:
            lig_edge_attr = self.lig_edge_adapter(lig_edge_attr, lig_edge_attr)
        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        if self.use_nucleic_feat_fusion and hasattr(data['receptor'], 'nucleic_feat'):
            try:
                nuc_feat = data['receptor'].nucleic_feat
                if torch.is_tensor(nuc_feat):
                    nuc_feat = nuc_feat.to(rec_node_attr.device).float()
                    nuc_feat = torch.nan_to_num(nuc_feat, nan=0.0, posinf=0.0, neginf=0.0)
                    rec_node_attr = rec_node_attr + self.nuc_feat_gate(nuc_feat) * self.nuc_feat_proj(nuc_feat)
            except Exception:
                pass
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        # build surface graph
        surface_node_attr,surface_edge_index, surface_edge_attr, surface_edge_sh , masked_input, mask = self.build_surface_conv_graph(data,mode=self.confidence_mode)
        surface_src, surface_dst = surface_edge_index

        surface_node_attr = self.surface_node_embedding(surface_node_attr)
        ligand_patch_logits = None
        if self.ligand_patch_tokenizer is not None:
            ligand_scalar, ligand_patch_logits = self.ligand_patch_tokenizer(
                lig_node_attr[:, :self.ns], data['ligand'].pos, data['ligand'].batch,
                surface_node_attr[:, :self.ns], data['surface'].pos, data['surface'].batch,
            )
            lig_node_attr = torch.cat([ligand_scalar, lig_node_attr[:, self.ns:]], dim=-1)
        # Apply adapter if enabled
        if self.use_adaptive_transfer and self.surface_node_adapter is not None:
            surface_node_attr = self.surface_node_adapter(surface_node_attr, surface_node_attr)
        
        surface_edge_attr = self.surface_edge_embedding(surface_edge_attr)
        # Apply adapter if enabled
        if self.use_adaptive_transfer and self.surface_edge_adapter is not None:
            surface_edge_attr = self.surface_edge_adapter(surface_edge_attr, surface_edge_attr)
        # use a layer to get residue feature to surface nodes
        # then  drop residus nodes
        # build cross graph
        if self.dynamic_max_cross:
            # this distance may can be changed for given pocket
            cross_cutoff = (tr_sigma * 3 + 10).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance

        # surface cross graph build
        surface_cross_edge_index, surface_cross_edge_attr, surface_cross_edge_sh = self.build_surface_cross_conv_graph(data, cross_cutoff)
        surface_cross_lig, surface_cross_rec = surface_cross_edge_index
        surface_cross_edge_attr = self.cross_edge_embedding(surface_cross_edge_attr)
        # Apply cross edge adapter if enabled (full scope)
        if self.use_adaptive_transfer and self.cross_edge_adapter is not None:
            surface_cross_edge_attr = self.cross_edge_adapter(surface_cross_edge_attr, surface_cross_edge_attr)

        # sesidueurface ,r cross graph builld this info will use one shot
        surface_rec_cross_edge_index, surface_rec_cross_edge_attr, surface_rec_cross_edge_sh = self.build_surface_rec_cross_conv_graph(data)
        surface_rec_cross_rec, surface_rec_cross_surface = surface_rec_cross_edge_index
        surface_rec_cross_edge_attr = self.surface_rec_cross_edge_embedding( surface_rec_cross_edge_attr)

        residue_to_surface_edge_attr_ = torch.cat([surface_rec_cross_edge_attr, rec_node_attr[ surface_rec_cross_rec, :self.ns], surface_node_attr[surface_rec_cross_surface, :self.ns]], -1)
        
        # update receptor embedding and then update embedding to surface
        rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
        rec_intra_update = self.rec_conv_layers[0](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
        rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
        rec_node_attr = rec_node_attr + rec_intra_update

        # just one layer for feature update ,maybe can add more layers?
        surface_inter_residue_update = self.residue_to_surface_conv_layers[0](rec_node_attr, torch.flip(surface_rec_cross_edge_index,dims = [0]), residue_to_surface_edge_attr_, surface_rec_cross_edge_sh,
                                                            out_nodes=surface_node_attr.shape[0])
        
        # Apply adapter if enabled
        # Note: We need to pad surface_node_attr to match update dimension if needed
        if self.use_adaptive_transfer and self.residue_to_surface_adapter is not None:
            surface_node_attr_padded = F.pad(surface_node_attr, (0, surface_inter_residue_update.shape[-1] - surface_node_attr.shape[-1]))
            surface_inter_residue_update = self.residue_to_surface_adapter(surface_node_attr_padded, surface_inter_residue_update)
        
        surface_node_attr = F.pad(surface_node_attr, (0, surface_inter_residue_update.shape[-1] - surface_node_attr.shape[-1]))
        status3=torch.isnan(surface_node_attr).any()
        if status3:
            logger.info(f'nan surface_node_attr 3 : {status3}')

        #no update for receptor node for pretrain
        surface_node_attr = surface_node_attr + surface_inter_residue_update
        anchor_state = None
        if self.nusurf_fusion is not None and hasattr(data['receptor'], 'nucleic_anchor_pos'):
            anchor_pos = data['receptor'].nucleic_anchor_pos
            if anchor_pos.numel() > 0:
                fused_scalar, anchor_state = self.nusurf_fusion(
                    surface_node_attr[:, :self.ns], data['surface'].pos, data['surface'].batch,
                    anchor_pos, data['receptor'].nucleic_anchor_type,
                    data['receptor'].nucleic_anchor_parent_index, data['receptor'].batch,
                    getattr(data['receptor'], 'nucleic_feat', None),
                )
                surface_node_attr = torch.cat([fused_scalar, surface_node_attr[:, self.ns:]], dim=-1)

        status4=torch.isnan(lig_node_attr).any()
        if status4:
            logger.info(f'nan lig_node_attr 4 : {status4}')

        for l in range(len(self.lig_conv_layers)):

            # intra graph message passing
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)
            
            # Apply ligand conv adapter if enabled (full scope)
            if self.use_adaptive_transfer and l < len(self.lig_conv_adapters) and len(self.lig_conv_adapters) > 0:
                input_padded = F.pad(lig_node_attr, (0, max(0, self.lig_conv_adapters[l].feature_dim - lig_node_attr.shape[-1])))
                update_padded = F.pad(lig_intra_update, (0, max(0, self.lig_conv_adapters[l].feature_dim - lig_intra_update.shape[-1])))
                adapted_update = self.lig_conv_adapters[l](input_padded, update_padded)
                lig_intra_update = adapted_update[:, :lig_intra_update.shape[-1]]

            # surface inter graph message passing
            surface_to_lig_edge_attr_ = torch.cat([surface_cross_edge_attr, lig_node_attr[surface_cross_lig, :self.ns], surface_node_attr[surface_cross_rec, :self.ns]], -1)
            surface_lig_inter_update = self.surface_to_lig_conv_layers[l](surface_node_attr, surface_cross_edge_index, surface_to_lig_edge_attr_, surface_cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0])
            # Apply adapter if enabled
            if self.use_adaptive_transfer and l < len(self.surface_to_lig_adapters) and self.surface_to_lig_adapters[l] is not None:
                input_padded = F.pad(lig_node_attr, (0, max(0, self.surface_to_lig_adapters[l].feature_dim - lig_node_attr.shape[-1])))
                update_padded = F.pad(surface_lig_inter_update, (0, max(0, self.surface_to_lig_adapters[l].feature_dim - surface_lig_inter_update.shape[-1])))
                adapted_update = self.surface_to_lig_adapters[l](input_padded, update_padded)
                surface_lig_inter_update = adapted_update[:, :surface_lig_inter_update.shape[-1]]
            
            if l != len(self.lig_conv_layers) - 1:

                # surface intra graph message passing
                surface_edge_attr_ = torch.cat([surface_edge_attr, surface_node_attr[surface_src, :self.ns], surface_node_attr[surface_dst, :self.ns]], -1)
                surface_intra_update = self.surface_conv_layers[l](surface_node_attr, surface_edge_index, surface_edge_attr_, surface_edge_sh)
                # Apply adapter if enabled
                if self.use_adaptive_transfer and l < len(self.surface_conv_adapters) and self.surface_conv_adapters[l] is not None:
                    # Pad input to match adapter dimension if needed
                    input_padded = F.pad(surface_node_attr, (0, max(0, self.surface_conv_adapters[l].feature_dim - surface_node_attr.shape[-1])))
                    update_padded = F.pad(surface_intra_update, (0, max(0, self.surface_conv_adapters[l].feature_dim - surface_intra_update.shape[-1])))
                    adapted_update = self.surface_conv_adapters[l](input_padded, update_padded)
                    surface_intra_update = adapted_update[:, :surface_intra_update.shape[-1]]

                # lig to surface inter graph message passing
                lig_to_surface_edge_attr_ = torch.cat([surface_cross_edge_attr, lig_node_attr[surface_cross_lig, :self.ns],surface_node_attr[surface_cross_rec, :self.ns]], -1)
                
                surface_inter_update = self.lig_to_surface_conv_layers[l](lig_node_attr, torch.flip(surface_cross_edge_index, dims=[0]), lig_to_surface_edge_attr_, surface_cross_edge_sh,
                                                              out_nodes=surface_node_attr.shape[0])
                # Apply adapter if enabled
                if self.use_adaptive_transfer and l < len(self.lig_to_surface_adapters) and self.lig_to_surface_adapters[l] is not None:
                    input_padded = F.pad(surface_node_attr, (0, max(0, self.lig_to_surface_adapters[l].feature_dim - surface_node_attr.shape[-1])))
                    update_padded = F.pad(surface_inter_update, (0, max(0, self.lig_to_surface_adapters[l].feature_dim - surface_inter_update.shape[-1])))
                    adapted_update = self.lig_to_surface_adapters[l](input_padded, update_padded)
                    surface_inter_update = adapted_update[:, :surface_inter_update.shape[-1]]
                # print(lig_node_attr.shape, torch.flip(surface_cross_edge_index, dims=[0]).shape, lig_to_surface_edge_attr_.shape, surface_cross_edge_sh.shape,
                #                                               surface_node_attr.shape[0])

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update  + surface_lig_inter_update
            if l != len(self.lig_conv_layers) - 1:

                # surface update
                surface_node_attr = F.pad(surface_node_attr, (0, surface_intra_update.shape[-1] - surface_node_attr.shape[-1]))
                surface_node_attr = surface_node_attr + surface_intra_update + surface_inter_update

        # The first NuSurf block conditions the input surface representation.
        # Optional refinement blocks then re-condition the task-facing surface
        # after ligand/surface equivariant message passing.  These blocks only
        # consume scalar features and pairwise distances, so they preserve the
        # SE(3) contract of the backbone.
        if len(self.nusurf_refine_blocks) > 0 and hasattr(data['receptor'], 'nucleic_anchor_pos'):
            anchor_pos = data['receptor'].nucleic_anchor_pos
            if anchor_pos.numel() > 0:
                for nusurf_block in self.nusurf_refine_blocks:
                    fused_scalar, anchor_state = nusurf_block(
                        surface_node_attr[:, :self.ns], data['surface'].pos, data['surface'].batch,
                        anchor_pos, data['receptor'].nucleic_anchor_type,
                        data['receptor'].nucleic_anchor_parent_index, data['receptor'].batch,
                        getattr(data['receptor'], 'nucleic_feat', None),
                    )
                    surface_node_attr = torch.cat([fused_scalar, surface_node_attr[:, self.ns:]], dim=-1)

        if self.task_aligned_aux and not self.confidence_mode:
            self._compute_task_aligned_auxiliary(data, surface_node_attr, lig_node_attr, anchor_state)

        # compute confidence score, now for pretrain
        if self.confidence_mode:
            if self.pretrain_v2_mode:
                contact_radius = getattr(self, 'pretrain_contact_radius', None)
                contact_radius = contact_radius or 4.5
                surface_distance = torch.cdist(data['surface'].pos, data['ligand'].pos)
                same_complex = data['surface'].batch[:, None] == data['ligand'].batch[None, :]
                surface_distance = surface_distance.masked_fill(~same_complex, float('inf'))
                surface_target = (surface_distance.min(dim=1).values <= contact_radius).float()
                surface_logits = self.pretrain_surface_contact_head(surface_node_attr).squeeze(-1)
                pos = surface_target.sum()
                neg = surface_target.numel() - pos
                pos_weight = (neg / pos.clamp_min(1.0)).clamp(max=20.0)
                field_loss = F.binary_cross_entropy_with_logits(surface_logits, surface_target, pos_weight=pos_weight)
                patch_loss = surface_logits.new_zeros(())
                patch_positive_fraction = surface_logits.new_full((), -1.0)
                if ligand_patch_logits is not None:
                    patch_target = (surface_distance.min(dim=0).values <= contact_radius).float()
                    patch_positive_fraction = patch_target.mean()
                    patch_pos = patch_target.sum()
                    patch_neg = patch_target.numel() - patch_pos
                    patch_weight = (patch_neg / patch_pos.clamp_min(1.0)).clamp(max=20.0)
                    patch_loss = F.binary_cross_entropy_with_logits(ligand_patch_logits, patch_target, pos_weight=patch_weight)
                anchor_loss = surface_logits.new_zeros(())
                anchor_positive_fraction = surface_logits.new_full((), -1.0)
                if anchor_state is not None and anchor_state.numel() > 0:
                    anchor_distance = torch.cdist(data['receptor'].nucleic_anchor_pos, data['ligand'].pos)
                    anchor_batch = data['receptor'].batch[data['receptor'].nucleic_anchor_parent_index]
                    same_complex = anchor_batch[:, None] == data['ligand'].batch[None, :]
                    anchor_distance = anchor_distance.masked_fill(~same_complex, float('inf'))
                    anchor_target = (anchor_distance.min(dim=1).values <= contact_radius).float()
                    anchor_positive_fraction = anchor_target.mean()
                    anchor_logits = self.pretrain_anchor_contact_head(anchor_state).squeeze(-1)
                    anchor_pos = anchor_target.sum()
                    anchor_neg = anchor_target.numel() - anchor_pos
                    anchor_weight = (anchor_neg / anchor_pos.clamp_min(1.0)).clamp(max=20.0)
                    anchor_loss = F.binary_cross_entropy_with_logits(anchor_logits, anchor_target, pos_weight=anchor_weight)
                # Pair each pooled surface with its native ligand and use both
                # in-batch mismatches and a detached cross-batch queue as
                # negatives.  The queue matters because the formal geometry
                # budget is batch_size=2, where in-batch contrast alone would
                # otherwise provide just one negative per example.
                surface_pool = scatter_mean(surface_node_attr[:, :self.ns], data['surface'].batch, dim=0)
                ligand_pool = scatter_mean(lig_node_attr[:, :self.ns], data['ligand'].batch, dim=0)
                pair_count = min(surface_pool.shape[0], ligand_pool.shape[0])
                contrastive_loss = surface_logits.new_zeros(())
                queue_count_before = int(self.pretrain_contrastive_queue_count.item())
                if pair_count > 0:
                    surface_pool = F.normalize(surface_pool[:pair_count], dim=-1)
                    ligand_pool = F.normalize(ligand_pool[:pair_count], dim=-1)
                    temperature = max(float(getattr(self, 'pretrain_contrastive_temperature', 0.2)), 1e-3)
                    pair_logits = surface_pool @ ligand_pool.transpose(0, 1) / temperature
                    queue_count = min(int(self.pretrain_contrastive_queue_count.item()),
                                      self.pretrain_contrastive_queue_size)
                    if queue_count:
                        # Clone the detached snapshot: the queue is updated
                        # before backward, and a view would otherwise trigger
                        # autograd's in-place-version guard.
                        queued_surface = self.pretrain_surface_queue[:queue_count].detach().clone()
                        queued_ligand = self.pretrain_ligand_queue[:queue_count].detach().clone()
                        surface_to_lig_logits = torch.cat(
                            [pair_logits, surface_pool @ queued_ligand.transpose(0, 1) / temperature], dim=1
                        )
                        ligand_to_surface_logits = torch.cat(
                            [pair_logits.transpose(0, 1), ligand_pool @ queued_surface.transpose(0, 1) / temperature], dim=1
                        )
                    else:
                        surface_to_lig_logits = pair_logits
                        ligand_to_surface_logits = pair_logits.transpose(0, 1)
                    pair_target = torch.arange(pair_count, device=pair_logits.device)
                    contrastive_loss = 0.5 * (
                        F.cross_entropy(surface_to_lig_logits, pair_target) +
                        F.cross_entropy(ligand_to_surface_logits, pair_target)
                    )
                    if self.training and self.pretrain_contrastive_queue_size:
                        with torch.no_grad():
                            queue_capacity = self.pretrain_contrastive_queue_size
                            enqueue_count = min(pair_count, queue_capacity)
                            enqueue_surface = surface_pool.detach()[-enqueue_count:]
                            enqueue_ligand = ligand_pool.detach()[-enqueue_count:]
                            queue_ptr = int(self.pretrain_contrastive_queue_ptr.item())
                            first = min(enqueue_count, queue_capacity - queue_ptr)
                            self.pretrain_surface_queue[queue_ptr:queue_ptr + first].copy_(enqueue_surface[:first])
                            self.pretrain_ligand_queue[queue_ptr:queue_ptr + first].copy_(enqueue_ligand[:first])
                            if first < enqueue_count:
                                rest = enqueue_count - first
                                self.pretrain_surface_queue[:rest].copy_(enqueue_surface[first:])
                                self.pretrain_ligand_queue[:rest].copy_(enqueue_ligand[first:])
                            self.pretrain_contrastive_queue_ptr.fill_((queue_ptr + enqueue_count) % queue_capacity)
                            self.pretrain_contrastive_queue_count.fill_(min(queue_capacity, queue_count + enqueue_count))
                interaction_loss = (field_loss + getattr(self, 'pretrain_patch_weight', 0.5) * patch_loss +
                                    getattr(self, 'pretrain_anchor_weight', 0.5) * anchor_loss +
                                    getattr(self, 'pretrain_contrastive_weight', 0.25) * contrastive_loss)
                reconstruction = self.pretrain_surface_reconstruction_head(surface_node_attr)
                masked = ~mask
                reconstruction_loss = (F.smooth_l1_loss(reconstruction[masked], masked_input[masked])
                                       if masked.any() else reconstruction.sum() * 0.0)
                self.last_pretrain_metrics = {
                    'field_loss': field_loss.detach(),
                    'patch_loss': patch_loss.detach(),
                    'anchor_loss': anchor_loss.detach(),
                    'contrastive_loss': contrastive_loss.detach(),
                    'interaction_loss': interaction_loss.detach(),
                    'reconstruction_loss': reconstruction_loss.detach(),
                    'surface_positive_fraction': surface_target.mean().detach(),
                    'patch_positive_fraction': patch_positive_fraction.detach(),
                    'anchor_positive_fraction': anchor_positive_fraction.detach(),
                    'masked_surface_fraction': masked.float().mean().detach(),
                    'pair_count': surface_logits.new_tensor(float(pair_count)),
                    'queue_count_before': surface_logits.new_tensor(float(queue_count_before)),
                    'queue_count_after': self.pretrain_contrastive_queue_count.detach().float().clone(),
                    'queue_ptr_after': self.pretrain_contrastive_queue_ptr.detach().float().clone(),
                }
                return interaction_loss, getattr(self, 'pretrain_reconstruction_weight', 1.0) * reconstruction_loss
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
            status1=torch.isnan(scalar_lig_attr).any()
            if status1:
                logger.info(f'nan scalar_lig_attr : {status1}')
            pred_rmsd = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            status2=torch.isnan(surface_node_attr).any()
            if status2:
                logger.info(f'nan surface_node_attr : {status2}')
            pred_surface = self.surface_head(surface_node_attr)
            loss_interaction = nn.MSELoss()(data['RMSD'], pred_rmsd)
            loss_surface = nn.MSELoss()(masked_input[~mask], pred_surface[~mask])
            # logger.info('loss_interaction: ', loss_interaction.item())
            # logger.info('loss_surface: ', loss_surface.item())
            return loss_interaction, loss_surface

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        # print(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh,data.num_graphs)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_final_out = self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        # Apply tr_final adapter if enabled (full scope)
        if self.use_adaptive_transfer and self.tr_final_adapter is not None:
            tr_final_out = self.tr_final_adapter(tr_final_out, tr_final_out)
        tr_pred = tr_pred / tr_norm.clamp_min(1e-8) * tr_final_out
        tr_pred = torch.nan_to_num(tr_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_final_out = self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))
        # Apply rot_final adapter if enabled (full scope)
        if self.use_adaptive_transfer and self.rot_final_adapter is not None:
            rot_final_out = self.rot_final_adapter(rot_final_out, rot_final_out)
        rot_pred = rot_pred / rot_norm.clamp_min(1e-8) * rot_final_out
        rot_pred = torch.nan_to_num(rot_pred, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: return tr_pred, rot_pred, torch.empty(0, device=self.device)
        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean')
        tor_final_out = self.tor_final_layer(tor_pred)
        # Apply tor_final adapter if enabled (full scope)
        if self.use_adaptive_transfer and self.tor_final_adapter is not None:
            tor_final_out = self.tor_final_adapter(tor_final_out, tor_final_out)
        tor_pred = tor_final_out.squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        # #for pretraining                               
        # pred_surface = self.surface_head(surface_node_attr)
        # mask_loss = nn.MSELoss()(masked_input[~mask], pred_surface[~mask])

        return tr_pred, rot_pred, tor_pred

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])

        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


    def build_surface_conv_graph(self, data, mode):
        # builds the receptor initial node and edge embeddings
        # tr = data['receptor'].node_t['tr']
        if hasattr(data['surface'], 'batch') and hasattr(data, 'complex_t'):
            tr = data.complex_t['tr'][data['surface'].batch].to(data['surface'].pos.device)
        else:
            tr0 = data['receptor'].node_t['tr'][0]
            tr = tr0 * torch.ones(data['surface'].num_nodes, device=tr0.device)

        data['surface'].node_sigma_emb = self.timestep_emb_func(tr) # tr rot and tor noise is all the same
        # surface may have nan in features
                ####### setting mask_matrix here
        masked_input,mask = None,None
        # For pretrain
        surface_x = torch.nan_to_num(data['surface'].x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if mode:
            mask = torch.bernoulli(torch.full_like(surface_x, 0.2)).bool()  # 伯努利采样
            masked_input = surface_x.clone()
            surface_x = surface_x.clone()
            surface_x[~mask] = 0
        else:
            mask = torch.ones_like(surface_x, dtype=torch.bool)
            masked_input = surface_x
        #######

        node_attr = torch.cat([surface_x, data['surface'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['surface','surface_edge','surface'].edge_index
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['surface'].pos[src.long()]

        edge_length_emb = self.surface_distance_expansion(edge_vec.norm(dim=-1))

        edge_sigma_emb = data['surface'].node_sigma_emb[edge_index[0].long()]

        # The cached PyG ``Cartesian`` pseudo-coordinate is a normalized XYZ
        # vector in the preprocessing frame.  Feeding its components through a
        # scalar MLP breaks SE(3) equivariance.  Direction is already encoded by
        # ``edge_sh`` and length by ``edge_length_emb``; retain three zero
        # placeholders to keep all historical checkpoint shapes compatible.
        invariant_placeholders = torch.zeros_like(
            data['surface','surface_edge','surface'].edge_attr
        )
        edge_attr = torch.cat([invariant_placeholders, edge_sigma_emb, edge_length_emb], 1).float()
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh , masked_input, mask
        
    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        # data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        node_attr = data['receptor'].x

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]

        edge_attr = torch.cat([data['receptor', 'rec_contact', 'receptor'].edge_attr, edge_length_emb], 1).float()
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_surface_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['surface'].pos / cross_distance_cutoff[data['surface'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['surface'].batch, data['ligand'].batch, max_num_neighbors=30)
        else:
            edge_index = radius(data['surface'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['surface'].batch, data['ligand'].batch, max_num_neighbors=30)
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh
    def build_surface_rec_cross_conv_graph(self, data, cross_distance_cutoff = 15):
        edge_index = radius(data['surface'].pos, data['receptor'].pos, cross_distance_cutoff,
                        data['surface'].batch, data['receptor'].batch, max_num_neighbors=30)
        src, dst = edge_index
        edge_vec = data['surface'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        # edge_sigma_emb = data['receptor'].node_sigma_emb[src.long()]
        edge_attr = edge_length_emb#torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh


    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
