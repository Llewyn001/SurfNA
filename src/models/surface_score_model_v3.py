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
        self.num_conv_layers = num_conv_layers
        self.surface_feature_dim = getattr(args, 'surface_feature_dim', 4) if args is not None else 4

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
                    nn.Linear(2*ns+6*nv, 2*ns, bias=False),
                    nn.BatchNorm1d(2*ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(2*ns, 3, bias=False)
                )
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



    def forward(self, data):
        #logger.info(f'forward data: {data.name}')
        
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

        # compute confidence score, now for pretrain
        if self.confidence_mode:
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

        edge_attr = torch.cat([data['surface','surface_edge','surface'].edge_attr,edge_sigma_emb, edge_length_emb], 1).float()
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
