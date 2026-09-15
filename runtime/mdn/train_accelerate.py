#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")
import copy
import math
import os
import re
os.environ["WANDB_MODE"] = "offline" 
from functools import partial

import wandb
import torch
from accelerate.utils import set_seed
torch.multiprocessing.set_sharing_strategy('file_system')

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch,inference_epoch_parallel
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage
import datetime
from loguru import logger
# from models.score_model_mdn_energy import TensorProductEnergyModel

SURFACE_RESTART_KEYS = (
    'surface_node_embedding',
    'surface_edge_embedding',
    'surface_rec_cross_edge_embedding',
    'surface_distance_expansion',
    'surface_conv_layers',
    'lig_to_surface_conv_layers',
    'surface_to_lig_conv_layers',
    'residue_to_surface_conv_layers',
)


def _resolve_restart_path(args, default_file):
    restart_file = getattr(args, 'restart_file', None) or default_file
    if os.path.isabs(restart_file):
        return restart_file
    return os.path.join(args.restart_dir, restart_file)


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _filter_restart_state(model, state_dict, key_filter):
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    skipped_filter = 0
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith('module.') else key
        if key_filter == 'surface' and not any(token in normalized_key for token in SURFACE_RESTART_KEYS):
            skipped_filter += 1
            continue
        if normalized_key not in model_state:
            continue
        if tuple(model_state[normalized_key].shape) != tuple(value.shape):
            skipped_shape.append((normalized_key, tuple(value.shape), tuple(model_state[normalized_key].shape)))
            continue
        filtered[normalized_key] = value
    return filtered, skipped_filter, skipped_shape


def load_restart_weights(model, args, default_file):
    checkpoint_path = _resolve_restart_path(args, default_file)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = _extract_model_state(checkpoint)
    key_filter = getattr(args, 'restart_key_filter', 'all')
    filtered, skipped_filter, skipped_shape = _filter_restart_state(model, state_dict, key_filter)
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    logger.info(
        f"Loaded restart weights from {checkpoint_path}; filter={key_filter}; "
        f"loaded={len(filtered)} missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
        f"skipped_filter={skipped_filter} skipped_shape={len(skipped_shape)}"
    )
    if skipped_shape:
        logger.info(f"First skipped shape mismatches: {skipped_shape[:10]}")
    return checkpoint, checkpoint_path


def _apply_run_overrides(args):
    """Apply audited per-run operational overrides before a new run starts."""
    override_path = os.path.join(args.log_dir, '.run_overrides.yaml')
    if not os.path.isfile(override_path):
        return None
    with open(override_path, 'r') as handle:
        all_overrides = yaml.safe_load(handle) or {}
    run_overrides = all_overrides.get(args.run_name)
    if not run_overrides:
        return None

    allowed = {'val_inference_freq', 'scheduler_patience'}
    unknown = set(run_overrides) - allowed
    if unknown:
        raise ValueError(f'Unsupported run override fields: {sorted(unknown)}')
    applied = {}
    for field in sorted(allowed):
        if field not in run_overrides:
            continue
        value = int(run_overrides[field])
        if value <= 0:
            raise ValueError(f'{field} override must be positive, got {value}')
        applied[field] = {'old': getattr(args, field), 'new': value}
        setattr(args, field, value)
    return override_path, applied


def _set_plateau_bad_epochs(scheduler, minimum=1, force=False):
    """Handle both raw PyTorch and Accelerate-wrapped plateau schedulers."""
    if scheduler is None:
        return
    raw_scheduler = getattr(scheduler, 'scheduler', scheduler)
    if not hasattr(raw_scheduler, 'num_bad_epochs'):
        return
    if force or raw_scheduler.num_bad_epochs < minimum:
        raw_scheduler.num_bad_epochs = minimum


def _recover_prior_best_metrics(run_dir, args, resume_checkpoint):
    """Recover checkpoint-selection history when resuming an older checkpoint."""
    best_val_loss = float(resume_checkpoint.get('best_val_loss', math.inf))
    best_epoch = int(resume_checkpoint.get('best_epoch', 0))
    default_inference = math.inf if args.inference_earlystop_goal == 'min' else 0.0
    best_val_inference_value = float(resume_checkpoint.get('best_val_inference_value', default_inference))
    best_val_inference_epoch = int(resume_checkpoint.get('best_val_inference_epoch', 0))
    if all(key in resume_checkpoint for key in (
        'best_val_loss', 'best_epoch', 'best_val_inference_value', 'best_val_inference_epoch'
    )):
        return best_val_loss, best_epoch, best_val_inference_value, best_val_inference_epoch

    log_path = os.path.join(run_dir, 'LogFile.log')
    if not os.path.isfile(log_path):
        return best_val_loss, best_epoch, best_val_inference_value, best_val_inference_epoch
    val_pattern = re.compile(r'Epoch (\d+): Validation loss ([0-9.eE+-]+)')
    inf_pattern = re.compile(r'Epoch (\d+): Val inference rmsds_lt2 ([0-9.eE+-]+) rmsds_lt5 ([0-9.eE+-]+)')
    with open(log_path, 'r', errors='replace') as handle:
        for line in handle:
            val_match = val_pattern.search(line)
            if val_match:
                epoch, value = int(val_match.group(1)), float(val_match.group(2))
                if value <= best_val_loss:
                    best_val_loss, best_epoch = value, epoch
            inf_match = inf_pattern.search(line)
            if inf_match:
                epoch = int(inf_match.group(1))
                values = {
                    'valinf_rmsds_lt2': float(inf_match.group(2)),
                    'valinf_rmsds_lt5': float(inf_match.group(3)),
                }
                if args.inference_earlystop_metric not in values:
                    continue
                value = values[args.inference_earlystop_metric]
                improved = (value <= best_val_inference_value if args.inference_earlystop_goal == 'min'
                            else value >= best_val_inference_value)
                if improved:
                    best_val_inference_value, best_val_inference_epoch = value, epoch
    return best_val_loss, best_epoch, best_val_inference_value, best_val_inference_epoch


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader,
          t_to_sigma, run_dir, accelerator, start_epoch=0, resume_checkpoint=None):
    if resume_checkpoint is not None:
        best_val_loss, best_epoch, best_val_inference_value, best_val_inference_epoch = \
            _recover_prior_best_metrics(run_dir, args, resume_checkpoint)
    else:
        best_val_loss = math.inf
        best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
        best_epoch = 0
        best_val_inference_epoch = 0
    
    # Progressive unfreezing configuration
    progressive_unfreeze_enabled = args.use_adaptive_transfer and args.freeze_backbone and getattr(args, 'progressive_unfreeze', False)
    unfreeze_schedule = getattr(args, 'unfreeze_schedule', None)  # e.g., "20:surface_conv_layers.0,40:lig_to_surface_conv_layers"
    
    # Parse unfreeze schedule: epoch:layer1,layer2;epoch2:layer3
    unfreeze_plan = {}
    if progressive_unfreeze_enabled and unfreeze_schedule:
        for item in unfreeze_schedule.split(';'):
            if ':' in item:
                epoch_str, layers_str = item.split(':', 1)
                epoch = int(epoch_str.strip())
                layers = [l.strip() for l in layers_str.split(',')]
                unfreeze_plan[epoch] = layers
        if accelerator.is_local_main_process:
            logger.info(f"Progressive unfreezing enabled with schedule: {unfreeze_plan}")
    
    loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, no_torsion=args.no_torsion)
    if accelerator.is_local_main_process:
        logger.info("Starting training...")
        
        logger.info('Load val inference dataset ...')
    val_inference_datalist = val_loader.dataset.get_complexs_list(args.num_inference_complexes)
    if accelerator.is_local_main_process:
        logger.info(f'Size of dataset is : {len(val_inference_datalist)}.')
    _set_plateau_bad_epochs(scheduler, force=(start_epoch == 0))
    if accelerator.is_local_main_process and start_epoch > 0:
        logger.info(
            f"Resuming one-stage training at epoch {start_epoch}/{args.n_epochs}; "
            f"prior best val={best_val_loss} (epoch {best_epoch}), "
            f"prior best inference={best_val_inference_value} (epoch {best_val_inference_epoch})"
        )
    for epoch in range(start_epoch, args.n_epochs):
        # Progressive unfreezing: check if we should unfreeze more layers
        if progressive_unfreeze_enabled and epoch in unfreeze_plan:
            layers_to_unfreeze = unfreeze_plan[epoch]
            if accelerator.is_local_main_process:
                logger.info(f"Epoch {epoch}: Progressive unfreezing layers: {layers_to_unfreeze}")
            for layer_pattern in layers_to_unfreeze:
                for name, param in model.named_parameters():
                    if layer_pattern in name and not param.requires_grad:
                        param.requires_grad = True
                        if accelerator.is_local_main_process:
                            logger.info(f"  Unfrozen: {name}")
            
            # Reinitialize optimizer with new trainable parameters
            optimizer, scheduler = get_optimizer_and_scheduler(args, model, accelerator, scheduler_mode='max')
            optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                optimizer, train_loader, val_loader, scheduler)
            
            if accelerator.is_local_main_process:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"After unfreezing: Trainable parameters: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)")
        if accelerator.is_local_main_process:
            if epoch % 5 == 0: logger.info(f"Run name: {args.run_name}")
        logs = {}
        #################trainging ########################
        # logger.info('model intance',isinstance(model,TensorProductEnergyModel))
        train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn,accelerator,ema_weights)
        # accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"epoch【{epoch}】@{nowtime} --> train_metric=")
            #for pretrain
            # logger.info("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  mask {:.4f}"
            #     .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
            #             train_losses['tor_loss'],train_losses['mask_loss']), flush=True)
            logger.info("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
                .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                        train_losses['tor_loss']), flush=True)
            if args.task_aligned_aux:
                logger.info(
                    "Epoch {}: Task-aligned aux {:.4f}  optimized total {:.4f}  aux_weight {:.3f}"
                    .format(epoch, train_losses['task_aux_loss'], train_losses['optimized_loss'],
                            args.task_aux_weight)
                )
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        ema_weights.store(model.parameters())
        if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        ############### trainging end#######################

        val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, accelerator,args.test_sigma_intervals,model_type=args.model_type)
        #####################
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"epoch【{epoch}】@{nowtime} --> eval_metric=")
            #for pretrain
            # logger.info("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}  mask {:.4f}"
            #     .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss'], val_losses['mask_loss']))
            logger.info("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
                .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss']))
        if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0 and (epoch + 1) > args.skip_inference_freq:
            
            inf_metrics = inference_epoch_parallel(model, val_inference_datalist, device, t_to_sigma, args,accelerator)
            if accelerator.is_local_main_process:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"epoch【{epoch}】@{nowtime} --> inference_metric=")
                logger.info("Epoch {}: Val inference rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                    .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
                
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if not args.use_ema: ema_weights.copy_to(model.parameters())
        accelerator.wait_for_everyone()
        # ema weight state dict
        unwrapped_model = accelerator.unwrap_model(model)
        ema_state_dict = copy.deepcopy(unwrapped_model.state_dict() if device.type == 'cuda' else unwrapped_model.state_dict())
        # last model weight state dict
        ema_weights.restore(model.parameters())
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # ema_state_dict = copy.deepcopy(unwrapped_model.state_dict() if device.type == 'cuda' else unwrapped_model.state_dict())
        state_dict = unwrapped_model.state_dict() if device.type == 'cuda' else unwrapped_model.state_dict()
        
        logs.update({'train_' + k: v for k, v in train_losses.items()})
        logs.update({'val_' + k: v for k, v in val_losses.items()})
        logs['current_lr'] = optimizer.param_groups[0]['lr']
        
        if args.wandb and accelerator.is_local_main_process:
            wandb.log(logs, step=epoch + 1)
       

        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            if accelerator.is_local_main_process:
                torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
            
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            if accelerator.is_local_main_process:
                torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))
        
        if scheduler and (epoch + 1) % args.val_inference_freq == 0 and (epoch + 1) > args.skip_inference_freq:
            if args.val_inference_freq is not None and (epoch + 1) > args.skip_inference_freq:

                scheduler.step(best_val_inference_value)
                
            else:

                scheduler.step(-1*val_losses['loss'])
            _set_plateau_bad_epochs(scheduler, minimum=accelerator.num_processes)

        if accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            # unwrapped_optimizer = accelerator.unwrap_model(optimizer)
            torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'ema_weights': ema_weights.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'best_val_inference_value': best_val_inference_value,
            'best_val_inference_epoch': best_val_inference_epoch,
        }, os.path.join(run_dir, 'last_model.pt'))
    if accelerator.is_local_main_process:

        logger.info("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
        logger.info("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))
    if args.wandb:
        wandb.finish()

# from accelerate.utils import DummyOptim, DummyScheduler, set_seed
def main_function():
    import typing
    args = parse_train_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            elif isinstance(value, typing.Dict):
                arg_dict[key] = value['value']
            else:
                arg_dict[key] = value
        # args.config = args.config.name 
    run_override = _apply_run_overrides(args)
    set_seed(int(args.seed))
    # logger.info(args)
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.add(os.path.join(run_dir,'LogFile.log'), rotation='100 MB')
    if run_override is not None:
        logger.info(f'Applied run overrides from {run_override[0]}: {run_override[1]}')
    logger.info(f'Args:{args}')
    if accelerator.is_local_main_process:
        # os.makedirs(args.log_dir, exist_ok=True)
        # args.run_name =args.run_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if args.wandb:
            wandb.login(key = '176158d39a3001dfb0b445034525f33e619ce092')
            
            wandb.init(
                entity='luollewyn',
                settings=wandb.Settings(start_method="fork"),
                project=args.project,
                name=args.run_name ,
                dir = args.wandb_dir,
                config=args
            )
            # wandb.log({'numel': numel})
    # args.run_name = args.run_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)
    # logger.info(" t_to_sigma: ",  t_to_sigma)
    model = get_model(args, device, t_to_sigma=t_to_sigma,model_type = args.model_type)

    #################################################
    restart_checkpoint = None
    restart_checkpoint_path = None
    if args.restart_dir:
        restart_checkpoint, restart_checkpoint_path = load_restart_weights(model, args, default_file='best_model.pt')
    
    #################################################
    # Adaptive transfer strategy: Use layered learning rate strategy
    if args.use_adaptive_transfer:
        surface_backbone_keys = [
            'surface_node_embedding',  # Surface node encoding (backbone)
            'surface_edge_embedding',   # Surface edge encoding (backbone)
            'surface_conv_layers',      # Surface convolution layers (backbone)
            'lig_to_surface_conv_layers',  # Ligand to surface layers (backbone)
            'surface_to_lig_conv_layers',  # Surface to ligand layers (backbone)
            'lig_node_embedding',       # Ligand encoding (shared backbone)
            'lig_edge_embedding',       # Ligand edge encoding (shared backbone)
            'lig_conv_layers',          # Ligand convolution layers (shared backbone)
            'cross_edge_embedding',     # Cross edge encoding (shared backbone)
        ]
        
        nucleic_layer_keys = [
            'rec_node_embedding',      # Nucleic acid residue node encoding
            'rec_edge_embedding',       # Nucleic acid residue edge encoding
            'rec_conv_layers',          # Nucleic acid residue convolution layers
            'residue_to_surface_conv_layers',  # Residue to surface interaction
            'nuc_feat_proj',            # Nucleic acid feature projection
            'nuc_feat_gate',            # Nucleic acid feature gating
        ]
        
        if args.freeze_backbone:
            logger.info("Freezing surface-related backbone layers completely, keeping nucleic-acid layers and adapters trainable...")
            # Freeze surface-related backbone layers
            frozen_count = 0
            for name, param in model.named_parameters():
                if any(key in name for key in surface_backbone_keys) and 'adapter' not in name.lower():
                    param.requires_grad = False
                    frozen_count += 1
            
            # Ensure nucleic-acid layers and adapters are trainable
            for name, param in model.named_parameters():
                if any(key in name for key in nucleic_layer_keys) or 'adapter' in name.lower() or 'scalar_gate' in name.lower():
                    param.requires_grad = True
            
            # Unfreeze specific layers if requested
            if args.unfreeze_layers:
                unfreeze_list = [layer.strip() for layer in args.unfreeze_layers.split(',') if layer.strip()]
                for layer_name in unfreeze_list:
                    for name, param in model.named_parameters():
                        if layer_name in name:
                            param.requires_grad = True
                            logger.info(f"Unfreezing layer: {name}")
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Frozen {frozen_count} surface-related backbone parameters")
            logger.info(f"Trainable parameters: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)")
        else:
            logger.info("Using layered learning rate strategy: 50%% LR for surface layers, base LR for other layers...")
            surface_lr_ratio = getattr(args, 'surface_lr_ratio', 0.5)
            logger.info(f"Base learning rate: {args.lr:.6f}")
            logger.info(f"Surface layers will use {surface_lr_ratio * 100:.0f}% of base LR (LR={args.lr * surface_lr_ratio:.6f})")
            logger.info(f"Nucleic-acid layers will use base LR (LR={args.lr:.6f})")
            logger.info(f"Adapter layers will use base LR (LR={args.lr:.6f})")
            # All layers are trainable, learning rates will be set in optimizer
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"All parameters trainable with layered LR: {trainable_params}/{total_params} (100%)")
    
    #################################################
    # Initialize EMA after model structure is finalized (after loading pretrained weights and freezing)
    optimizer, scheduler = get_optimizer_and_scheduler(args,model, accelerator,scheduler_mode='max')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)
    
    #################################################
    model = accelerator.prepare(model)
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    optimizer,train_loader, val_loader, scheduler)

    resume_full_state = os.environ.get('SURFNA_RESUME_FULL_STATE', '0') == '1'
    start_epoch = 0
    if resume_full_state:
        if not isinstance(restart_checkpoint, dict):
            raise ValueError('SURFNA_RESUME_FULL_STATE=1 requires a structured checkpoint')
        missing = {'epoch', 'optimizer', 'ema_weights'} - set(restart_checkpoint)
        if missing:
            raise ValueError(f'Full-state resume checkpoint is missing: {sorted(missing)}')
        optimizer.load_state_dict(restart_checkpoint['optimizer'])
        ema_weights.load_state_dict(restart_checkpoint['ema_weights'], device)
        if restart_checkpoint.get('scheduler') is not None and scheduler is not None:
            scheduler.load_state_dict(restart_checkpoint['scheduler'])
        start_epoch = int(restart_checkpoint['epoch']) + 1
        if start_epoch >= args.n_epochs:
            raise ValueError(f'Resume epoch {start_epoch} is not below target n_epochs={args.n_epochs}')
        logger.info(
            f'Loaded full training state from {restart_checkpoint_path}; '
            f'next_epoch={start_epoch}; optimizer_lr={[group["lr"] for group in optimizer.param_groups]}'
        )

    numel = sum([p.numel() for p in model.parameters()])
    if accelerator.is_local_main_process:
        logger.info(f'Model with {numel} parameters')
    # record parameters
    # run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device
    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader,
          t_to_sigma, run_dir, accelerator, start_epoch=start_epoch,
          resume_checkpoint=restart_checkpoint if resume_full_state else None)
    # wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device

    # accelerator = Accelerator(mixed_precision=mixed_precision)
    logger.info(f'device {str(accelerator.device)} is used!')
    logger.info(f"Total processes: {accelerator.state.num_processes}")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
