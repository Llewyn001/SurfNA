import copy
import math
import os
from functools import partial

import wandb
import torch
from accelerate.utils import set_seed
torch.multiprocessing.set_sharing_strategy('file_system')

import resource
# 获取当前限制
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

# 新的软限制不能超过当前的硬限制
new_soft_limit = min(64000, rlimit[1])

# 设置限制
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, rlimit[1]))
import yaml

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training_mdn import train_mdn_epoch, test_mdn_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model, ExponentialMovingAverage
import datetime
from loguru import logger


def _checkpoint_state_dict(checkpoint):
    """Return a plain model state_dict from raw or training checkpoints."""
    if isinstance(checkpoint, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith('module.') for k in state_dict):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


def _adapt_surface_embedding(src_tensor, dst_tensor):
    """Adapt old 3-feature MDN surface embeddings to the SurfNA 4-feature input."""
    adapted = dst_tensor.clone()
    rows = min(src_tensor.shape[0], adapted.shape[0])
    cols = min(src_tensor.shape[1], adapted.shape[1])
    adapted[:rows, :cols] = src_tensor[:rows, :cols]
    return adapted


def load_transfer_weights(model, checkpoint_path, key_filter='all'):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    source = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
    target = model.state_dict()
    loaded, adapted, skipped_shape, skipped_filter, missing = 0, 0, 0, 0, 0
    next_state = {}

    surface_prefixes = (
        'surface_node_embedding', 'surface_edge_embedding', 'surface_conv_layers',
        'surface_rec_cross_edge_embedding', 'residue_to_surface_conv_layers',
    )
    for key, target_tensor in target.items():
        if key_filter == 'surface' and not key.startswith(surface_prefixes):
            next_state[key] = target_tensor
            skipped_filter += 1
            continue
        if key not in source:
            next_state[key] = target_tensor
            missing += 1
            continue
        src_tensor = source[key]
        if src_tensor.shape == target_tensor.shape:
            next_state[key] = src_tensor
            loaded += 1
        elif key == 'surface_node_embedding.linear.weight' and src_tensor.ndim == target_tensor.ndim == 2:
            next_state[key] = _adapt_surface_embedding(src_tensor, target_tensor)
            adapted += 1
        else:
            next_state[key] = target_tensor
            skipped_shape += 1

    model.load_state_dict(next_state, strict=True)
    logger.info(
        f"Loaded transfer weights from {checkpoint_path}; filter={key_filter}; "
        f"loaded={loaded} adapted={adapted} missing={missing} "
        f"skipped_filter={skipped_filter} skipped_shape={skipped_shape}"
    )


def _format_mdn_losses(losses):
    text = (
        f"loss {losses['loss']:.4f} "
        f"inter {losses['mdn_loss_interaction']:.4f} "
        f"lig {losses['mdn_loss_ligand']:.4f} "
        f"atom {losses['atom_types_loss']:.4f} "
        f"bond {losses['bond_types_loss']:.4f} "
        f"res {losses['residue_types_loss']:.4f}"
    )
    if 'pose_bce_loss' in losses:
        text += f" pose_bce {losses['pose_bce_loss']:.4f} pairwise {losses['pairwise_loss']:.4f}"
    if 'rerank_top1_lt2' in losses:
        text += (
            f" top1_lt2 {losses['rerank_top1_lt2']:.3f}"
            f" top1_lt5 {losses['rerank_top1_lt5']:.3f}"
            f" top5_lt2 {losses['rerank_top5_lt2']:.3f}"
            f" oracle_lt2 {losses['rerank_oracle_lt2']:.3f}"
            f" top1_median {losses['rerank_top1_median']:.3f}"
        )
    return text


def train(args, model, optimizer, scheduler,  ema_weights,train_loader, val_loader, t_to_sigma, run_dir,accelerator):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    early_stop_patience = args.mdn_early_stop_patience
    patience_count = 0
    logger.info("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: logger.info("Run name: {}".format(args.run_name))
        logs = {}
        #################trainging ########################
        train_losses = train_mdn_epoch(
            model, train_loader, optimizer, device, accelerator, ema_weights,
            gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
            aux_loss_weight=getattr(args, 'mdn_aux_loss_weight', 0.001)
        )
        # accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"epoch【{epoch}】@{nowtime} --> train_metric=")
            logger.info("Epoch {}: Training {}".format(epoch, _format_mdn_losses(train_losses), flush=True))
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        ema_weights.store(model.parameters())
        if args.use_ema: ema_weights.copy_to(model.parameters()) # load ema parameters into model for running validation and inference
        ############### trainging end#######################
        val_losses = test_mdn_epoch(
            model, val_loader, device, accelerator,args.test_sigma_intervals,
            aux_loss_weight=getattr(args, 'mdn_aux_loss_weight', 0.001)
        )
        #####################
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"epoch【{epoch}】@{nowtime} --> eval_metric=")
            logger.info("Epoch {}: Validation {}".format(epoch, _format_mdn_losses(val_losses)))
            logger.info("Epoch {}: current_lr {:.6g}".format(epoch, optimizer.param_groups[0]['lr']))

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
        if accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            if args.wandb:
                logs.update({'train_' + k: v for k, v in train_losses.items()})
                logs.update({'val_' + k: v for k, v in val_losses.items()})
                logs['current_lr'] = optimizer.param_groups[0]['lr']
                wandb.log(logs, step=epoch + 1)
            
            # if args.inference_earlystop_metric in logs.keys() and \
            #         (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
            #         args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            #     best_val_inference_value = logs[args.inference_earlystop_metric]
            #     best_val_inference_epoch = epoch
            #     torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            #     torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
            patience_count += 1
            val_loss_is_finite = math.isfinite(val_losses['loss'])
            decoy_mode = getattr(args, 'mdn_rerank_mode', 'native_mdn') == 'decoy_supervised'
            if decoy_mode and 'rerank_top1_lt2' in val_losses:
                current_key = (
                    val_losses['rerank_top1_lt2'],
                    val_losses.get('rerank_top1_lt5', 0.0),
                    -val_losses.get('rerank_top1_median', float('inf')),
                )
                best_key = getattr(train, '_best_rerank_key', (-float('inf'), -float('inf'), -float('inf')))
                if current_key >= best_key:
                    patience_count = 0
                    train._best_rerank_key = current_key
                    best_val_loss = val_losses['loss']
                    best_epoch = epoch
                    best_val_inference_value = val_losses['rerank_top1_lt2']
                    best_val_inference_epoch = epoch
                    torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
                    torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))
                    logger.info(f"New best rerank checkpoint at epoch {epoch}: key={current_key}")
            elif val_loss_is_finite and val_losses['loss'] <= best_val_loss:
                patience_count =0
                best_val_loss = val_losses['loss']
                best_epoch = epoch
                torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))
            elif not val_loss_is_finite:
                logger.info(f"Non-finite validation loss at epoch {epoch}; not updating best checkpoint")
            if patience_count >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if scheduler and math.isfinite(val_losses['loss']):
            # MDN/SurfScore does not run diffusion inference during training here.
            # Drive ReduceLROnPlateau with validation MDN loss instead of the
            # placeholder inference value, which is constant when no rerank
            # validation pass is implemented.
            scheduler.step(val_losses['loss'])
        if accelerator.is_local_main_process:
            # accelerator.wait_for_everyone()
            # unwrapped_optimizer = accelerator.unwrap_model(optimizer)
            torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
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
        # logger.info(value['value'])
            else:
                arg_dict[key] = value
        # args.config = args.config.name 
    set_seed(int(args.seed))
    # logger.info(args)
    if os.environ.get('SURFNA_KEEP_RUN_NAME', '0') != '1':
        args.run_name = args.run_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if accelerator.is_local_main_process:
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
    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)
    model = get_model(args, device, t_to_sigma=t_to_sigma,model_type = args.model_type)
    # get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
    #                                 mdn_mode=True)
    optimizer, scheduler = get_optimizer_and_scheduler(args,model, accelerator,scheduler_mode='min')
    ema_weights = ExponentialMovingAverage(model.parameters(),decay=args.ema_rate)
    #################################################
    if getattr(args, 'restart_file', None):
        checkpoint_path = args.restart_file
        if not os.path.isabs(checkpoint_path):
            if not args.restart_dir:
                raise ValueError("--restart_file is relative, so --restart_dir is required.")
            checkpoint_path = os.path.join(args.restart_dir, checkpoint_path)
        load_transfer_weights(model, checkpoint_path, getattr(args, 'restart_key_filter', 'all'))
        # Transfer loading happens after the initial EMA object was created.
        # Rebuild the shadows from the transferred model; otherwise epoch-0
        # validation copies random pre-transfer shadows into the model.
        ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
    elif args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
            if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
            optimizer.load_state_dict(dict['optimizer'])
            model.load_state_dict(dict['model'], strict=True)
            if hasattr(args, 'ema_rate'):
                ema_weights.load_state_dict(dict['ema_weights'], device=device)
            logger.info(f"Restarting from epoch {dict['epoch']}")
        except Exception as e:
            logger.info(f"Exception: {e}")
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.module.load_state_dict(dict, strict=True)
            logger.info("Due to exception had to take the best epoch and no optimiser")
    #################################################
    model = accelerator.prepare(model)
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    optimizer,train_loader, val_loader, scheduler)

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model with {numel} parameters')

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    if accelerator.is_local_main_process:
        os.makedirs(run_dir, exist_ok=True)
        logger.add(os.path.join(run_dir, 'LogFile.log'))
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device
    train(args, model, optimizer, scheduler, ema_weights,train_loader, val_loader, t_to_sigma, run_dir,accelerator)
    # if args.wandb:
    #     wandb.finish()
if __name__ == '__main__':
    from accelerate import Accelerator
    # from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    # accelerator = Accelerator(mixed_precision=mixed_precision)
    logger.info(f'device {str(accelerator.device)} is used!')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()
    # exit()
