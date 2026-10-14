"""Public training API for discrete diffusion models."""

import os

import hydra
import lightning as L
import omegaconf
import torch

from .callbacks.rng_state import RngStateCallback
from .data import get_dataloaders, get_tokenizer
from .data.resumable import ResumableLoaderDataModule
from .data.resume_skip import (
  ResumeAlignmentCheck, read_resume_position, with_resume_skip)
from .parallel import train_pipeline_1f1b
from . import utils


def _disable_torch_compile(fn):
  compiler = getattr(torch, 'compiler', None)
  disable = getattr(compiler, 'disable', None) if compiler is not None else None
  if disable is None:
    disable = torch._dynamo.disable
  return disable(fn)


def train(config):
  """Main training API.
  
  Args:
    config: Hydra DictConfig or config object with training parameters.
    
  Returns:
    None. Model checkpoints are saved according to config.checkpointing.
  """
  # Set matmul precision to 'high' (TF32) to match FlexMDM
  torch.set_float32_matmul_precision("high")
  
  logger = utils.get_logger(__name__)
  logger.info('Starting Training.')

  if omegaconf.OmegaConf.select(config, 'parallel.pipeline.enabled', default=False):
    logger.info('Dispatching to the dedicated 1F1B pipeline trainer.')
    train_pipeline_1f1b(config)
    return
  
  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  
  # Ensure dataset processing happens on rank 0 first
  fabric = L.Fabric(num_nodes=config.trainer.num_nodes,
                    devices=config.trainer.devices,
                    accelerator='cuda')
  fabric.launch()
  with fabric.rank_zero_first():
    train_ds, valid_ds = get_dataloaders(config, tokenizer)
  fabric.barrier()
  del fabric
  
  # WandB logger
  wandb_logger = L.pytorch.loggers.WandbLogger(
    config=omegaconf.OmegaConf.to_object(config), **config.wandb
  ) if config.get('wandb', None) is not None else None

  # Resume checkpoint path
  ckpt_path = config.checkpointing.resume_ckpt_path if (
    config.checkpointing.resume_from_ckpt and 
    config.checkpointing.resume_ckpt_path is not None and 
    utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
  ) else None

  # Lightning callbacks
  callbacks = [hydra.utils.instantiate(cb) for _, cb in config.callbacks.items()] if 'callbacks' in config else []

  if omegaconf.OmegaConf.select(config, 'training.save_rng_state', default=True):
    # Lightning checkpoints carry no RNG state; save each rank's alongside.
    callbacks.append(RngStateCallback(
      dirpath=os.path.join(
        config.checkpointing.save_dir, 'checkpoints', 'rng_states'),
      reseed_every_step=bool(omegaconf.OmegaConf.select(
        config, 'training.rng_reseed_every_step', default=False)),
      reseed_from_step=int(omegaconf.OmegaConf.select(
        config, 'training.rng_reseed_from_step', default=0))))

  # Without exact_resume, Lightning restores its counters but restarts the
  # epoch's data order from the beginning.  Skip what the checkpoint consumed.
  exact_data_resume = bool(config.loader.get('exact_resume', False))
  resume_position = None
  if (ckpt_path is not None and not exact_data_resume
      and not config.training.get('validate_only', False)
      and omegaconf.OmegaConf.select(
        config, 'training.resume_skip_seen_batches', default=True)):
    if isinstance(train_ds.sampler, torch.utils.data.RandomSampler):
      resume_position = read_resume_position(
        ckpt_path, int(config.trainer.accumulate_grad_batches))
      _, skip_batches, resume_step = resume_position
      callbacks.append(ResumeAlignmentCheck(
        expected_batch_idx=skip_batches, expected_global_step=resume_step))
    else:
      logger.warning(
        'Resuming with a %s train sampler: already-seen batches are not '
        'skipped.', type(train_ds.sampler).__name__)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    model = algo_cls.load_from_checkpoint(
      config.training.finetune_path, tokenizer=tokenizer, config=config)
  else:
    model = algo_cls(config, tokenizer=tokenizer)

  # The rho/sigma probe loader must be independent from the loader Lightning
  # iterates for optimization.  See data.loaders._make_scion_trace_dataloader.
  model._scion_trace_dataloader = getattr(
    train_ds, '_scion_trace_dataloader', None)

  # Torch compile: auto compiles algorithms with compile_for_training.
  compile_mode = str(omegaconf.OmegaConf.select(
    config, 'training.torch_compile', default='auto')).strip().lower()
  if compile_mode not in {'auto', 'true', 'false'}:
    raise ValueError(
      "training.torch_compile must be auto, true or false, got "
      f"{compile_mode!r}.")
  compile_model = compile_mode == 'true' or (
    compile_mode == 'auto' and hasattr(model, 'compile_for_training'))
  if compile_model:
    if hasattr(model, 'compile_for_training'):
      logger.info('Compiling model layers and loss in place with torch.compile.')
      model.compile_for_training()
    else:
      # Keep Lightning logging out of Dynamo
      logger.info('Disabling Lightning logging methods inside torch.compile.')
      model.log = _disable_torch_compile(model.log)
      model.log_dict = _disable_torch_compile(model.log_dict)
      logger.info('Compiling LightningModule with torch.compile.')
      model = torch.compile(model)

  if config.training.get('fault_tolerant', False):
    os.environ.setdefault('PL_FAULT_TOLERANT_TRAINING', '1')

  trainer_kwargs = omegaconf.OmegaConf.to_container(
    config.trainer, resolve=True)
  if exact_data_resume:
    # Keep our checkpointable sampler; Lightning's automatic replacement is
    # not stateful and therefore cannot preserve the exact sample cursor.
    trainer_kwargs['use_distributed_sampler'] = False
  trainer = L.Trainer(
    **trainer_kwargs, default_root_dir=os.getcwd(), callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy), logger=wandb_logger)

  if resume_position is not None:
    # The Trainer knows the DDP world size and rank the sampler is built for.
    resume_epoch, skip_batches, _ = resume_position
    train_ds = with_resume_skip(
      train_ds, num_replicas=trainer.world_size, rank=trainer.global_rank,
      epoch=resume_epoch, skip_batches=skip_batches)

  if config.training.get('validate_only', False):
    if ckpt_path is None:
      raise ValueError(
        'training.validate_only=true requires a valid '
        'checkpointing.resume_ckpt_path pointing at an existing checkpoint.')
    logger.info(f'Validation-only mode: running trainer.validate() on {ckpt_path}')
    trainer.validate(model, dataloaders=valid_ds, ckpt_path=ckpt_path)
    return

  if exact_data_resume:
    data_module = ResumableLoaderDataModule(train_ds, valid_ds)
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
  else:
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

  # ModelCheckpoint writes last.ckpt only at its save points (every
  # validation, every_n_train_steps), so it can lag the final optimizer step
  # by up to a validation interval.  Overwrite it with the final weights.
  # All ranks call save_checkpoint; only rank 0 writes.
  for checkpoint_cb in trainer.checkpoint_callbacks:
    if checkpoint_cb.save_last and checkpoint_cb.dirpath:
      last_path = os.path.join(
        checkpoint_cb.dirpath,
        checkpoint_cb.CHECKPOINT_NAME_LAST + checkpoint_cb.FILE_EXTENSION)
      trainer.save_checkpoint(last_path)
      logger.info('Saved final weights (global_step=%d) to %s',
                  trainer.global_step, last_path)
      break

  if config.training.get('validate_after_training', False):
    logger.info(
      'Training complete at global_step=%d; validating final in-memory weights.',
      trainer.global_step)
    trainer.validate(model, dataloaders=valid_ds, ckpt_path=None)
