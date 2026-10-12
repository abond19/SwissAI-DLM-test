"""Top-level loader API for discrete diffusion training."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import datasets
import tokenizers
import torch
import transformers
from torch.utils.data.distributed import DistributedSampler

from .. import utils
from .datasets import (
    generate_fresh_random_token_dataset,
    generate_in_memory_synthetic_token_dataset,
    generate_synthetic_dataset,
    get_lambada_test_dataset,
    get_text8_dataset,
)
from .processing import (
    _apply_detokenizer,
    _group_texts,
    lm1b_detokenizer,
    lambada_detokenizer,
    ptb_detokenizer,
    scientific_papers_detokenizer,
    wt_detokenizer,
)
from .tokenizers import SyntheticTokenizer, Text8Tokenizer
from .flex_chunking import chunk_documents
from .lazy_dataset import LazyDiskDataset
from .resumable import StatefulDistributedSampler

LOGGER = utils.get_logger(__name__)

__all__ = [
    "get_tokenizer",
    "get_dataset",
    "get_dataloaders",
]


def get_dataset(dataset_name,
                tokenizer,
                wrap,
                mode,
                cache_dir,
                insert_eos=True,
                insert_special_tokens=True,
                block_size=1024,
                num_proc=len(os.sched_getaffinity(0)),
                streaming=False,
                revision: Optional[str] = None,
                min_length: int = 0,
                chunking: str = "none",
                pretok_tokens_column: Optional[str] = None,
                pretok_local_dir: Optional[str] = None,
                pretok_pack_num_proc: Optional[int] = None,
                pretok_pack_writer_batch_size: Optional[int] = None,
                synthetic_dataset_size: Optional[int] = None,
                synthetic_unique_samples: int = 1024,
                synthetic_seed: int = 1234):
  chunking_mode = (chunking or "none").lower()
  if chunking_mode not in {"none", "double_newline"}:
    raise ValueError(f"Unsupported chunking mode: {chunking_mode}")
  if wrap and chunking_mode != "none":
    raise ValueError("Delimiter-based chunking only applies when wrap=False.")
  eos_tag = ""
  if not insert_eos:
    eos_tag += "_eosFalse"
  if not insert_special_tokens:
    eos_tag += "_specialFalse"
  min_len_tag = f"_min{min_length}" if (min_length and not wrap) else ""
  chunk_tag = "_flexchunk" if (not wrap and chunking_mode != "none") else ""
  if wrap:
    filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat"
  else:
    filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped{chunk_tag}{eos_tag}{min_len_tag}.dat"
  _path = os.path.join(cache_dir, filename)

  if utils.fsspec_exists(_path):
    LOGGER.info("Loading data from: %s", _path)
    return datasets.load_from_disk(_path).with_format("torch")
  LOGGER.info("Generating new data at: %s", _path)
  LOGGER.info("streaming=%s", streaming)

  crop_train = dataset_name == "text8-crop"
  if mode == "train" and crop_train:
    block_size *= 2

  if dataset_name == "wikitext103":
    dataset = datasets.load_dataset(
      "wikitext",
      name="wikitext-103-raw-v1",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "wikitext2":
    dataset = datasets.load_dataset(
      "wikitext",
      name="wikitext-2-raw-v1",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "ptb":
    dataset = datasets.load_dataset(
      "ptb_text_only",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "lambada":
    dataset = get_lambada_test_dataset()
  elif dataset_name == "text8":
    assert wrap
    assert revision is None
    dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
  elif dataset_name == "text8-crop":
    assert revision is None
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == "openwebtext-train":
    dataset = datasets.load_dataset(
      "openwebtext",
      split="train[:-100000]",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)
  elif dataset_name == "openwebtext-valid":
    dataset = datasets.load_dataset(
      "openwebtext",
      split="train[-100000:]",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)
    
  elif dataset_name == "slimpajama":
    
    dataset = datasets.load_dataset(
      "MBZUAI-LLM/SlimPajama-627B-DC",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)




    
  elif dataset_name == "scientific_papers_arxiv":
    dataset = datasets.load_dataset(
      "scientific_papers", "arxiv",
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "scientific_papers_pubmed":
    dataset = datasets.load_dataset(
      "scientific_papers", "pubmed",
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "ag_news":
    dataset = datasets.load_dataset(
      "ag_news",
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "synthetic":
    assert streaming
    assert wrap
    dataset = generate_synthetic_dataset(
      train_dataset_size=100000,
      validation_dataset_size=1024,
      seq_len=32,
      vocab_size=256,
    )
  elif dataset_name == "synthetic-fixed":
    dataset_size = int(
      synthetic_dataset_size
      if synthetic_dataset_size is not None
      else (100000 if mode == "train" else 1024))
    seed_offset = 0 if mode == "train" else 1
    return generate_in_memory_synthetic_token_dataset(
      dataset_size=dataset_size,
      num_unique_samples=synthetic_unique_samples,
      seq_len=block_size,
      vocab_size=len(tokenizer),
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      mask_token_id=tokenizer.mask_token_id,
      seed=int(synthetic_seed) + seed_offset)
  elif dataset_name == "synthetic-random":
    # Fresh random tokens for every index (no fixed pool); used by the speed
    # benchmark so it runs the real training path without any dataset on disk.
    dataset_size = int(
      synthetic_dataset_size
      if synthetic_dataset_size is not None
      else (1_000_000 if mode == "train" else 4096))
    seed_offset = 0 if mode == "train" else 1
    return generate_fresh_random_token_dataset(
      dataset_size=dataset_size,
      seq_len=block_size,
      vocab_size=len(tokenizer),
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      mask_token_id=tokenizer.mask_token_id,
      seed=int(synthetic_seed) + seed_offset)
  elif dataset_name in {"nemotron-cc-pretok-train", "nemotron-cc-pretok-valid"}:
    if wrap:
      raise ValueError(
        "nemotron-cc-pretok datasets are already tokenized and must use wrap=False.")

    split_expr = (
      "train[:-100000]"
      if dataset_name == "nemotron-cc-pretok-train"
      else "train[-100000:]")
    local_parquets = []
    local_dir = None
    if pretok_local_dir:
      local_dir = Path(
        os.path.expandvars(os.path.expanduser(pretok_local_dir)))
      if not local_dir.is_dir():
        raise ValueError(
          f"Configured pretok_local_dir does not exist or is not a directory: "
          f"{local_dir}")
      local_parquets = sorted(str(p) for p in local_dir.rglob("*.parquet"))
      if not local_parquets:
        raise ValueError(
          f"No parquet files found under pretok_local_dir: {local_dir}")
      LOGGER.info(
        "Using local nemotron pretokenized data from %s (%d parquet files).",
        local_dir, len(local_parquets))

    if local_parquets:
      data = datasets.load_dataset(
        "parquet",
        data_files={"train": local_parquets},
        split=split_expr,
        cache_dir=cache_dir,
        streaming=streaming,
        revision=revision,
        num_proc=None if streaming else num_proc,
        trust_remote_code=True)
    else:
      LOGGER.info(
        "No pretok_local_dir set for nemotron pretokenized data. "
        "Falling back to HF: %s",
        "dvruette/gidd-nemotron-cc-pretok")
      data = datasets.load_dataset(
        "dvruette/gidd-nemotron-cc-pretok",
        split=split_expr,
        cache_dir=cache_dir,
        streaming=streaming,
        revision=revision,
        trust_remote_code=True)

    preferred_cols = ["input_ids", "token_ids", "tokens", "ids"]
    column_names = data.column_names
    if pretok_tokens_column is not None:
      token_col = pretok_tokens_column
      if token_col not in column_names:
        raise ValueError(
          f"Configured pretok_tokens_column={token_col!r} not found in dataset "
          f"columns={column_names}")
    else:
      token_col = next((c for c in preferred_cols if c in column_names), None)
      if token_col is None:
        raise ValueError(
          "Could not infer token column for pretokenized dataset. "
          f"Available columns={column_names}; set data.pretok_tokens_column.")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
      raise ValueError(
        "Tokenizer must define eos_token_id for nemotron pretokenized packing.")

    tokenizer_vocab_size = int(len(tokenizer))

    # Match original gidd-easydel flow: concatenate token rows with EOS
    # separators, then repack to fixed sequence length.
    def _to_token_rows(example):
      batch_ids = example[token_col]
      out_ids = []
      max_seen_id = -1
      for ids in batch_ids:
        row = list(ids)
        if row:
          row_max = max(row)
          if row_max > max_seen_id:
            max_seen_id = row_max
        row.append(int(eos_id))
        out_ids.append(row)
      if max_seen_id >= tokenizer_vocab_size:
        raise ValueError(
          "Pretokenized dataset contains token ids that exceed tokenizer vocab size: "
          f"max_token_id={max_seen_id}, tokenizer_vocab_size={tokenizer_vocab_size}, "
          f"tokenizer={getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)!r}, "
          f"dataset={dataset_name}, token_column={token_col!r}. "
          "Use the same tokenizer that was used to pretokenize this dataset.")
      return {"input_ids": out_ids}

    map_kwargs = {"batched": True}
    if not streaming:
      map_kwargs.update(
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Preparing pretokenized rows")
    token_rows = data.map(_to_token_rows, **map_kwargs)
    remove_cols = [c for c in token_rows.column_names if c != "input_ids"]
    if remove_cols:
      token_rows = token_rows.remove_columns(remove_cols)

    group_texts = functools.partial(
      _group_texts,
      block_size=block_size,
      bos=tokenizer.bos_token_id,
      eos=eos_id,
      insert_special_tokens=False)
    pack_num_proc = (
      num_proc if pretok_pack_num_proc is None else int(pretok_pack_num_proc))
    if pack_num_proc < 1:
      raise ValueError(
        f"pretok_pack_num_proc must be at least 1, got {pack_num_proc}")
    pack_map_kwargs = {
      "batched": True,
      "num_proc": pack_num_proc,
      "load_from_cache_file": True,
      "desc": "Packing token rows",
    }
    if pretok_pack_writer_batch_size is not None:
      pack_writer_batch_size = int(pretok_pack_writer_batch_size)
      if pack_writer_batch_size < 1:
        raise ValueError(
          "pretok_pack_writer_batch_size must be at least 1, got "
          f"{pack_writer_batch_size}")
      pack_map_kwargs["writer_batch_size"] = pack_writer_batch_size
    LOGGER.info(
      "Packing nemotron rows with num_proc=%d, writer_batch_size=%s.",
      pack_num_proc,
      pack_map_kwargs.get("writer_batch_size", "datasets default"))
    if streaming:
      processed = token_rows.map(group_texts, batched=True)
    else:
      processed = token_rows.map(group_texts, **pack_map_kwargs)
      processed.save_to_disk(_path)
    return processed.with_format("torch")
  
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming,
      trust_remote_code=True,
      revision=revision)

  if dataset_name in ["lambada", "openwebtext-train",
                      "openwebtext-valid"]:
    data = dataset
  else:
    data = dataset[mode]
    if dataset_name == "synthetic":
      return data

  if dataset_name.startswith("wikitext"):
    detokenizer = wt_detokenizer
  elif dataset_name == "lm1b":
    detokenizer = lm1b_detokenizer
  elif dataset_name == "ptb":
    detokenizer = ptb_detokenizer
  elif dataset_name == "lambada":
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith("scientific_papers"):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  EOS = tokenizer.eos_token_id
  BOS = tokenizer.bos_token_id

  tokenizer.padding_side = "right"
  tokenizer.truncation_side = "right"

  use_chunking = chunking_mode != "none"
  if use_chunking:
    if chunking_mode == "double_newline":
      delimiter_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
    else:
      delimiter_tokens = []
    if not delimiter_tokens:
      raise ValueError(
        "Tokenizer did not produce any tokens for the specified chunking delimiter.")
  else:
    delimiter_tokens = []

  def preprocess_and_tokenize(example):
    if dataset_name == "ptb":
      text = example["sentence"]
    elif "scientific_papers" in dataset_name:
      text = example["article"]
    else:
      text = example["text"]
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)
    if use_chunking:
      return chunk_documents(
        tokenizer,
        text,
        max_length=block_size,
        delimiter_tokens=delimiter_tokens,
        add_special_tokens=insert_special_tokens)
    if wrap:
      tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False)
      if insert_eos:
        tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
    else:
      tokens = tokenizer(
        text,
        max_length=block_size,
        padding="max_length",
        truncation=True,
        add_special_tokens=insert_special_tokens,
        return_attention_mask=True,
        return_token_type_ids=True)
    return tokens

  map_kwargs = {
    "batched": True,
  }
  if use_chunking:
    map_kwargs["remove_columns"] = ["text"]
  if not streaming:
    map_kwargs.update(
      num_proc=num_proc,
      load_from_cache_file=True,
      desc="Tokenizing")
  tokenized_dataset = data.map(
    preprocess_and_tokenize,
    **map_kwargs)
  if dataset_name == "ptb":
    tokenized_dataset = tokenized_dataset.remove_columns("sentence")
  elif "scientific_papers" in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns(
      ["article", "abstract", "section_names"])
  elif dataset_name == "ag_news":
    tokenized_dataset = tokenized_dataset.remove_columns(
      ["text", "label"])
  elif "text" in tokenized_dataset.column_names:
    tokenized_dataset = tokenized_dataset.remove_columns("text")

  if (not wrap) and min_length > 0 and (not streaming):
    def _has_min_length(example):
      mask = example.get("attention_mask", None)
      if mask is None:
        return True
      return sum(mask) >= min_length

    tokenized_dataset = tokenized_dataset.filter(
      _has_min_length,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc="Filtering min length")

  if not wrap:
    if not streaming:
      tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format("torch")

  group_texts = functools.partial(
    _group_texts,
    block_size=block_size,
    bos=BOS,
    eos=EOS,
    insert_special_tokens=insert_special_tokens)
  if streaming:
    chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      num_proc=num_proc,
      load_from_cache_file=True,
      desc="Grouping")
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format("torch")
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == "text8":
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == "bert-base-uncased":
    tokenizer = transformers.BertTokenizer.from_pretrained(
      "bert-base-uncased")
  elif config.data.tokenizer_name_or_path == "synthetic":
    tokenizer = SyntheticTokenizer(vocab_size=256)
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)
  if isinstance(tokenizer, (transformers.GPT2TokenizerFast,
                            transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = (
      tokenizers.processors.BertProcessing(
        (tokenizer.bos_token, tokenizer.bos_token_id),
        (tokenizer.eos_token, tokenizer.eos_token_id)))
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        "Tokenizer must have a bos_token or "
        f"cls_token: {tokenizer}")
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        "Tokenizer must have a eos_token "
        f"or sep_token: {tokenizer}")
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  if getattr(tokenizer, 'mask_token', None) is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
  return tokenizer


def _log_worker_start(worker_id):
  """worker_init_fn: logs the moment a DataLoader worker process actually
  starts running (as opposed to when the DataLoader object is constructed,
  which does not spawn workers -- that happens lazily on first iteration).
  Used to test whether multiprocessing_context='spawn' worker startup is
  the source of the observed idle gap before the first training batch."""
  LOGGER.info(
    "[WORKER_TIMING] worker %d alive, pid=%d", worker_id, os.getpid())


def _maybe_make_spawn_friendly_dataset(dataset, config):
  """Replace a loaded HF cache with a path-only dataset before ``spawn``.

  This is opt-in because it is only useful for map-style packed datasets and
  because worker prefetch is incompatible with the exact-resume contract.
  ``cache_files`` is populated by ``datasets.load_from_disk`` and identifies
  the directory containing ``state.json``.
  """
  enabled = bool(config.loader.get('lazy_spawn_dataset', False))
  mp_context = config.loader.get('multiprocessing_context', None)
  if not enabled or mp_context != 'spawn' or int(config.loader.num_workers) == 0:
    return dataset
  if config.data.streaming:
    raise ValueError('loader.lazy_spawn_dataset requires a map-style dataset.')
  if bool(config.loader.get('exact_resume', False)):
    raise ValueError(
      'loader.lazy_spawn_dataset is a multi-worker path and cannot be used '
      'with loader.exact_resume=true.')
  cache_files = getattr(dataset, 'cache_files', None)
  if not cache_files:
    raise ValueError(
      'loader.lazy_spawn_dataset requires a dataset loaded from a local '
      'Arrow cache (no cache_files found).')
  first_file = cache_files[0].get('filename')
  if not first_file:
    raise ValueError('Could not determine the local Arrow cache path.')
  cache_path = os.path.dirname(first_file)
  if not os.path.isfile(os.path.join(cache_path, 'state.json')):
    raise ValueError(
      f'Expected a datasets.load_from_disk directory, got {cache_path!r}.')
  LOGGER.info(
    '[WORKER_TIMING] replacing loaded HF Dataset with lazy path-only wrapper: %s',
    cache_path)
  return LazyDiskDataset(cache_path, len(dataset))


def _make_scion_trace_dataloader(train_set, config):
  """Make an independent, correctly sharded loader for rho/sigma probes.

  The trace code consumes ``trace_m`` *additional* global batches immediately
  before an optimizer step.  It must not call ``iter(train_loader)``: with
  persistent multiprocessing workers PyTorch resets the active training
  iterator, which can discard prefetched training samples.  Keeping this
  auxiliary loader synchronous also avoids creating another worker pool just
  for a rare measurement (every 500 steps in the rho/sigma sweep).
  """
  trace_enabled = bool(getattr(config.optim, 'trace_enabled', False))
  collect_noise = bool(getattr(
    config.optim, 'trace_collect_noise_stats', False))
  # ``ScionTrace`` has two independent features: legacy per-step tracing and
  # periodic rho/sigma collection.  The latter deliberately keeps
  # ``trace_enabled=false`` to avoid the legacy trace's per-step cloning cost,
  # but it still needs this dedicated loader.
  if not (trace_enabled or collect_noise):
    return None
  if config.data.streaming:
    raise ValueError(
      'SCION rho/sigma tracing requires a map-style training dataset.')

  if torch.distributed.is_available() and torch.distributed.is_initialized():
    num_replicas = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
  else:
    num_replicas = 1
    rank = 0
  # Use a distinct deterministic permutation so trace batches are independent
  # of the optimizer-training stream.  All ranks receive disjoint shards and
  # DDP averages their local gradients into the intended global-batch sample.
  trace_sampler = DistributedSampler(
    train_set,
    num_replicas=num_replicas,
    rank=rank,
    shuffle=True,
    seed=int(config.seed) + 1_000_003,
    drop_last=False)
  trace_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=0,
    pin_memory=False,
    shuffle=False,
    sampler=trace_sampler)
  trace_loader.tokenizer = getattr(train_set, 'tokenizer', None)
  LOGGER.info(
    '[SCION_TRACE] independent trace loader: rank=%d/%d, workers=0, '
    'seed=%d, local_batch=%d', rank, num_replicas,
    int(config.seed) + 1_000_003, int(config.loader.batch_size))
  return trace_loader


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f"Train Batch Size {config.loader.batch_size} "
      f"not divisible by {num_gpus} gpus with accumulation "
      f"{config.trainer.accumulate_grad_batches}.")
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f"Eval Batch Size for {config.eval.batch_size} "
      f"not divisible by {num_gpus}.")
  default_chunking = config.data.get("chunking", "none")
  train_chunking = config.data.get("train_chunking", default_chunking)
  valid_chunking = config.data.get("valid_chunking", default_chunking)
  if skip_train:
    train_set = None
  else:
    train_min_length = config.data.get(
      "train_min_length", config.data.get("min_length", 0))
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode="train",
      wrap=config.data.wrap,
      insert_eos=config.data.insert_train_eos,
      insert_special_tokens=getattr(
        config.data, "insert_train_special", True),
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      revision=config.data.get("train_revision", None),
      min_length=train_min_length,
      chunking=train_chunking,
      pretok_tokens_column=config.data.get("pretok_tokens_column", None),
      pretok_local_dir=config.data.get("pretok_local_dir", None),
      synthetic_dataset_size=config.data.get("synthetic_train_size", None),
      synthetic_unique_samples=config.data.get("synthetic_unique_samples", 1024),
      synthetic_seed=config.data.get("synthetic_seed", 1234))

  if config.data.valid in ["text8", "lm1b", "ag_news"]:
    validation_split = "test"
  else:
    validation_split = "validation"
  if skip_valid:
    valid_set = None
  else:
    valid_min_length = config.data.get(
      "valid_min_length", config.data.get("min_length", 0))
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      insert_eos=config.data.insert_valid_eos,
      insert_special_tokens=getattr(
        config.data, "insert_valid_special", True),
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      revision=config.data.get("valid_revision", None),
      min_length=valid_min_length,
      chunking=valid_chunking,
      pretok_tokens_column=config.data.get("pretok_tokens_column", None),
      pretok_local_dir=config.data.get("pretok_local_dir", None),
      synthetic_dataset_size=config.data.get("synthetic_validation_size", None),
      synthetic_unique_samples=config.data.get("synthetic_unique_samples", 1024),
      synthetic_seed=config.data.get("synthetic_seed", 1234))

  mp_context = config.loader.get('multiprocessing_context', None)
  if skip_train:
    train_loader = None
  else:
    train_set = _maybe_make_spawn_friendly_dataset(train_set, config)
    exact_resume = bool(config.loader.get('exact_resume', False))
    train_sampler = None
    if exact_resume:
      if config.data.streaming:
        raise ValueError(
          "loader.exact_resume=true currently requires a map-style dataset.")
      if int(config.loader.num_workers) != 0:
        raise ValueError(
          "loader.exact_resume=true currently requires loader.num_workers=0 "
          "so worker prefetch cannot advance the saved cursor past the last "
          "completed optimizer batch.")
      train_sampler = StatefulDistributedSampler(
        train_set,
        shuffle=True,
        seed=int(config.seed),
        drop_last=False,
        batch_size=int(config.loader.batch_size))
    LOGGER.info("[WORKER_TIMING] constructing train_loader (workers not "
                "spawned yet -- lazy, happens on first iteration)")
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=(not config.data.streaming) if train_sampler is None else False,
      sampler=train_sampler,
      persistent_workers=config.loader.num_workers > 0,
      multiprocessing_context=mp_context if config.loader.num_workers > 0 else None,
      worker_init_fn=_log_worker_start if config.loader.num_workers > 0 else None)
    train_loader.tokenizer = tokenizer
    # Stored on the ordinary loader only as a hand-off to train.py. Lightning
    # may reconstruct the actual train loader when it installs its distributed
    # sampler, so TrainerBase receives the trace loader directly from the model.
    train_loader._scion_trace_dataloader = _make_scion_trace_dataloader(
      train_set, config)
  if skip_valid:
    valid_loader = None
  else:
    valid_set = _maybe_make_spawn_friendly_dataset(valid_set, config)
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=config.loader.num_workers > 0,
      multiprocessing_context=mp_context if config.loader.num_workers > 0 else None,
      worker_init_fn=_log_worker_start if config.loader.num_workers > 0 else None)
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader
