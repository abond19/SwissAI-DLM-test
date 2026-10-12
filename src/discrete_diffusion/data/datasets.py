"""Specialized dataset builders re-used in the discrete diffusion loader."""

from __future__ import annotations

import json
import os
import shutil
import urllib
import zipfile

import datasets
import fsspec
import numpy as np
import requests
import torch

from .. import utils

LOGGER = utils.get_logger(__name__)

__all__ = [
    "generate_fresh_random_token_dataset",
    "generate_in_memory_synthetic_token_dataset",
    "generate_synthetic_dataset",
    "get_lambada_test_dataset",
    "get_text8_dataset",
]


class _InMemorySyntheticTokenDataset(torch.utils.data.Dataset):
  """Small in-memory token dataset for throughput diagnostics."""

  def __init__(self, tokens: torch.Tensor, attention_mask: torch.Tensor,
               dataset_size: int):
    self.tokens = tokens
    self.attention_mask = attention_mask
    self.dataset_size = int(dataset_size)

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, idx):
    sample_idx = int(idx) % self.tokens.shape[0]
    return {
      'input_ids': self.tokens[sample_idx],
      'attention_mask': self.attention_mask[sample_idx],
    }


def generate_in_memory_synthetic_token_dataset(
    dataset_size, num_unique_samples, seq_len, vocab_size,
    bos_token_id=None, eos_token_id=None, mask_token_id=None, seed=1234):
  """Generate fixed-length random token samples without disk/tokenization I/O."""
  num_unique_samples = max(1, min(int(num_unique_samples), int(dataset_size)))
  generator = torch.Generator().manual_seed(int(seed))
  tokens = torch.randint(
    low=0,
    high=max(int(vocab_size), 1),
    size=(num_unique_samples, int(seq_len)),
    generator=generator,
    dtype=torch.long)
  if mask_token_id is not None:
    mask_token_id = int(mask_token_id)
    if 0 <= mask_token_id < int(vocab_size):
      tokens[tokens == mask_token_id] = 0
  if bos_token_id is not None and seq_len > 0:
    tokens[:, 0] = int(bos_token_id)
  if eos_token_id is not None and seq_len > 1:
    tokens[:, -1] = int(eos_token_id)
  attention_mask = torch.ones_like(tokens)
  return _InMemorySyntheticTokenDataset(tokens, attention_mask, dataset_size)


class _FreshRandomTokenDataset(torch.utils.data.Dataset):
  """Map-style dataset that draws new random tokens for every index.

  Unlike ``_InMemorySyntheticTokenDataset`` there is no fixed pool: every
  index is a distinct sample, generated on the fly from ``(seed, idx)``.  Each
  training step therefore sees fresh data, while two runs with the same seed
  and sampler order see identical batches (useful for A/B speed benchmarks).
  """

  def __init__(self, dataset_size, seq_len, vocab_size, bos_token_id=None,
               eos_token_id=None, mask_token_id=None, seed=1234):
    self.dataset_size = int(dataset_size)
    self.seq_len = int(seq_len)
    self.vocab_size = max(int(vocab_size), 1)
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.mask_token_id = mask_token_id
    self.seed = int(seed)

  def __len__(self):
    return self.dataset_size

  def __getitem__(self, idx):
    generator = torch.Generator().manual_seed(
      self.seed * 1_000_003 + int(idx))
    tokens = torch.randint(
      low=0,
      high=self.vocab_size,
      size=(self.seq_len,),
      generator=generator,
      dtype=torch.long)
    if (self.mask_token_id is not None
        and 0 <= int(self.mask_token_id) < self.vocab_size):
      tokens[tokens == int(self.mask_token_id)] = 0
    if self.bos_token_id is not None and self.seq_len > 0:
      tokens[0] = int(self.bos_token_id)
    if self.eos_token_id is not None and self.seq_len > 1:
      tokens[-1] = int(self.eos_token_id)
    return {
      'input_ids': tokens,
      'attention_mask': torch.ones_like(tokens),
    }


def generate_fresh_random_token_dataset(
    dataset_size, seq_len, vocab_size, bos_token_id=None, eos_token_id=None,
    mask_token_id=None, seed=1234):
  """Random tokens generated per index, with no disk, tokenizer or reuse."""
  return _FreshRandomTokenDataset(
    dataset_size=dataset_size,
    seq_len=seq_len,
    vocab_size=vocab_size,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    mask_token_id=mask_token_id,
    seed=seed)


def _generate_synthetic_data(dataset_size, seq_len, vocab_size):
  dataset = np.zeros((dataset_size, seq_len), dtype=int)
  dataset[:, 0] = vocab_size - 2  # bos
  dataset[:, -1] = vocab_size - 1  # eos
  for i in range(dataset_size):
    temp = np.random.randint(vocab_size - 2)
    for j in reversed(range(1, seq_len - 1)):
      dataset[i, j] = temp
      if temp != 0:
        temp = temp // 4
      else:
        temp = np.random.randint(vocab_size - 2)
  return dataset


def generate_synthetic_dataset(train_dataset_size, validation_dataset_size,
                               seq_len, vocab_size):
  np.random.seed(42)
  train_data = torch.from_numpy(
    _generate_synthetic_data(train_dataset_size, seq_len, vocab_size))
  train_dataset = datasets.Dataset.from_dict({
    'input_ids': train_data,
    'attention_mask': torch.ones_like(train_data),
  })
  train_dataset.set_format(type='torch')

  np.random.seed(41)
  validation_data = torch.from_numpy(
    _generate_synthetic_data(validation_dataset_size, seq_len, vocab_size))
  validation_dataset = datasets.Dataset.from_dict({
    'input_ids': validation_data,
    'attention_mask': torch.ones_like(validation_data),
  })
  validation_dataset.set_format(type='torch')

  return {
    'train': train_dataset,
    'validation': validation_dataset,
  }


def get_lambada_test_dataset():
  url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

  def read_jsonl_to_list(url):
    response = requests.get(url, stream=True)
    data_list = []
    for line in response.iter_lines(decode_unicode=True):
      if line:
        data = json.loads(line)
        data_list.append(data)
    return data_list

  lambada_data = read_jsonl_to_list(url)
  dataset = datasets.Dataset.from_list(lambada_data)
  return dataset


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True,
                      crop_train=False):
  """Adapted from D3PM text datasets."""
  url = 'http://mattmahoney.net/dc/text8.zip'
  cache_dir = f'{cache_dir}/text8' if not crop_train else f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL %s.', url)
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)
      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read('text8').decode('utf-8')
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }
      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()
    def chunks(lst, n):
      for i in range(0, len(lst), n):
        yield lst[i:i + n]
    dataset_dict = {}
    for k, v in splits.items():
      chunk_size = 2 * max_seq_length if (k == 'train' and crop_train) else max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)
  return dataset
