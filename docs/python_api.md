# TokenFlux Python API (v0.3.3)

Module:

```python
import tokenflux as tf
```

Exports:

- `tf.__version__`
- `tf.TrainerKind`
- `tf.TrainConfig`
- `tf.TokenizeArgs`
- `tf.Tokenizer`
- `tf.train(config, inputs=[])`
- `tf.tokenize(args, inputs=[])`

## 1) `TrainerKind`

- `TrainerKind.byte_bpe`
- `TrainerKind.bpe`
- `TrainerKind.wordpiece`
- `TrainerKind.unigram`

## 2) `TrainConfig` (training config)

Commonly used fields:

- Input/output:
`env_path`, `data_glob`, `data_list`, `input_entries`, `text_field`, `output_json`, `output_vocab`, `output_merges`
- Model:
`trainer`, `vocab_size`, `unk_token`, `special_tokens`
- Performance:
`threads`, `max_memory_mb`, `prescan_records`, `resume`

Advanced fields:

- `min_freq`, `min_pair_freq`, `chunk_files`, `chunk_docs`, `top_k`
- `max_chars_per_doc`, `progress_interval_ms`, `pair_max_entries`
- `records_per_chunk`, `queue_capacity`, `max_token_length`
- `unigram_em_iters`, `unigram_seed_multiplier`, `unigram_prune_ratio`
- `wordpiece_continuing_prefix`, `chunk_dir`, `write_vocab`, `write_merges`

## 3) `TokenizeArgs` (dataset pre-tokenization config)

Input/output:

- `env_file`, `data_glob`, `data_list`, `input_entries`, `text_field`
- `tokenizer_path`, `out_dir`

Control:

- `threads`, `encode_batch_size`, `max_tokens_per_shard`
- `min_chars`, `max_chars`, `max_memory_mb`
- `cache_max_entries`, `prescan_records`, `resume`
- `progress_every` (compatibility field)

Document boundary options:

- `add_eos` (default `True`)
- `eos_token` (default `"<|endoftext|>"`)
- `bos_token` (default `""`)

Compatibility:

- `max_docs` exists for compatibility and should stay `0` for C++ tokenize path.

## 4) `Tokenizer` (in-memory encode API)

### Constructor

- `Tokenizer()`
- `Tokenizer(tokenizer_path: str)`

### Methods

- `load(tokenizer_path: str) -> None`
- `token_to_id(token: str) -> Optional[int]`
- `encode(text: str, bos_token: str = "", eos_token: str = "", reset_cache: bool = False) -> list[int]`
- `encode_batch(texts: list[str], bos_token: str = "", eos_token: str = "", reset_cache: bool = False) -> list[list[int]]`
- `encode_to_torch(text: str, bos_token: str = "", eos_token: str = "", dtype: str = "int64", reset_cache: bool = False) -> torch.Tensor`
- `encode_batch_to_torch(texts: list[str], bos_token: str = "", eos_token: str = "", pad_id: int = 0, dtype: str = "int64", reset_cache: bool = False) -> dict`
- `tokenize_inputs_to_torch(inputs: list[str], text_field: str = "text", min_chars: int = 1, max_chars: int = 20000, bos_token: str = "", eos_token: str = "", dtype: str = "int64", reset_cache: bool = False) -> dict`

Readonly properties:

- `tokenizer_path`
- `vocab_size`
- `model_name`

`tokenize_inputs_to_torch` return fields:

- `token_ids`: flattened tokens
- `doc_offsets`: start offset per doc, with final sentinel offset
- `doc_lengths`: token length per doc
- `num_docs`
- `num_skipped`
- `sources`

## 5) Top-level functions

### `tf.train(config, inputs=[])`

Runs tokenizer training using `TrainConfig`.

- `inputs`: list of files/URLs (optional). When provided, overrides `config.input_entries`.
- Raises `RuntimeError` when training fails.

### `tf.tokenize(args, inputs=[])`

Runs dataset pre-tokenization to shard files.

- `inputs`: list of files/URLs (optional). When provided, overrides `args.input_entries`.
- Output contains `meta.json`, `shards/`, and `cache/completed.list`.
- Raises `RuntimeError` when tokenize fails.

## 6) Minimal examples

Train:

```python
import tokenflux as tf

cfg = tf.TrainConfig()
cfg.trainer = tf.TrainerKind.byte_bpe
cfg.vocab_size = 16000
cfg.output_json = "tokenizer.json"
cfg.output_vocab = "vocab.json"
cfg.output_merges = "merges.txt"
tf.train(cfg, ["data/train.jsonl"])
```

Tokenize dataset (disable EOS):

```python
import tokenflux as tf

args = tf.TokenizeArgs()
args.tokenizer_path = "tokenizer.json"
args.out_dir = "data/tokens"
args.add_eos = False
tf.tokenize(args, ["data/train.jsonl"])
```
