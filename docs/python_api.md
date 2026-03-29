# TokenFlux Python API

## Contents

- [Module](#module)
- [Runtime Requirements](#runtime-requirements)
- [TrainerKind](#trainerkind)
- [TrainConfig](#trainconfig)
- [TokenizeArgs](#tokenizeargs)
- [Tokenizer](#tokenizer)
- [Top-level functions](#top-level-functions)
- [TokenAnalyzer](#tokenanalyzer)
- [TokenVisualizer](#tokenvisualizer)
- [Examples](#examples)

## Module

```python
import tokenflux as tf
```

Top-level exports:

- `tf.__version__`
- `tf.TrainerKind`
- `tf.TrainConfig`
- `tf.TokenizeArgs`
- `tf.Tokenizer`
- `tf.train(config, inputs=[])`
- `tf.tokenize(args, inputs=[])`
- `tf.TokenAnalyzer`
- `tf.TokenVisualizer`

## Runtime Requirements

Core encode/train/tokenize features require the compiled `tokenflux_cpp` extension.

Additional Python helpers have extra runtime requirements:

- `Tokenizer.encode_to_torch`, `Tokenizer.encode_batch_to_torch`, and `Tokenizer.tokenize_inputs_to_torch` require `torch`.
- `TokenAnalyzer` requires `numpy`.
- `TokenVisualizer` requires `matplotlib` and `seaborn`.

Current package layout re-exports `TokenAnalyzer` and `TokenVisualizer` from `tokenflux.__init__`, so a plain `import tokenflux as tf` also expects `numpy`, `matplotlib`, and `seaborn` to be available.

## TrainerKind

Available trainer kinds:

- `tf.TrainerKind.byte_bpe`: byte-level BPE, best compatibility with `tokenizer.json` style byte-level vocabularies.
- `tf.TrainerKind.bpe`: whitespace-pretokenized BPE.
- `tf.TrainerKind.wordpiece`: WordPiece with configurable continuing subword prefix.
- `tf.TrainerKind.unigram`: unigram language-model tokenizer.

## TrainConfig

`TrainConfig` controls tokenizer training. Create with `cfg = tf.TrainConfig()`.

### Input and output fields

| Field | Default | Meaning |
|---|---|---|
| `env_path` | `".env"` | Environment file used by CLI-style path resolution and overrides. |
| `data_glob` | `""` | Glob pattern for training inputs. |
| `data_list` | `""` | File containing one input path/URI per line. |
| `input_entries` | `[]` | Explicit list of files or URIs. Python `tf.train(..., inputs=[...])` overrides this. |
| `text_field` | `"text"` | Field name read from JSON/JSONL records. |
| `output_json` | `"tokenizer.json"` | Main tokenizer output. |
| `output_vocab` | `"vocab.json"` | Auxiliary vocab output. |
| `output_merges` | `"merges.txt"` | Auxiliary merges output for merge-based trainers. |
| `chunk_dir` | `"artifacts/bpe/chunks"` | Temporary/intermediate chunk directory used during counting and merge stages. |

### Model fields

| Field | Default | Meaning |
|---|---|---|
| `trainer` | `tf.TrainerKind.byte_bpe` | Trainer backend. |
| `vocab_size` | `50000` | Target vocabulary size. |
| `unk_token` | `"<|endoftext|>"` | Unknown token string used by applicable trainers. |
| `special_tokens` | `["<|endoftext|>"]` | Special tokens injected into the tokenizer output. |
| `min_freq` | `2` | Minimum token frequency required before a token is kept. |
| `min_pair_freq` | `2` | Minimum pair frequency for merge-based trainers. |
| `max_token_length` | `16` | Maximum seed token length used by unigram training. |
| `unigram_em_iters` | `4` | Number of EM refinement iterations for unigram training. |
| `unigram_seed_multiplier` | `4` | Initial unigram seed size multiplier relative to `vocab_size`. |
| `unigram_prune_ratio` | `0.75` | Fraction retained when pruning unigram candidates between EM rounds. |
| `wordpiece_continuing_prefix` | `"##"` | Prefix attached to continuing WordPiece subwords. |

### Throughput and memory fields

| Field | Default | Meaning |
|---|---|---|
| `threads` | `0` | Worker count. `0` means auto-select from hardware concurrency. |
| `chunk_files` | `1` | Number of source files processed together before chunk merge boundaries. |
| `chunk_docs` | `20000` | In-chunk reduce cadence. Lower values reduce peak memory; higher values reduce merge overhead. |
| `top_k` | `200000` | Per-chunk local count cap. `0` means uncapped unless memory limits derive a cap. |
| `max_chars_per_doc` | `20000` | Input documents longer than this are truncated during training. |
| `progress_interval_ms` | `1000` | Progress print interval in milliseconds. |
| `max_memory_mb` | `0` | Memory cap hint. `0` means unlimited. When set, several internal caps are derived from it. |
| `pair_max_entries` | `0` | Explicit pair-table cap. `0` means derive from `max_memory_mb` when possible, otherwise uncapped. |
| `records_per_chunk` | `5000` | Chunk write granularity and progress granularity. `0` is normalized internally to `1`. |
| `queue_capacity` | `0` | Internal pipeline queue capacity. `0` means derive from thread count. |
| `prescan_records` | `False` | Two-pass prescan for more accurate progress totals. Off keeps training single-pass. |

### Resume and artifact flags

| Field | Default | Meaning |
|---|---|---|
| `resume` | `True` | Reuse existing intermediate chunk state when possible. |
| `write_vocab` | `True` | Write `output_vocab`. |
| `write_merges` | `True` | Write `output_merges` when the trainer produces merge rules. |

## TokenizeArgs

`TokenizeArgs` controls dataset pre-tokenization. Create with `args = tf.TokenizeArgs()`.

### Input and output fields

| Field | Default | Meaning |
|---|---|---|
| `env_file` | `".env"` | Environment file used by CLI-style path resolution and overrides. |
| `data_glob` | `""` | Glob pattern for input data. |
| `data_list` | `""` | File containing one input path/URI per line. |
| `input_entries` | `[]` | Explicit list of files or URIs. Python `tf.tokenize(..., inputs=[...])` overrides this. |
| `text_field` | `"text"` | JSON/JSONL text field name. |
| `tokenizer_path` | `"tokenizer.json"` | Tokenizer to load. |
| `out_dir` | `"data/tokens"` | Output root. Contains `meta.json`, `shards/`, and `cache/completed.list`. |

### Sharding and filtering fields

| Field | Default | Meaning |
|---|---|---|
| `max_tokens_per_shard` | `50000000` | Hard token cap per shard file. |
| `encode_batch_size` | `256` | Per-worker document batch size during encode. |
| `min_chars` | `1` | Documents shorter than this are skipped. |
| `max_chars` | `20000` | Documents longer than this are truncated before encoding. |
| `max_docs` | `0` | Compatibility field. Keep `0` for the C++ tokenize path. |
| `add_eos` | `True` | Append one EOS token to each document when `eos_token` resolves in the vocab. |
| `eos_token` | `"<|endoftext|>"` | EOS token string used when `add_eos=True`. |
| `bos_token` | `""` | Optional BOS token prepended to each document. Empty string disables BOS. |

### Throughput, progress, and reuse fields

| Field | Default | Meaning |
|---|---|---|
| `progress_every` | `10000` | Progress reporting cadence in documents. |
| `threads` | `0` | Worker count. `0` means auto-select from hardware concurrency. With one large file, TokenFlux splits work into in-file encode threads. |
| `cache_max_entries` | `50000` | Token-piece cache cap per worker. When the cap is hit, the cache is cleared. |
| `max_memory_mb` | `0` | Per-worker memory cap hint. `0` means unlimited. |
| `prescan_records` | `False` | Two-pass prescan for more accurate total-doc progress. |
| `resume` | `True` | Reuse compatible completed file work from `cache/completed.list`. |

## Tokenizer

`Tokenizer` is the in-memory Python runtime API.

### Constructor

- `tf.Tokenizer()`
- `tf.Tokenizer(tokenizer_path: str)`

### Properties

| Property | Type | Meaning |
|---|---|---|
| `tokenizer_path` | `str` | Loaded tokenizer path. Empty until `load()` or path constructor is used. |
| `vocab_size` | `int` | Vocabulary size. |
| `model_name` | `str` | Runtime model name such as `BPE`, `WordPiece`, or `Unigram`. |

### Methods

| Method | Return | Notes |
|---|---|---|
| `load(tokenizer_path)` | `None` | Loads a tokenizer and clears the encode cache. Raises `RuntimeError` on failure. |
| `token_to_id(token)` | `Optional[int]` | Returns `None` when the token does not exist. |
| `id_to_token(id)` | `str` | Returns the token string. Out-of-range ids return an empty string. |
| `encode(text, bos_token="", eos_token="", reset_cache=False)` | `list[int]` | Encodes one string. Missing BOS/EOS tokens raise `RuntimeError`. |
| `encode_batch(texts, bos_token="", eos_token="", reset_cache=False)` | `list[list[int]]` | Encodes a batch with one shared cache per tokenizer instance. |
| `decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)` | `str` | Native C++ decode path. |
| `decode_batch(token_ids_batch, skip_special_tokens=False, clean_up_tokenization_spaces=True)` | `list[str]` | Batch decode. |
| `encode_to_torch(text, ..., dtype="int64")` | `torch.Tensor` | 1D tensor containing encoded token ids. Requires `torch`. |
| `encode_batch_to_torch(texts, ..., pad_id=0, dtype="int64")` | `dict` | Returns `{"input_ids": tensor, "lengths": tensor}`. Requires `torch`. |
| `tokenize_inputs_to_torch(inputs, text_field="text", min_chars=1, max_chars=20000, bos_token="", eos_token="", dtype="int64", reset_cache=False)` | `dict` | Streams input files/URIs, tokenizes them, and returns flattened tensors. Requires `torch`. |

### Decode behavior

- `skip_special_tokens=True` removes configured special-token ids from the output text.
- `clean_up_tokenization_spaces=True` collapses repeated whitespace and trims leading/trailing spaces after decode.
- Byte-level tokenizers decode back through the tokenizer's byte-to-unicode map.

### `tokenize_inputs_to_torch` return fields

| Key | Meaning |
|---|---|
| `token_ids` | Flattened token stream for all accepted documents. |
| `doc_offsets` | Inclusive start offset per document, with one final sentinel offset. |
| `doc_lengths` | Token length for each document. |
| `num_docs` | Number of accepted documents. |
| `num_skipped` | Number of skipped documents due to filtering. |
| `sources` | Source path/URI for each accepted document. |

### Threading note

A single `Tokenizer` instance is best treated as one encode worker because it keeps a mutable piece cache. For concurrent encode/decode work, create one tokenizer instance per thread.

## Top-level functions

### `tf.train(config, inputs=[])`

Runs tokenizer training.

- `config`: `tf.TrainConfig`
- `inputs`: optional explicit list of files/URIs. When provided, it replaces `config.input_entries`.
- Raises `RuntimeError` when training fails.

### `tf.tokenize(args, inputs=[])`

Runs dataset pre-tokenization into shard files.

- `args`: `tf.TokenizeArgs`
- `inputs`: optional explicit list of files/URIs. When provided, it replaces `args.input_entries`.
- Output layout: `meta.json`, `shards/`, `cache/completed.list`.
- Raises `RuntimeError` when tokenization fails.

## TokenAnalyzer

`TokenAnalyzer` is exported at the top level as `tf.TokenAnalyzer`.

### Constructor

- `tf.TokenAnalyzer(tokenizer)`

`tokenizer` may be either:

- a `tf.Tokenizer` instance
- a `tokenizer.json` path string

### Methods

| Method | Return | Meaning |
|---|---|---|
| `analyze_tokens(texts)` | `dict` | Computes aggregate token statistics for a list of texts. |
| `compute_oov_rate(texts, reserved_tokens=None)` | `float` | Computes OOV rate over encoded token ids. |
| `get_token_statistics(texts)` | `dict` | Returns aggregate stats plus per-document stats. |

### `analyze_tokens()` return fields

| Key | Meaning |
|---|---|
| `total_tokens` | Total token count across all input texts. |
| `unique_tokens` | Number of distinct token ids observed. |
| `vocab_size` | Vocabulary size loaded from `tokenizer.json`. |
| `vocab_coverage` | `unique_tokens / vocab_size`. |
| `token_frequency` | Token-string to frequency map. |
| `top_20_tokens` | Top-20 `(token, count)` pairs. |
| `token_length_distribution` | Token-string-length distribution as proportions. |
| `zipf_analysis` | Zipf fit statistics and top-rank series. |
| `avg_token_length` | Mean token-string length among observed vocabulary items. |
| `median_token_length` | Median token-string length among observed vocabulary items. |

### `get_token_statistics()` extra field

- `per_document_stats`: one entry per input text with `doc_id`, `char_length`, `token_count`, and `chars_per_token`.

## TokenVisualizer

`TokenVisualizer` is exported at the top level as `tf.TokenVisualizer`.

### Constructor

- `tf.TokenVisualizer(style="default", figsize=(12, 8))`

Arguments:

- `style`: one of `default`, `dark`, `whitegrid`
- `figsize`: matplotlib figure size tuple

### Methods

| Method | Return | Meaning |
|---|---|---|
| `plot_vocab_distribution(token_stats, top_n=50, save_path=None)` | `matplotlib.figure.Figure` | Plots the most frequent tokens. |
| `plot_token_length_distribution(token_stats, save_path=None)` | `matplotlib.figure.Figure` | Plots token-string length proportions. |
| `plot_zipf_law(token_stats, save_path=None)` | `matplotlib.figure.Figure` | Plots Zipf log-log distribution and fitted exponent. |
| `plot_vocab_coverage(token_stats, save_path=None)` | `matplotlib.figure.Figure` | Plots used vs unused vocabulary share. |
| `plot_comprehensive_analysis(token_stats, save_path=None)` | `list[matplotlib.figure.Figure]` | Produces a combined multi-panel figure and returns it in a list. |

All visualizer methods expect `token_stats` in the format returned by `TokenAnalyzer.analyze_tokens()` or `TokenAnalyzer.get_token_statistics()`.

## Examples

### Train and encode/decode

```python
import tokenflux as tf

cfg = tf.TrainConfig()
cfg.trainer = tf.TrainerKind.byte_bpe
cfg.vocab_size = 16000
cfg.output_json = "tokenizer.json"
cfg.output_vocab = "vocab.json"
cfg.output_merges = "merges.txt"
tf.train(cfg, ["data/train.jsonl"])

tok = tf.Tokenizer("tokenizer.json")
ids = tok.encode("hello TokenFlux")
text = tok.decode(ids)
print(ids)
print(text)
```

### Dataset pre-tokenization

```python
import tokenflux as tf

args = tf.TokenizeArgs()
args.tokenizer_path = "tokenizer.json"
args.out_dir = "data/tokens"
args.add_eos = True
tf.tokenize(args, ["data/train.jsonl"])
```

### Torch helpers

```python
import tokenflux as tf

tok = tf.Tokenizer("tokenizer.json")
batch = tok.encode_batch_to_torch(["hello", "token flux"], pad_id=0)
print(batch["input_ids"].shape)
print(batch["lengths"].tolist())
```

### Analysis and visualization

```python
import tokenflux as tf

analyzer = tf.TokenAnalyzer("tokenizer.json")
stats = analyzer.get_token_statistics(["hello world", "token flux"])

visualizer = tf.TokenVisualizer(style="whitegrid")
fig = visualizer.plot_vocab_distribution(stats, top_n=10)
```

