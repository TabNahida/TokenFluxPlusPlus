# TokenFlux C++ API

## Contents

- [Header map](#header-map)
- [High-level tokenizer API](#high-level-tokenizer-api)
- [Training API](#training-api)
- [Dataset pre-tokenization API](#dataset-pre-tokenization-api)
- [Low-level tokenizer core](#low-level-tokenizer-core)
- [Notes](#notes)

## Header map

Use the header that matches the layer you want to integrate:

- `tokenizer/tokenflux_tokenizer_api.hpp`: high-level in-memory tokenizer facade for encode and decode.
- `tokenizer/tokenflux_config.hpp`: training config struct.
- `tokenizer/train_pipeline.hpp`: `run_train(Config)` entry point.
- `tokenizer/tokenize_common.hpp`: dataset tokenize args and common tokenizer data types.
- `tokenizer/tokenize_pipeline.hpp`: `tokenflux::tokenize::run_tokenize(const Args&)` entry point.
- `tokenizer/tokenize_tokenizer.hpp`: low-level `TokenizerEncoder` core.

## High-level tokenizer API

Header:

```cpp
#include "tokenizer/tokenflux_tokenizer_api.hpp"
```

Example:

```cpp
#include "tokenizer/tokenflux_tokenizer_api.hpp"
#include <iostream>

int main() {
    tokenflux::Tokenizer tok("tokenizer.json");

    tokenflux::EncodeOptions opts;
    opts.eos_token = "<|endoftext|>";

    auto ids = tok.encode("hello tokenflux", opts);
    auto text = tok.decode(ids, true, true);

    std::cout << "model=" << tok.model_name()
              << " vocab=" << tok.vocab_size()
              << " tokens=" << ids.size()
              << " text=" << text << "\n";
    return 0;
}
```

### `tokenflux::EncodeOptions`

| Field | Default | Meaning |
|---|---|---|
| `bos_token` | `""` | Optional BOS token prepended before encoding. Empty string disables BOS. |
| `eos_token` | `""` | Optional EOS token appended after encoding. Empty string disables EOS. |
| `reset_cache` | `false` | Clears the tokenizer's piece cache before encoding. |

If `bos_token` or `eos_token` is set but not found in the tokenizer vocabulary, encode throws `std::runtime_error`.

### `tokenflux::Tokenizer`

Constructors:

- `Tokenizer()`
- `explicit Tokenizer(const std::string& tokenizer_path)`

Methods:

| Method | Return | Meaning |
|---|---|---|
| `load(tokenizer_path, err)` | `bool` | Loads a tokenizer and writes any error message into `err`. Clears the encode cache. |
| `load_or_throw(tokenizer_path)` | `void` | Loads a tokenizer or throws `std::runtime_error`. |
| `vocab_size()` | `std::size_t` | Vocabulary size. |
| `model_name()` | `std::string` | Runtime model name such as `BPE`, `WordPiece`, or `Unigram`. |
| `token_to_id(token, id)` | `bool` | Looks up a token id. Returns `false` if missing. |
| `id_to_token(id)` | `std::string` | Looks up the token string for an id. Out-of-range ids return an empty string. |
| `clear_cache()` | `void` | Clears the internal encode cache. |
| `encode(text, options={})` | `std::vector<std::uint32_t>` | Encodes one text. |
| `encode_batch(texts, options={})` | `std::vector<std::vector<std::uint32_t>>` | Encodes a batch of texts. |
| `decode(token_ids, skip_special_tokens=false, clean_up_tokenization_spaces=true)` | `std::string` | Decodes one token-id sequence. |
| `decode_batch(token_ids_batch, skip_special_tokens=false, clean_up_tokenization_spaces=true)` | `std::vector<std::string>` | Batch decode. |

Decode notes:

- `skip_special_tokens=true` removes special-token ids from the decoded output.
- `clean_up_tokenization_spaces=true` collapses repeated whitespace and trims the final text.
- Byte-level tokenizers decode through the tokenizer's byte-to-unicode map before cleanup.

## Training API

Headers:

```cpp
#include "tokenizer/tokenflux_config.hpp"
#include "tokenizer/train_pipeline.hpp"
```

Entry point:

- `int run_train(Config cfg);`

`run_train` returns `0` on success and non-zero on failure.

### `TrainerKind`

- `TrainerKind::byte_bpe`
- `TrainerKind::bpe`
- `TrainerKind::wordpiece`
- `TrainerKind::unigram`

### `Config`

`Config` is the training configuration structure.

#### Input and output fields

| Field | Default | Meaning |
|---|---|---|
| `env_path` | `".env"` | Environment file used for CLI-style path overrides. |
| `data_glob` | `""` | Glob pattern for training inputs. |
| `data_list` | `""` | File containing one input path/URI per line. |
| `input_entries` | `{}` | Explicit file/URI list. |
| `text_field` | `"text"` | JSON/JSONL field to read. |
| `output_json` | `"tokenizer.json"` | Main tokenizer artifact. |
| `output_vocab` | `"vocab.json"` | Auxiliary vocab artifact. |
| `output_merges` | `"merges.txt"` | Auxiliary merges artifact. |
| `chunk_dir` | `"artifacts/bpe/chunks"` | Intermediate chunk directory used during counting and merge. |

#### Model fields

| Field | Default | Meaning |
|---|---|---|
| `trainer` | `TrainerKind::byte_bpe` | Training backend. |
| `vocab_size` | `50000` | Target vocabulary size. |
| `unk_token` | `"<|endoftext|>"` | Unknown token string. |
| `special_tokens` | `{ "<|endoftext|>" }` | Special tokens injected into the final tokenizer. |
| `min_freq` | `2` | Minimum token frequency. |
| `min_pair_freq` | `2` | Minimum pair frequency for merge-based trainers. |
| `max_token_length` | `16` | Maximum seed token length for unigram training. |
| `unigram_em_iters` | `4` | Number of unigram EM iterations. |
| `unigram_seed_multiplier` | `4` | Initial unigram seed multiplier relative to `vocab_size`. |
| `unigram_prune_ratio` | `0.75` | Candidate retention ratio during unigram pruning. |
| `wordpiece_continuing_prefix` | `"##"` | Prefix attached to continuing WordPiece subwords. |

#### Throughput and memory fields

| Field | Default | Meaning |
|---|---|---|
| `threads` | `0` | Worker count. `0` means auto-select hardware concurrency. |
| `chunk_files` | `1` | Number of source files processed together before merge boundaries. |
| `chunk_docs` | `20000` | In-chunk reduce cadence. |
| `top_k` | `200000` | Per-chunk local count cap. `0` means uncapped unless memory limits derive a cap. |
| `max_chars_per_doc` | `20000` | Documents longer than this are truncated during training. |
| `progress_interval_ms` | `1000` | Progress print interval in milliseconds. |
| `max_memory_mb` | `0` | Memory cap hint. `0` means unlimited. |
| `pair_max_entries` | `0` | Explicit pair-table cap. `0` derives from `max_memory_mb` when possible. |
| `records_per_chunk` | `5000` | Chunk write granularity and progress granularity. Internally normalized to at least `1`. |
| `queue_capacity` | `0` | Internal queue capacity. `0` means derive from thread count. |
| `prescan_records` | `false` | Two-pass prescan for better progress totals. |

#### Resume and artifact fields

| Field | Default | Meaning |
|---|---|---|
| `resume` | `true` | Reuse compatible intermediate chunk state when present. |
| `write_vocab` | `true` | Write `output_vocab`. |
| `write_merges` | `true` | Write `output_merges` when the trainer emits merge rules. |

### Training example

```cpp
#include "tokenizer/tokenflux_config.hpp"
#include "tokenizer/train_pipeline.hpp"

int main() {
    Config cfg;
    cfg.trainer = TrainerKind::byte_bpe;
    cfg.input_entries = {"data/train.jsonl"};
    cfg.output_json = "tokenizer.json";
    cfg.output_vocab = "vocab.json";
    cfg.output_merges = "merges.txt";
    cfg.threads = 8;
    return run_train(cfg);
}
```

## Dataset pre-tokenization API

Headers:

```cpp
#include "tokenizer/tokenize_common.hpp"
#include "tokenizer/tokenize_pipeline.hpp"
```

Entry point:

- `int tokenflux::tokenize::run_tokenize(const Args& args);`

`run_tokenize` returns `0` on success and non-zero on failure.

### `tokenflux::tokenize::Args`

#### Input and output fields

| Field | Default | Meaning |
|---|---|---|
| `env_file` | `".env"` | Environment file used for CLI-style path overrides. |
| `data_glob` | `""` | Glob pattern for input files. |
| `data_list` | `""` | File containing one input path/URI per line. |
| `input_entries` | `{}` | Explicit input file/URI list. |
| `text_field` | `"text"` | JSON/JSONL field name. |
| `tokenizer_path` | `"tokenizer.json"` | Tokenizer to load. |
| `out_dir` | `"data/tokens"` | Output root directory. |

#### Sharding and filtering fields

| Field | Default | Meaning |
|---|---|---|
| `max_tokens_per_shard` | `50000000` | Hard token cap per shard file. |
| `encode_batch_size` | `256` | Per-worker document batch size during encode. |
| `min_chars` | `1` | Documents shorter than this are skipped. |
| `max_chars` | `20000` | Documents longer than this are truncated before encode. |
| `max_docs` | `0` | Compatibility field; keep `0` for the current C++ tokenize path. |
| `add_eos` | `true` | Append one EOS token to each document when possible. |
| `eos_token` | `"<|endoftext|>"` | EOS token string. |
| `bos_token` | `""` | Optional BOS token string. Empty disables BOS. |

#### Throughput, memory, and resume fields

| Field | Default | Meaning |
|---|---|---|
| `progress_every` | `10000` | Progress reporting cadence in documents. |
| `threads` | `0` | Worker count. `0` means auto-select hardware concurrency. |
| `cache_max_entries` | `50000` | Token-piece cache cap per worker. When the cap is reached, the cache is cleared. |
| `max_memory_mb` | `0` | Per-worker memory cap hint. `0` means unlimited. |
| `prescan_records` | `false` | Two-pass prescan for more accurate total-doc progress. |
| `resume` | `true` | Reuse compatible completed work from `cache/completed.list`. |

### Output layout

`run_tokenize` writes a concatenated document token stream across shard files:

- `out_dir/meta.json`
- `out_dir/shards/train_000000.bin`, `train_000001.bin`, ...
- `out_dir/cache/completed.list`

Behavior notes:

- Documents are processed in source order.
- Output is a concatenated stream, not one file per document.
- When `add_eos=true`, document boundaries can be recovered by scanning the EOS id recorded in `meta.json`.
- With one large input file and `threads > 1`, TokenFlux uses one file worker and multiple in-file encode threads.

### Tokenization example

```cpp
#include "tokenizer/tokenize_common.hpp"
#include "tokenizer/tokenize_pipeline.hpp"

int main() {
    tokenflux::tokenize::Args args;
    args.input_entries = {"data/train.jsonl"};
    args.tokenizer_path = "tokenizer.json";
    args.out_dir = "data/tokens";
    args.threads = 8;
    return tokenflux::tokenize::run_tokenize(args);
}
```

## Low-level tokenizer core

Header:

```cpp
#include "tokenizer/tokenize_tokenizer.hpp"
```

Use `tokenflux::tokenize::TokenizerEncoder` when you want lower-level control over caching or to integrate encode/decode into your own pipeline.

### `tokenflux::tokenize::TokenizerEncoder`

| Method | Return | Meaning |
|---|---|---|
| `load(path, err)` | `bool` | Loads `tokenizer.json`. |
| `vocab_size()` | `std::size_t` | Vocabulary size. |
| `model_name()` | `std::string` | Model name. |
| `token_to_id(token, id)` | `bool` | Token lookup. |
| `id_to_token(id)` | `std::string` | Reverse token lookup. |
| `encode_text_append(text, cache, out_ids)` | `void` | Encodes one text and appends ids to `out_ids` using the caller-owned cache map. |
| `decode(token_ids, skip_special_tokens=false, clean_up_tokenization_spaces=true)` | `std::string` | Decodes one token-id sequence. |
| `decode_batch(token_ids_batch, skip_special_tokens=false, clean_up_tokenization_spaces=true)` | `std::vector<std::string>` | Batch decode. |

This is the underlying core used by both the high-level C++ facade and the Python bindings.

## Notes

- `threads = 0` means auto-threading for both training and dataset pre-tokenization.
- `max_memory_mb = 0` means no explicit memory cap.
- For concurrent encode-heavy workloads, prefer one `tokenflux::Tokenizer` instance per thread because encode uses a mutable cache.
- `run_train` and `run_tokenize` are library entry points; the CLI binaries `TokenFluxTrain` and `TokenFluxTokenize` wrap the same config structs and functions.
