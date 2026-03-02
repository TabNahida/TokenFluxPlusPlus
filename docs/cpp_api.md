# TokenFlux C++ API (v0.3.3)

## 1) High-level tokenizer API

Header: `tokenizer/tokenflux_tokenizer_api.hpp`

```cpp
#include "tokenizer/tokenflux_tokenizer_api.hpp"
#include <iostream>

int main() {
    tokenflux::Tokenizer tok("tokenizer.json");

    tokenflux::EncodeOptions opts;
    opts.eos_token = "<|endoftext|>"; // leave empty to disable EOS

    auto ids = tok.encode("hello tokenflux", opts);
    std::cout << "model=" << tok.model_name() << " vocab=" << tok.vocab_size()
              << " tokens=" << ids.size() << "\n";
    return 0;
}
```

### `tokenflux::EncodeOptions`

- `bos_token`: prepend one BOS token if found in vocab.
- `eos_token`: append one EOS token if found in vocab.
- `reset_cache`: clear piece-cache before encoding.

### `tokenflux::Tokenizer`

- `Tokenizer()`
- `explicit Tokenizer(const std::string& tokenizer_path)`
- `bool load(const std::string& tokenizer_path, std::string& err)`
- `void load_or_throw(const std::string& tokenizer_path)`
- `std::size_t vocab_size() const`
- `std::string model_name() const` (`BPE` / `WordPiece` / `Unigram`)
- `bool token_to_id(const std::string& token, std::uint32_t& id) const`
- `void clear_cache()`
- `std::vector<std::uint32_t> encode(const std::string& text, const EncodeOptions& options = {})`
- `std::vector<std::vector<std::uint32_t>> encode_batch(const std::vector<std::string>& texts, const EncodeOptions& options = {})`

Notes:
- Missing BOS/EOS token throws `std::runtime_error`.
- The internal cache is mutable; one `Tokenizer` instance is intended for one encoding thread.

## 2) CLI tokenization output layout

`TokenFluxTokenize` writes one concatenated token stream across shard files:

- `out_dir/shards/train_000000.bin`, `train_000001.bin`, ...
- `out_dir/meta.json`
- `out_dir/cache/completed.list`

Document handling:

- Input records are processed as documents in order.
- Output storage is a concatenated stream, not one file per doc.
- By default each doc ends with EOS (`--add-eos`, default on).
- Use `--no-eos` to disable EOS append.

Boundary recovery:

- When EOS is on, doc boundaries can be recovered by scanning EOS id from `meta.json`.
- When EOS is off, boundaries are not explicitly encoded in shard payload.

Relevant CLI flags:

- `--add-eos` / `--no-eos`
- `--eos-token <token>`
- `--bos-token <token>`
