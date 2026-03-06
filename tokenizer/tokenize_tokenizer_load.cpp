#include "tokenize_tokenizer.hpp"

#include "tokenize_tokenizer_internal.hpp"

#include <algorithm>
#include <utility>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize
{
bool TokenizerEncoder::load(const std::string &path, std::string &err)
{
    std::string content = read_file_all(path);
    if (content.empty())
    {
        err = "failed to read tokenizer file: " + path;
        return false;
    }
    TokenizerData data;
    data.vocab.reserve(65536);
    if (!detail::parse_tokenizer_json_content(content, data))
    {
        err = "failed to parse tokenizer.json: " + path;
        return false;
    }
    vocab_ = std::move(data.vocab);
    model_type_ = data.model_type;
    pretokenizer_type_ = data.pretokenizer_type;
    continuing_subword_prefix_ = data.continuing_subword_prefix;
    if (continuing_subword_prefix_.empty())
    {
        continuing_subword_prefix_ = "##";
    }
    unk_token_ = data.unk_token;
    has_unk_ = false;
    if (!unk_token_.empty())
    {
        auto it = vocab_.find(unk_token_);
        if (it != vocab_.end())
        {
            has_unk_ = true;
            unk_id_ = it->second;
        }
    }

    auto cp_map = build_byte_to_unicode_cp();
    byte_to_unicode_ = build_byte_to_unicode_str(cp_map);

    symbols_.clear();
    symbol_to_id_.clear();
    merge_rules_.clear();
    unigram_tokens_.clear();
    unigram_index_.clear();

    if (model_type_ == ModelType::bpe)
    {
        symbols_.reserve(vocab_.size() + data.merges.size() + 256);
        symbol_to_id_.reserve(vocab_.size() + data.merges.size() + 256);
        merge_rules_.reserve(data.merges.size() * 13 / 10 + 8);

        if (pretokenizer_type_ == PretokenizerType::byte_level)
        {
            for (const auto &s : byte_to_unicode_)
            {
                ensure_symbol(s);
            }
        }
        for (const auto &kv : vocab_)
        {
            ensure_symbol(kv.first);
        }
        for (std::size_t rank = 0; rank < data.merges.size(); ++rank)
        {
            const auto &m = data.merges[rank];
            std::uint32_t left = ensure_symbol(m.first);
            std::uint32_t right = ensure_symbol(m.second);
            std::uint32_t merged = ensure_symbol(m.first + m.second);
            std::uint64_t key = pair_key(left, right);
            if (merge_rules_.find(key) == merge_rules_.end())
            {
                merge_rules_[key] = MergeRule{static_cast<std::uint32_t>(rank), merged};
            }
        }
    }
    else if (model_type_ == ModelType::unigram)
    {
        std::vector<UnigramEntry> entries = data.unigram_vocab;
        if (entries.empty())
        {
            std::vector<std::pair<std::string, std::uint32_t>> ordered;
            ordered.reserve(vocab_.size());
            for (const auto &kv : vocab_)
            {
                ordered.push_back(kv);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
            for (const auto &kv : ordered)
            {
                entries.push_back({kv.first, -1.0});
            }
        }
        unigram_tokens_.reserve(entries.size());
        for (const auto &entry : entries)
        {
            auto it = vocab_.find(entry.token);
            if (it == vocab_.end())
            {
                continue;
            }
            auto cps = split_codepoints_utf8(entry.token);
            if (cps.empty())
            {
                continue;
            }
            unigram_tokens_.push_back({entry.token, std::move(cps), entry.score, it->second});
        }
        for (std::size_t i = 0; i < unigram_tokens_.size(); ++i)
        {
            unigram_index_[unigram_tokens_[i].cps.front()].push_back(i);
        }
        for (auto &kv : unigram_index_)
        {
            auto &vec = kv.second;
            std::sort(vec.begin(), vec.end(), [&](std::size_t a, std::size_t b) {
                return unigram_tokens_[a].cps.size() > unigram_tokens_[b].cps.size();
            });
        }
    }

    // Build id_to_token_list_ for decoding
    id_to_token_list_.resize(vocab_.size());
    for (const auto &kv : vocab_)
    {
        if (kv.second < id_to_token_list_.size())
        {
            id_to_token_list_[kv.second] = kv.first;
        }
    }

    // Build unicode_to_byte_ mapping for byte-level decoding
    unicode_to_byte_.clear();
    for (std::size_t i = 0; i < byte_to_unicode_.size(); ++i)
    {
        if (!byte_to_unicode_[i].empty())
        {
            // Get the Unicode code point of the character
            std::size_t pos = 0;
            std::uint32_t unicode_cp = 0;
            if (next_codepoint(byte_to_unicode_[i], pos, unicode_cp))
            {
                unicode_to_byte_[unicode_cp] = static_cast<std::uint8_t>(i);
            }
        }
    }

    // Identify special token IDs (common special tokens)
    special_token_ids_.clear();
    std::vector<std::string> special_tokens = {"<unk>", "<s>", "</s>", "<pad>", "<cls>", "<sep>"};
    for (const auto &special : special_tokens)
    {
        auto it = vocab_.find(special);
        if (it != vocab_.end())
        {
            special_token_ids_.insert(it->second);
        }
    }

    return true;
}

} // namespace tokenflux::tokenize
