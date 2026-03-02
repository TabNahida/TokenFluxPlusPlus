#include "tokenize_tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize
{
std::size_t TokenizerEncoder::vocab_size() const
{
    return vocab_.size();
}

std::string TokenizerEncoder::model_name() const
{
    if (model_type_ == ModelType::bpe)
    {
        return "BPE";
    }
    if (model_type_ == ModelType::wordpiece)
    {
        return "WordPiece";
    }
    return "Unigram";
}

bool TokenizerEncoder::token_to_id(const std::string &token, std::uint32_t &id) const
{
    auto it = vocab_.find(token);
    if (it == vocab_.end())
    {
        return false;
    }
    id = it->second;
    return true;
}

void TokenizerEncoder::encode_text_append(const std::string &text,
                                          std::unordered_map<std::string, std::vector<std::uint32_t>> &cache,
                                          std::vector<std::uint32_t> &out_ids) const
{
    std::vector<std::string> pieces;
    if (pretokenizer_type_ == PretokenizerType::byte_level)
    {
        pieces = pretokenize(text);
    }
    else
    {
        pieces = split_whitespace_words(text);
    }
    for (const auto &piece : pieces)
    {
        if (piece.empty())
        {
            continue;
        }
        const std::vector<std::uint32_t> *ids = nullptr;
        if (model_type_ == ModelType::bpe)
        {
            std::string encoded = piece;
            if (pretokenizer_type_ == PretokenizerType::byte_level)
            {
                encoded = byte_level_encode(piece, byte_to_unicode_);
            }
            ids = &encode_piece_bpe(encoded, cache);
        }
        else if (model_type_ == ModelType::wordpiece)
        {
            ids = &encode_piece_wordpiece(piece, cache);
        }
        else
        {
            ids = &encode_piece_unigram(piece, cache);
        }
        if (ids)
        {
            out_ids.insert(out_ids.end(), ids->begin(), ids->end());
        }
    }
}

std::uint32_t TokenizerEncoder::ensure_symbol(const std::string &sym)
{
    auto it = symbol_to_id_.find(sym);
    if (it != symbol_to_id_.end())
    {
        return it->second;
    }
    std::uint32_t id = static_cast<std::uint32_t>(symbols_.size());
    symbols_.push_back(sym);
    symbol_to_id_.emplace(symbols_.back(), id);
    return id;
}

std::uint64_t TokenizerEncoder::pair_key(std::uint32_t a, std::uint32_t b)
{
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

const std::vector<std::uint32_t> &TokenizerEncoder::encode_piece_bpe(
    const std::string &encoded, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
{
    std::string cache_key = std::string("bpe:") + encoded;
    auto it_cache = cache.find(cache_key);
    if (it_cache != cache.end())
    {
        return it_cache->second;
    }

    std::vector<std::uint32_t> symbols;
    symbols.reserve(encoded.size());
    std::size_t i = 0;
    while (i < encoded.size())
    {
        std::size_t prev = i;
        std::uint32_t cp = 0;
        if (!next_codepoint(encoded, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        auto it_sym = symbol_to_id_.find(encoded.substr(prev, i - prev));
        if (it_sym == symbol_to_id_.end())
        {
            symbols.clear();
            break;
        }
        symbols.push_back(it_sym->second);
    }

    while (symbols.size() >= 2)
    {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best_pos = static_cast<std::size_t>(-1);
        std::uint32_t best_left = 0;
        std::uint32_t best_right = 0;
        std::uint32_t best_merged = 0;
        for (std::size_t pos = 0; pos + 1 < symbols.size(); ++pos)
        {
            std::uint64_t key = pair_key(symbols[pos], symbols[pos + 1]);
            auto it_rule = merge_rules_.find(key);
            if (it_rule == merge_rules_.end())
            {
                continue;
            }
            if (it_rule->second.rank < best_rank)
            {
                best_rank = it_rule->second.rank;
                best_pos = pos;
                best_left = symbols[pos];
                best_right = symbols[pos + 1];
                best_merged = it_rule->second.merged_symbol;
            }
        }
        if (best_pos == static_cast<std::size_t>(-1))
        {
            break;
        }

        std::vector<std::uint32_t> merged;
        merged.reserve(symbols.size());
        for (std::size_t p = 0; p < symbols.size();)
        {
            if (p + 1 < symbols.size() && symbols[p] == best_left && symbols[p + 1] == best_right)
            {
                merged.push_back(best_merged);
                p += 2;
            }
            else
            {
                merged.push_back(symbols[p]);
                ++p;
            }
        }
        symbols.swap(merged);
    }

    std::vector<std::uint32_t> ids;
    ids.reserve(symbols.size());
    for (std::uint32_t sid : symbols)
    {
        if (sid >= symbols_.size())
        {
            continue;
        }
        const auto &tok = symbols_[sid];
        auto it_vocab = vocab_.find(tok);
        if (it_vocab != vocab_.end())
        {
            ids.push_back(it_vocab->second);
        }
        else if (has_unk_)
        {
            ids.push_back(unk_id_);
        }
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(ids));
    return inserted.first->second;
}

const std::vector<std::uint32_t> &TokenizerEncoder::encode_piece_wordpiece(
    const std::string &piece, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
{
    std::string cache_key = std::string("wp:") + piece;
    auto it_cache = cache.find(cache_key);
    if (it_cache != cache.end())
    {
        return it_cache->second;
    }

    std::vector<std::uint32_t> ids;
    auto cps = split_codepoints_utf8(piece);
    if (!cps.empty())
    {
        std::size_t start = 0;
        bool fallback_to_unk = false;
        while (start < cps.size())
        {
            bool found = false;
            std::size_t best_end = start;
            std::uint32_t best_id = 0;
            for (std::size_t end = cps.size(); end > start; --end)
            {
                std::string cand;
                for (std::size_t k = start; k < end; ++k)
                {
                    cand += cps[k];
                }
                if (start > 0 && !continuing_subword_prefix_.empty())
                {
                    cand = continuing_subword_prefix_ + cand;
                }
                auto it = vocab_.find(cand);
                if (it != vocab_.end())
                {
                    best_id = it->second;
                    best_end = end;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                fallback_to_unk = true;
                break;
            }
            ids.push_back(best_id);
            start = best_end;
        }
        if (fallback_to_unk && has_unk_)
        {
            ids.clear();
            ids.push_back(unk_id_);
        }
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(ids));
    return inserted.first->second;
}

bool TokenizerEncoder::unigram_match(const std::vector<std::string> &token_cps, const std::vector<std::string> &word_cps,
                                     std::size_t pos)
{
    if (pos + token_cps.size() > word_cps.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < token_cps.size(); ++i)
    {
        if (token_cps[i] != word_cps[pos + i])
        {
            return false;
        }
    }
    return true;
}

const std::vector<std::uint32_t> &TokenizerEncoder::encode_piece_unigram(
    const std::string &piece, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const
{
    std::string cache_key = std::string("uni:") + piece;
    auto it_cache = cache.find(cache_key);
    if (it_cache != cache.end())
    {
        return it_cache->second;
    }

    std::vector<std::uint32_t> ids;
    auto cps = split_codepoints_utf8(piece);
    if (!cps.empty() && !unigram_tokens_.empty())
    {
        const double neg_inf = -1e100;
        std::size_t n = cps.size();
        std::vector<double> best(n + 1, neg_inf);
        std::vector<int> prev_pos(n + 1, -1);
        std::vector<int> prev_tok(n + 1, -1);
        best[0] = 0.0;

        for (std::size_t i = 0; i < n; ++i)
        {
            if (best[i] <= neg_inf / 2.0)
            {
                continue;
            }
            auto it = unigram_index_.find(cps[i]);
            if (it == unigram_index_.end())
            {
                continue;
            }
            for (std::size_t tok_idx : it->second)
            {
                const auto &tok = unigram_tokens_[tok_idx];
                if (!unigram_match(tok.cps, cps, i))
                {
                    continue;
                }
                std::size_t j = i + tok.cps.size();
                double cand = best[i] + tok.score;
                if (cand > best[j])
                {
                    best[j] = cand;
                    prev_pos[j] = static_cast<int>(i);
                    prev_tok[j] = static_cast<int>(tok_idx);
                }
            }
        }

        if (best[n] <= neg_inf / 2.0)
        {
            if (has_unk_)
            {
                ids.push_back(unk_id_);
            }
        }
        else
        {
            std::vector<std::uint32_t> rev;
            int cur = static_cast<int>(n);
            while (cur > 0)
            {
                int t = prev_tok[static_cast<std::size_t>(cur)];
                int p = prev_pos[static_cast<std::size_t>(cur)];
                if (t < 0 || p < 0)
                {
                    break;
                }
                rev.push_back(unigram_tokens_[static_cast<std::size_t>(t)].id);
                cur = p;
            }
            ids.assign(rev.rbegin(), rev.rend());
        }
    }

    auto inserted = cache.emplace(std::move(cache_key), std::move(ids));
    return inserted.first->second;
}

} // namespace tokenflux::tokenize

