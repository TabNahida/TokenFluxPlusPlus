#pragma once

#include "tokenize_tokenizer.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tokenflux
{

struct EncodeOptions
{
    std::string bos_token;
    std::string eos_token;
    bool reset_cache = false;
};

// Lightweight C++ facade over tokenflux::tokenize::TokenizerEncoder.
class Tokenizer
{
  public:
    Tokenizer() = default;

    explicit Tokenizer(const std::string &tokenizer_path)
    {
        load_or_throw(tokenizer_path);
    }

    bool load(const std::string &tokenizer_path, std::string &err)
    {
        cache_.clear();
        return encoder_.load(tokenizer_path, err);
    }

    void load_or_throw(const std::string &tokenizer_path)
    {
        std::string err;
        if (!load(tokenizer_path, err))
        {
            throw std::runtime_error(err);
        }
    }

    std::size_t vocab_size() const
    {
        return encoder_.vocab_size();
    }

    std::string model_name() const
    {
        return encoder_.model_name();
    }

    bool token_to_id(const std::string &token, std::uint32_t &id) const
    {
        return encoder_.token_to_id(token, id);
    }

    std::string id_to_token(std::uint32_t id) const
    {
        return encoder_.id_to_token(id);
    }

    void clear_cache()
    {
        cache_.clear();
    }

    std::vector<std::uint32_t> encode(const std::string &text, const EncodeOptions &options = {})
    {
        if (options.reset_cache)
        {
            cache_.clear();
        }

        std::int64_t bos_id = resolve_optional_token_id(options.bos_token, "bos");
        std::int64_t eos_id = resolve_optional_token_id(options.eos_token, "eos");

        std::vector<std::uint32_t> out;
        if (bos_id >= 0)
        {
            out.push_back(static_cast<std::uint32_t>(bos_id));
        }
        encoder_.encode_text_append(text, cache_, out);
        if (eos_id >= 0)
        {
            out.push_back(static_cast<std::uint32_t>(eos_id));
        }
        return out;
    }

    std::vector<std::vector<std::uint32_t>> encode_batch(const std::vector<std::string> &texts,
                                                         const EncodeOptions &options = {})
    {
        if (options.reset_cache)
        {
            cache_.clear();
        }

        std::int64_t bos_id = resolve_optional_token_id(options.bos_token, "bos");
        std::int64_t eos_id = resolve_optional_token_id(options.eos_token, "eos");

        std::vector<std::vector<std::uint32_t>> out;
        out.reserve(texts.size());
        for (const auto &text : texts)
        {
            std::vector<std::uint32_t> ids;
            if (bos_id >= 0)
            {
                ids.push_back(static_cast<std::uint32_t>(bos_id));
            }
            encoder_.encode_text_append(text, cache_, ids);
            if (eos_id >= 0)
            {
                ids.push_back(static_cast<std::uint32_t>(eos_id));
            }
            out.push_back(std::move(ids));
        }
        return out;
    }

    std::string decode(const std::vector<std::uint32_t> &token_ids, bool skip_special_tokens = false,
                       bool clean_up_tokenization_spaces = true) const
    {
        return encoder_.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces);
    }

    std::vector<std::string> decode_batch(const std::vector<std::vector<std::uint32_t>> &token_ids_batch,
                                          bool skip_special_tokens = false,
                                          bool clean_up_tokenization_spaces = true) const
    {
        return encoder_.decode_batch(token_ids_batch, skip_special_tokens, clean_up_tokenization_spaces);
    }

  private:
    std::int64_t resolve_optional_token_id(const std::string &token, const char *kind) const
    {
        if (token.empty())
        {
            return -1;
        }
        std::uint32_t id = 0;
        if (!encoder_.token_to_id(token, id))
        {
            throw std::runtime_error(std::string(kind) + " token not found in tokenizer: " + token);
        }
        return static_cast<std::int64_t>(id);
    }

    tokenflux::tokenize::TokenizerEncoder encoder_;
    std::unordered_map<std::string, std::vector<std::uint32_t>> cache_;
};

} // namespace tokenflux
