#include "tokenize_tokenizer.hpp"

#include "tokenflux_lib.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <utility>

namespace tokenflux::tokenize
{

std::string TokenizerEncoder::id_to_token(std::uint32_t id) const
{
    if (id >= id_to_token_list_.size())
    {
        return "";
    }
    return id_to_token_list_[id];
}

std::string TokenizerEncoder::decode(const std::vector<std::uint32_t> &token_ids, bool skip_special_tokens,
                                     bool clean_up_tokenization_spaces) const
{
    std::string text;

    for (std::uint32_t token_id : token_ids)
    {
        // Skip special tokens if requested
        if (skip_special_tokens && special_token_ids_.count(token_id) > 0)
        {
            continue;
        }

        // Get token string
        std::string token = id_to_token(token_id);
        if (token.empty())
        {
            continue;
        }

        // Handle continuing subword prefix for BPE and WordPiece
        if (model_type_ == ModelType::bpe || model_type_ == ModelType::wordpiece)
        {
            if (token.size() >= 2 && token.substr(0, 2) == "##")
            {
                token = token.substr(2);
            }
        }

        // Decode byte-level encoding if applicable
        if (pretokenizer_type_ == PretokenizerType::byte_level)
        {
            token = byte_level_decode(token);
        }

        text += token;
    }

    // Clean up tokenization spaces if requested
    if (clean_up_tokenization_spaces)
    {
        text = clean_up_tokenization(text);
    }

    return text;
}

std::vector<std::string> TokenizerEncoder::decode_batch(const std::vector<std::vector<std::uint32_t>> &token_ids_batch,
                                                        bool skip_special_tokens,
                                                        bool clean_up_tokenization_spaces) const
{
    std::vector<std::string> results;
    results.reserve(token_ids_batch.size());

    for (const auto &token_ids : token_ids_batch)
    {
        results.push_back(decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces));
    }

    return results;
}

std::string TokenizerEncoder::byte_level_decode(const std::string &token) const
{
    std::string result;
    result.reserve(token.size());

    std::size_t i = 0;
    while (i < token.size())
    {
        // Decode UTF-8 character to get codepoint
        std::uint32_t cp = 0;
        if (!next_codepoint(token, i, cp))
        {
            break;
        }

        // Check if this is a byte-level encoded character
        auto it = unicode_to_byte_.find(cp);
        if (it != unicode_to_byte_.end())
        {
            // Convert back to original byte
            result += static_cast<char>(it->second);
        }
        else
        {
            // Not a byte-level encoded character, keep as-is
            // Re-decode the character to append it
            std::string char_utf8;
            append_utf8(cp, char_utf8);
            result += char_utf8;
        }
    }

    return result;
}

std::string TokenizerEncoder::clean_up_tokenization(const std::string &text) const
{
    std::string cleaned;
    cleaned.reserve(text.size());

    bool prev_was_space = false;

    for (char c : text)
    {
        if (std::isspace(static_cast<unsigned char>(c)))
        {
            if (!prev_was_space)
            {
                cleaned += ' ';
                prev_was_space = true;
            }
        }
        else
        {
            cleaned += c;
            prev_was_space = false;
        }
    }

    // Trim leading and trailing spaces
    size_t start = cleaned.find_first_not_of(' ');
    size_t end = cleaned.find_last_not_of(' ');

    if (start == std::string::npos)
    {
        return "";
    }

    return cleaned.substr(start, end - start + 1);
}

} // namespace tokenflux::tokenize