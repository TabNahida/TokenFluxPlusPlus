#pragma once

#include "tokenize_common.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tokenflux
{
namespace tokenize
{

class TokenizerEncoder
{
  public:
    bool load(const std::string &path, std::string &err);
    std::size_t vocab_size() const;
    std::string model_name() const;
    bool token_to_id(const std::string &token, std::uint32_t &id) const;
    std::string id_to_token(std::uint32_t id) const;
    void encode_text_append(const std::string &text, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache,
                            std::vector<std::uint32_t> &out_ids) const;
    std::string decode(const std::vector<std::uint32_t> &token_ids, bool skip_special_tokens = false,
                       bool clean_up_tokenization_spaces = true) const;
    std::vector<std::string> decode_batch(const std::vector<std::vector<std::uint32_t>> &token_ids_batch,
                                          bool skip_special_tokens = false,
                                          bool clean_up_tokenization_spaces = true) const;

  private:
    struct UnigramTokenState
    {
        std::string token;
        std::vector<std::string> cps;
        double score = 0.0;
        std::uint32_t id = 0;
    };

    std::uint32_t ensure_symbol(const std::string &sym);
    static std::uint64_t pair_key(std::uint32_t a, std::uint32_t b);
    const std::vector<std::uint32_t> &encode_piece_bpe(
        const std::string &encoded, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const;
    const std::vector<std::uint32_t> &encode_piece_wordpiece(
        const std::string &piece, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const;
    static bool unigram_match(const std::vector<std::string> &token_cps, const std::vector<std::string> &word_cps,
                              std::size_t pos);
    const std::vector<std::uint32_t> &encode_piece_unigram(
        const std::string &piece, std::unordered_map<std::string, std::vector<std::uint32_t>> &cache) const;

    std::unordered_map<std::string, std::uint32_t> vocab_;
    ModelType model_type_ = ModelType::bpe;
    PretokenizerType pretokenizer_type_ = PretokenizerType::byte_level;
    std::string continuing_subword_prefix_ = "##";
    std::string unk_token_;
    bool has_unk_ = false;
    std::uint32_t unk_id_ = 0;
    std::vector<std::string> symbols_;
    std::unordered_map<std::string, std::uint32_t> symbol_to_id_;
    std::unordered_map<std::uint64_t, MergeRule> merge_rules_;
    std::vector<UnigramTokenState> unigram_tokens_;
    std::unordered_map<std::string, std::vector<std::size_t>> unigram_index_;
    std::array<std::string, 256> byte_to_unicode_{};

    // id to token mapping for decoding
    std::vector<std::string> id_to_token_list_;
    std::unordered_set<std::uint32_t> special_token_ids_;
    std::unordered_map<std::uint32_t, std::uint8_t> unicode_to_byte_;

    // decode helper methods
    std::string byte_level_decode(const std::string &token) const;
    std::string clean_up_tokenization(const std::string &text) const;
};

} // namespace tokenize
} // namespace tokenflux
