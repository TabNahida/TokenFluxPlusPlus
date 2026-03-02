#pragma once

#include "tokenize_common.hpp"

#include <string>

namespace tokenflux
{
namespace tokenize
{
namespace detail
{

bool parse_tokenizer_json_content(const std::string &content, TokenizerData &out);

} // namespace detail
} // namespace tokenize
} // namespace tokenflux
