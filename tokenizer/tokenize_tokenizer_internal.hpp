#pragma once

#include "tokenize_common.hpp"

#include <string>

namespace tokenflux::tokenize::detail
{

bool parse_tokenizer_json_content(const std::string &content, TokenizerData &out);

} // namespace tokenflux::tokenize::detail
