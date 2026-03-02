#include "tokenize_tokenizer_internal.hpp"

#include <cctype>
#include <limits>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize::detail
{

static void skip_ws(const std::string &s, std::size_t &i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
    {
        ++i;
    }
}

static bool parse_hex4(const std::string &s, std::size_t i, std::uint32_t &out)
{
    if (i + 4 > s.size())
    {
        return false;
    }
    std::uint32_t val = 0;
    for (std::size_t k = 0; k < 4; ++k)
    {
        char c = s[i + k];
        std::uint32_t v = 0;
        if (c >= '0' && c <= '9')
        {
            v = static_cast<std::uint32_t>(c - '0');
        }
        else if (c >= 'a' && c <= 'f')
        {
            v = static_cast<std::uint32_t>(10 + c - 'a');
        }
        else if (c >= 'A' && c <= 'F')
        {
            v = static_cast<std::uint32_t>(10 + c - 'A');
        }
        else
        {
            return false;
        }
        val = (val << 4) | v;
    }
    out = val;
    return true;
}

static bool parse_json_string(const std::string &s, std::size_t &i, std::string &out)
{
    if (i >= s.size() || s[i] != '"')
    {
        return false;
    }
    ++i;
    out.clear();
    while (i < s.size())
    {
        char c = s[i++];
        if (c == '"')
        {
            return true;
        }
        if (c == '\\')
        {
            if (i >= s.size())
            {
                return false;
            }
            char esc = s[i++];
            switch (esc)
            {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                std::uint32_t cp = 0;
                if (!parse_hex4(s, i, cp))
                {
                    return false;
                }
                i += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF)
                {
                    if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u')
                    {
                        std::uint32_t low = 0;
                        if (parse_hex4(s, i + 2, low) && low >= 0xDC00 && low <= 0xDFFF)
                        {
                            i += 6;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                        }
                    }
                }
                append_utf8(cp, out);
                break;
            }
            default:
                out.push_back(esc);
                break;
            }
        }
        else
        {
            out.push_back(c);
        }
    }
    return false;
}

static bool skip_json_value(const std::string &s, std::size_t &i);

static bool skip_json_number(const std::string &s, std::size_t &i)
{
    if (i < s.size() && (s[i] == '-' || s[i] == '+'))
    {
        ++i;
    }
    bool has_digit = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        has_digit = true;
        ++i;
    }
    if (i < s.size() && s[i] == '.')
    {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+'))
        {
            ++i;
        }
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    return has_digit;
}

static bool skip_json_literal(const std::string &s, std::size_t &i, const char *lit)
{
    std::size_t n = std::char_traits<char>::length(lit);
    if (i + n > s.size())
    {
        return false;
    }
    if (s.compare(i, n, lit) != 0)
    {
        return false;
    }
    i += n;
    return true;
}

static bool skip_json_object(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool skip_json_array(const std::string &s, std::size_t &i)
{
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool skip_json_value(const std::string &s, std::size_t &i)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    char c = s[i];
    if (c == '"')
    {
        std::string tmp;
        return parse_json_string(s, i, tmp);
    }
    if (c == '{')
    {
        return skip_json_object(s, i);
    }
    if (c == '[')
    {
        return skip_json_array(s, i);
    }
    if (c == 't')
    {
        return skip_json_literal(s, i, "true");
    }
    if (c == 'f')
    {
        return skip_json_literal(s, i, "false");
    }
    if (c == 'n')
    {
        return skip_json_literal(s, i, "null");
    }
    return skip_json_number(s, i);
}

static bool parse_json_uint(const std::string &s, std::size_t &i, std::uint32_t &out)
{
    skip_ws(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
    {
        return false;
    }
    std::uint64_t val = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        val = val * 10 + static_cast<std::uint64_t>(s[i] - '0');
        if (val > std::numeric_limits<std::uint32_t>::max())
        {
            return false;
        }
        ++i;
    }
    out = static_cast<std::uint32_t>(val);
    return true;
}

static bool parse_json_double(const std::string &s, std::size_t &i, double &out)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    std::size_t start = i;
    if (s[i] == '+' || s[i] == '-')
    {
        ++i;
    }
    bool has_digit = false;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        has_digit = true;
        ++i;
    }
    if (i < s.size() && s[i] == '.')
    {
        ++i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
    {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-'))
        {
            ++i;
        }
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        {
            has_digit = true;
            ++i;
        }
    }
    if (!has_digit)
    {
        return false;
    }
    try
    {
        out = std::stod(s.substr(start, i - start));
    }
    catch (...)
    {
        return false;
    }
    return true;
}

static bool parse_vocab_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        std::uint32_t val = 0;
        if (!parse_json_uint(s, i, val))
        {
            return false;
        }
        out.vocab[std::move(key)] = val;
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_merge_line(const std::string &line, std::pair<std::string, std::string> &out)
{
    auto pos = line.find(' ');
    if (pos == std::string::npos || pos + 1 >= line.size())
    {
        return false;
    }
    out.first = line.substr(0, pos);
    out.second = line.substr(pos + 1);
    return true;
}

static bool parse_merges_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] == '"')
        {
            std::string merge_line;
            if (!parse_json_string(s, i, merge_line))
            {
                return false;
            }
            std::pair<std::string, std::string> m;
            if (parse_merge_line(merge_line, m))
            {
                out.merges.push_back(std::move(m));
            }
        }
        else if (s[i] == '[')
        {
            ++i;
            skip_ws(s, i);
            std::string left;
            std::string right;
            bool ok = parse_json_string(s, i, left);
            skip_ws(s, i);
            if (ok && i < s.size() && s[i] == ',')
            {
                ++i;
                skip_ws(s, i);
                ok = parse_json_string(s, i, right);
            }
            while (i < s.size())
            {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    break;
                }
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    if (!skip_json_value(s, i))
                    {
                        return false;
                    }
                    continue;
                }
                return false;
            }
            if (ok && !left.empty() && !right.empty())
            {
                out.merges.push_back({std::move(left), std::move(right)});
            }
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static void set_model_type_from_string(const std::string &type, TokenizerData &out)
{
    if (type == "BPE")
    {
        out.model_type = ModelType::bpe;
        return;
    }
    if (type == "WordPiece")
    {
        out.model_type = ModelType::wordpiece;
        return;
    }
    if (type == "Unigram")
    {
        out.model_type = ModelType::unigram;
    }
}

static void set_pretokenizer_type_from_string(const std::string &type, TokenizerData &out)
{
    if (type == "ByteLevel")
    {
        out.pretokenizer_type = PretokenizerType::byte_level;
        return;
    }
    if (type == "WhitespaceSplit" || type == "Whitespace")
    {
        out.pretokenizer_type = PretokenizerType::whitespace;
    }
}

static bool parse_unigram_vocab_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    std::size_t index = 0;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] == '[')
        {
            ++i;
            skip_ws(s, i);
            std::string token;
            double score = 0.0;
            bool ok = parse_json_string(s, i, token);
            skip_ws(s, i);
            if (ok && i < s.size() && s[i] == ',')
            {
                ++i;
                skip_ws(s, i);
                ok = parse_json_double(s, i, score);
            }
            while (i < s.size())
            {
                skip_ws(s, i);
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    break;
                }
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    if (!skip_json_value(s, i))
                    {
                        return false;
                    }
                    continue;
                }
                return false;
            }
            if (ok && !token.empty())
            {
                out.unigram_vocab.push_back({token, score});
                out.vocab[token] = static_cast<std::uint32_t>(index);
                ++index;
            }
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_pre_tokenizer_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size())
    {
        return false;
    }
    if (s.compare(i, 4, "null") == 0)
    {
        i += 4;
        return true;
    }
    if (s[i] != '{')
    {
        return skip_json_value(s, i);
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "type")
        {
            std::string type;
            if (!parse_json_string(s, i, type))
            {
                return false;
            }
            set_pretokenizer_type_from_string(type, out);
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_added_tokens_array(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == ']')
        {
            ++i;
            return true;
        }
        if (s[i] != '{')
        {
            if (!skip_json_value(s, i))
            {
                return false;
            }
        }
        else
        {
            ++i;
            std::string content;
            std::uint32_t id = 0;
            bool has_content = false;
            bool has_id = false;
            while (true)
            {
                skip_ws(s, i);
                if (i >= s.size())
                {
                    return false;
                }
                if (s[i] == '}')
                {
                    ++i;
                    break;
                }
                std::string key;
                if (!parse_json_string(s, i, key))
                {
                    return false;
                }
                skip_ws(s, i);
                if (i >= s.size() || s[i] != ':')
                {
                    return false;
                }
                ++i;
                if (key == "id")
                {
                    std::uint32_t x = 0;
                    if (!parse_json_uint(s, i, x))
                    {
                        return false;
                    }
                    id = x;
                    has_id = true;
                }
                else if (key == "content")
                {
                    std::string x;
                    if (!parse_json_string(s, i, x))
                    {
                        return false;
                    }
                    content = std::move(x);
                    has_content = true;
                }
                else if (!skip_json_value(s, i))
                {
                    return false;
                }
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == '}')
                {
                    ++i;
                    break;
                }
                return false;
            }
            if (has_content && has_id)
            {
                out.added_tokens[content] = id;
            }
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == ']')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_model_object(const std::string &s, std::size_t &i, TokenizerData &out)
{
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(s, i);
        if (i >= s.size())
        {
            return false;
        }
        if (s[i] == '}')
        {
            ++i;
            return true;
        }
        std::string key;
        if (!parse_json_string(s, i, key))
        {
            return false;
        }
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "vocab")
        {
            skip_ws(s, i);
            if (i < s.size() && s[i] == '{')
            {
                if (!parse_vocab_object(s, i, out))
                {
                    return false;
                }
            }
            else if (i < s.size() && s[i] == '[')
            {
                if (!parse_unigram_vocab_array(s, i, out))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        else if (key == "merges")
        {
            if (!parse_merges_array(s, i, out))
            {
                return false;
            }
        }
        else if (key == "unk_token")
        {
            std::string unk;
            if (!parse_json_string(s, i, unk))
            {
                return false;
            }
            out.unk_token = std::move(unk);
        }
        else if (key == "type")
        {
            std::string type;
            if (!parse_json_string(s, i, type))
            {
                return false;
            }
            set_model_type_from_string(type, out);
        }
        else if (key == "continuing_subword_prefix")
        {
            std::string prefix;
            if (!parse_json_string(s, i, prefix))
            {
                return false;
            }
            out.continuing_subword_prefix = std::move(prefix);
        }
        else if (!skip_json_value(s, i))
        {
            return false;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            return true;
        }
        return false;
    }
}

static bool parse_tokenizer_json(const std::string &content, TokenizerData &out)
{
    std::size_t i = 0;
    skip_ws(content, i);
    if (i >= content.size() || content[i] != '{')
    {
        return false;
    }
    ++i;
    while (true)
    {
        skip_ws(content, i);
        if (i >= content.size())
        {
            return false;
        }
        if (content[i] == '}')
        {
            ++i;
            break;
        }
        std::string key;
        if (!parse_json_string(content, i, key))
        {
            return false;
        }
        skip_ws(content, i);
        if (i >= content.size() || content[i] != ':')
        {
            return false;
        }
        ++i;
        if (key == "model")
        {
            if (!parse_model_object(content, i, out))
            {
                return false;
            }
        }
        else if (key == "added_tokens")
        {
            if (!parse_added_tokens_array(content, i, out))
            {
                return false;
            }
        }
        else if (key == "pre_tokenizer")
        {
            if (!parse_pre_tokenizer_object(content, i, out))
            {
                return false;
            }
        }
        else if (!skip_json_value(content, i))
        {
            return false;
        }
        skip_ws(content, i);
        if (i < content.size() && content[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < content.size() && content[i] == '}')
        {
            ++i;
            break;
        }
        return false;
    }
    for (const auto &kv : out.added_tokens)
    {
        out.vocab[kv.first] = kv.second;
    }
    return !out.vocab.empty();
}
bool parse_tokenizer_json_content(const std::string &content, TokenizerData &out)
{
    return parse_tokenizer_json(content, out);
}

} // namespace tokenflux::tokenize::detail
