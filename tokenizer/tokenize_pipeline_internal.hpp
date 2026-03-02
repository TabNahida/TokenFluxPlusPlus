#pragma once

#include "tokenize_common.hpp"
#include "tokenize_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenflux
{
namespace tokenize
{
namespace detail
{

std::string make_tokenizer_fingerprint(const std::string &path);
std::string make_run_signature(const PartSignature &sig);

bool init_completion_records(const std::filesystem::path &path, const std::string &run_signature, bool resume,
                             std::unordered_map<std::string, CompletionRecord> &records, std::string &err);

bool append_completion_record(const std::filesystem::path &path, const FileTask &task, const PartResult &result,
                              std::mutex &mu, std::string &err);

bool completion_is_reusable(const FileTask &task, const std::unordered_map<std::string, CompletionRecord> &records,
                            PartResult &result);

std::string now_string();
std::size_t utf8_char_count(const std::string &s);
std::size_t derive_stream_target_bytes(std::uint64_t file_size, std::size_t worker_threads);

bool prescan_total_docs(const std::vector<FileTask> &tasks, const Args &args, std::uint64_t &total_docs,
                        std::string &err);

bool process_file_to_shards(const FileTask &task, const Args &args, const TokenizerEncoder &tokenizer,
                            const PartSignature &sig,
                            const std::function<bool(const std::vector<std::uint32_t> &, std::string &)> &append_tokens,
                            std::size_t encode_threads, PartResult &result,
                            const std::function<void(std::uint64_t)> &report_docs, std::string &err);

} // namespace detail
} // namespace tokenize
} // namespace tokenflux
