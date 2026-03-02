#include "tokenize_pipeline_internal.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize::detail
{
static bool get_file_stat(const std::string &path, std::uint64_t &file_size, std::int64_t &mtime)
{
    std::error_code ec;
    file_size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        return false;
    }
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec)
    {
        return false;
    }
    mtime = std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
    return true;
}

std::string make_tokenizer_fingerprint(const std::string &path)
{
    std::error_code ec;
    std::filesystem::path abs_path = std::filesystem::absolute(path, ec);
    std::uint64_t sz = 0;
    std::int64_t mt = 0;
    if (!get_file_stat(path, sz, mt))
    {
        return {};
    }
    std::ostringstream oss;
    oss << normalize_path_for_compare(abs_path.string()) << "|" << sz << "|" << mt;
    return oss.str();
}

std::string make_run_signature(const PartSignature &sig)
{
    std::ostringstream oss;
    oss << "tokenizer_fingerprint=" << sig.tokenizer_fingerprint << ";";
    oss << "text_field=" << sig.text_field << ";";
    oss << "min_chars=" << sig.min_chars << ";";
    oss << "max_chars=" << sig.max_chars << ";";
    oss << "bos_id=" << sig.bos_id << ";";
    oss << "eos_id=" << sig.eos_id << ";";
    oss << "dtype_bytes=" << sig.dtype_bytes;
    return oss.str();
}

static bool parse_completion_record_line(const std::string &line, std::string &path_out, CompletionRecord &record_out)
{
    std::istringstream iss(line);
    std::string path;
    std::uint64_t source_size = 0;
    std::int64_t source_mtime = 0;
    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t num_tokens = 0;
    if (!(iss >> std::quoted(path) >> source_size >> source_mtime >> num_docs >> num_skipped >> num_tokens))
    {
        return false;
    }
    path_out = std::move(path);
    record_out.source_size = source_size;
    record_out.source_mtime = source_mtime;
    record_out.result.num_docs = num_docs;
    record_out.result.num_skipped = num_skipped;
    record_out.result.num_tokens = num_tokens;
    record_out.result.reused = true;
    return true;
}

bool init_completion_records(const std::filesystem::path &path, const std::string &run_signature, bool resume,
                                    std::unordered_map<std::string, CompletionRecord> &records, std::string &err)
{
    records.clear();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec)
    {
        err = "failed to create completion dir: " + path.parent_path().string();
        return false;
    }

    bool rewrite_file = !resume;
    if (resume && std::filesystem::exists(path))
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            err = "failed to open completion list: " + path.string();
            return false;
        }
        std::string first_line;
        if (!std::getline(in, first_line))
        {
            rewrite_file = true;
        }
        else if (first_line != "signature=" + run_signature)
        {
            rewrite_file = true;
        }
        else
        {
            std::string line;
            while (std::getline(in, line))
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                if (line.empty() || line[0] == '#')
                {
                    continue;
                }
                std::string source_path;
                CompletionRecord rec;
                if (!parse_completion_record_line(line, source_path, rec))
                {
                    continue;
                }
                records[normalize_path_for_compare(source_path)] = rec;
            }
        }
    }
    else if (!resume)
    {
        rewrite_file = true;
    }

    if (rewrite_file || !std::filesystem::exists(path))
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            err = "failed to initialize completion list: " + path.string();
            return false;
        }
        out << "signature=" << run_signature << "\n";
        out << "# path size mtime docs skipped tokens\n";
        out.flush();
        if (!out)
        {
            err = "failed to flush completion list: " + path.string();
            return false;
        }
        records.clear();
    }

    return true;
}

bool append_completion_record(const std::filesystem::path &path, const FileTask &task, const PartResult &result,
                                     std::mutex &mu, std::string &err)
{
    std::lock_guard<std::mutex> lock(mu);
    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out)
    {
        err = "failed to append completion list: " + path.string();
        return false;
    }
    out << std::quoted(task.source) << "\t" << task.file_size << "\t" << task.file_mtime << "\t" << result.num_docs
        << "\t" << result.num_skipped << "\t" << result.num_tokens << "\n";
    out.flush();
    if (!out)
    {
        err = "failed to flush completion list: " + path.string();
        return false;
    }
    return true;
}

bool completion_is_reusable(const FileTask &task, const std::unordered_map<std::string, CompletionRecord> &records,
                                   PartResult &result)
{
    auto it = records.find(task.normalized_path);
    if (it == records.end())
    {
        return false;
    }
    if (it->second.source_size != task.file_size || it->second.source_mtime != task.file_mtime)
    {
        return false;
    }
    result = it->second.result;
    result.reused = true;
    return true;
}

std::string now_string()
{
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

std::size_t utf8_char_count(const std::string &s)
{
    std::size_t i = 0;
    std::size_t n = 0;
    while (i < s.size())
    {
        std::size_t prev = i;
        std::uint32_t cp = 0;
        if (!next_codepoint(s, i, cp))
        {
            break;
        }
        if (i <= prev)
        {
            break;
        }
        ++n;
    }
    return n;
}

std::size_t derive_stream_target_bytes(std::uint64_t file_size, std::size_t worker_threads)
{
    const std::uint64_t min_target = 4ull * 1024ull * 1024ull;
    const std::uint64_t max_target = 64ull * 1024ull * 1024ull;
    if (worker_threads == 0)
    {
        worker_threads = 1;
    }
    if (file_size == 0)
    {
        return static_cast<std::size_t>(min_target);
    }
    std::uint64_t desired_parts = std::max<std::uint64_t>(1, std::min<std::uint64_t>(worker_threads * 4ull, 128ull));
    std::uint64_t target = (file_size + desired_parts - 1ull) / desired_parts;
    target = std::max<std::uint64_t>(target, min_target);
    target = std::min<std::uint64_t>(target, max_target);
    return static_cast<std::size_t>(target);
}

bool prescan_total_docs(const std::vector<FileTask> &tasks, const Args &args, std::uint64_t &total_docs,
                               std::string &err)
{
    total_docs = 0;
    ProgressTracker scan_progress(tasks.size(), "scanning", 1000, "files");
    constexpr std::uint64_t report_docs_batch = 512;

    for (const auto &task : tasks)
    {
        std::uint64_t local_docs = 0;
        std::uint64_t local_reported = 0;
        std::string local_err;
        bool ok = for_each_text_record(
            task.path, args.text_field,
            [&](const std::string &incoming_text) {
                if (incoming_text.empty())
                {
                    return;
                }
                std::size_t chars = utf8_char_count(incoming_text);
                if (chars < args.min_chars)
                {
                    return;
                }
                ++local_docs;
                std::uint64_t pending = local_docs - local_reported;
                if (pending >= report_docs_batch)
                {
                    scan_progress.add(0, pending);
                    local_reported = local_docs;
                }
            },
            local_err);
        if (!ok)
        {
            err = local_err.empty() ? ("failed to scan input file: " + task.source) : local_err;
            return false;
        }
        if (local_docs > local_reported)
        {
            scan_progress.add(0, local_docs - local_reported);
        }
        total_docs += local_docs;
        scan_progress.add(1, 0);
    }

    scan_progress.finish();
    return true;
}

} // namespace tokenflux::tokenize::detail

