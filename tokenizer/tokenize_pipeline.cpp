#include "tokenize_pipeline.hpp"

#include "tokenize_pipeline_internal.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "input_source.hpp"
#include "tokenize_common.hpp"
#include "tokenize_tokenizer.hpp"

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize
{
static std::string shard_name(std::size_t idx)
{
    std::ostringstream oss;
    oss << "train_" << std::setw(6) << std::setfill('0') << idx << ".bin";
    return oss.str();
}

static bool parse_shard_index(const std::string &name, std::size_t &idx)
{
    if (!starts_with(name, "train_") || !ends_with(name, ".bin"))
    {
        return false;
    }
    std::string mid = name.substr(6, name.size() - 10);
    if (mid.empty())
    {
        return false;
    }
    std::uint64_t x = 0;
    if (!parse_u64_arg(mid, x))
    {
        return false;
    }
    if (x > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        return false;
    }
    idx = static_cast<std::size_t>(x);
    return true;
}

static bool remove_old_shards(const std::filesystem::path &out_dir);

class ShardWriter
{
  public:
    ShardWriter(std::filesystem::path out_dir, std::uint32_t dtype_bytes, std::uint64_t max_tokens_per_shard)
        : out_dir_(std::move(out_dir)), dtype_bytes_(dtype_bytes), max_tokens_per_shard_(max_tokens_per_shard)
    {
    }

    bool initialize(bool resume_existing, std::string &err)
    {
        std::lock_guard<std::mutex> lock(mu_);
        return initialize_locked(resume_existing, err);
    }

    bool append_tokens(const std::vector<std::uint32_t> &tokens, std::string &err)
    {
        if (tokens.empty())
        {
            return true;
        }
        std::lock_guard<std::mutex> lock(mu_);
        std::size_t offset = 0;
        while (offset < tokens.size())
        {
            if (!out_.is_open())
            {
                if (!open_new_shard_locked(err))
                {
                    return false;
                }
            }
            if (current_tokens_ >= max_tokens_per_shard_)
            {
                if (!rotate_shard_locked(err))
                {
                    return false;
                }
            }
            std::uint64_t remaining = max_tokens_per_shard_ - current_tokens_;
            std::size_t take = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(tokens.size() - offset)));
            if (!write_slice_locked(tokens.data() + offset, take, err))
            {
                return false;
            }
            current_tokens_ += static_cast<std::uint64_t>(take);
            total_tokens_ += static_cast<std::uint64_t>(take);
            offset += take;
        }
        return true;
    }

    bool finalize(std::vector<ShardInfo> &shards, std::uint64_t &total_tokens, std::string &err)
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!close_current_locked(true, err))
        {
            return false;
        }
        if (finalized_shards_.empty())
        {
            err = "no shards written";
            return false;
        }
        shards = finalized_shards_;
        total_tokens = total_tokens_;
        return true;
    }

    std::uint64_t total_tokens() const
    {
        return total_tokens_;
    }

  private:
    bool initialize_locked(bool resume_existing, std::string &err)
    {
        finalized_shards_.clear();
        total_tokens_ = 0;
        current_tokens_ = 0;
        shard_idx_ = 0;
        current_path_.clear();
        if (out_.is_open())
        {
            out_.close();
        }

        if (!resume_existing)
        {
            remove_old_shards(out_dir_);
            return open_new_shard_locked(err);
        }

        std::vector<std::pair<std::size_t, std::filesystem::path>> existing;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(out_dir_, ec); !ec && it != std::filesystem::directory_iterator();
             it.increment(ec))
        {
            if (!it->is_regular_file())
            {
                continue;
            }
            std::size_t idx = 0;
            std::string name = it->path().filename().string();
            if (parse_shard_index(name, idx))
            {
                existing.emplace_back(idx, it->path());
            }
        }
        if (ec)
        {
            err = "failed to list shard dir: " + out_dir_.string();
            return false;
        }
        if (existing.empty())
        {
            return open_new_shard_locked(err);
        }

        std::sort(existing.begin(), existing.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        for (std::size_t i = 0; i < existing.size(); ++i)
        {
            std::error_code sec;
            std::uint64_t bytes = std::filesystem::file_size(existing[i].second, sec);
            if (sec)
            {
                err = "failed to stat shard: " + existing[i].second.string();
                return false;
            }
            if (bytes % static_cast<std::uint64_t>(dtype_bytes_) != 0)
            {
                err = "corrupt shard file (size not aligned to dtype): " + existing[i].second.string();
                return false;
            }
            std::uint64_t tokens = bytes / static_cast<std::uint64_t>(dtype_bytes_);
            total_tokens_ += tokens;
            if (i + 1 < existing.size())
            {
                finalized_shards_.push_back({existing[i].second.filename().string(), tokens});
            }
            else
            {
                shard_idx_ = existing[i].first;
                current_path_ = existing[i].second;
                current_tokens_ = tokens;
            }
        }

        out_.open(current_path_, std::ios::binary | std::ios::app);
        if (!out_)
        {
            err = "failed to open shard for append: " + current_path_.string();
            return false;
        }
        return true;
    }

    bool open_new_shard_locked(std::string &err)
    {
        current_path_ = out_dir_ / shard_name(shard_idx_);
        out_.open(current_path_, std::ios::binary | std::ios::trunc);
        current_tokens_ = 0;
        if (!out_)
        {
            err = "failed to open shard for write: " + current_path_.string();
            return false;
        }
        return true;
    }

    bool close_current_locked(bool record, std::string &err)
    {
        if (!out_.is_open())
        {
            return true;
        }
        out_.flush();
        if (!out_)
        {
            err = "failed to flush shard: " + current_path_.string();
            return false;
        }
        out_.close();
        if (record)
        {
            finalized_shards_.push_back({current_path_.filename().string(), current_tokens_});
        }
        current_tokens_ = 0;
        return true;
    }

    bool rotate_shard_locked(std::string &err)
    {
        if (!close_current_locked(true, err))
        {
            return false;
        }
        ++shard_idx_;
        return open_new_shard_locked(err);
    }

    bool write_slice_locked(const std::uint32_t *tokens, std::size_t count, std::string &err)
    {
        if (count == 0)
        {
            return true;
        }
        if (dtype_bytes_ == 2)
        {
            tmp16_.clear();
            tmp16_.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                std::uint32_t v = tokens[i];
                if (v > std::numeric_limits<std::uint16_t>::max())
                {
                    err = "token id overflow for uint16 shards";
                    return false;
                }
                tmp16_.push_back(static_cast<std::uint16_t>(v));
            }
            out_.write(reinterpret_cast<const char *>(tmp16_.data()),
                       static_cast<std::streamsize>(tmp16_.size() * sizeof(std::uint16_t)));
        }
        else if (dtype_bytes_ == 4)
        {
            out_.write(reinterpret_cast<const char *>(tokens), static_cast<std::streamsize>(count * sizeof(std::uint32_t)));
        }
        else
        {
            err = "unsupported dtype bytes";
            return false;
        }
        if (!out_)
        {
            err = "failed to write shard: " + current_path_.string();
            return false;
        }
        return true;
    }

    std::filesystem::path out_dir_;
    std::uint32_t dtype_bytes_ = 2;
    std::uint64_t max_tokens_per_shard_ = 0;
    std::size_t shard_idx_ = 0;
    std::uint64_t current_tokens_ = 0;
    std::uint64_t total_tokens_ = 0;
    std::filesystem::path current_path_;
    std::ofstream out_;
    std::vector<std::uint16_t> tmp16_;
    std::vector<ShardInfo> finalized_shards_;
    std::mutex mu_;
};


static bool remove_old_shards(const std::filesystem::path &out_dir)
{
    std::error_code ec;
    for (std::filesystem::directory_iterator it(out_dir, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec))
    {
        if (!it->is_regular_file())
        {
            continue;
        }
        std::string name = it->path().filename().string();
        if (starts_with(name, "train_") && ends_with(name, ".bin"))
        {
            std::filesystem::remove(it->path(), ec);
            if (ec)
            {
                return false;
            }
        }
    }
    return !ec;
}

static bool write_meta_json(const std::filesystem::path &meta_path, const Args &args, const std::string &data_glob,
                            const std::string &data_list, const std::vector<std::string> &input_files, std::size_t vocab_size,
                            const std::string &dtype_name, std::int64_t eos_id, std::int64_t bos_id,
                            std::uint64_t num_docs, std::uint64_t num_skipped, std::uint64_t total_tokens,
                            const std::vector<ShardInfo> &shards, std::size_t reused_files, std::string &err)
{
    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        err = "failed to write meta.json: " + meta_path.string();
        return false;
    }
    out << "{\n";
    out << "  \"created_at\": \"" << json_escape(detail::now_string()) << "\",\n";
    out << "  \"tokenizer_path\": \"" << json_escape(args.tokenizer_path) << "\",\n";
    out << "  \"text_field\": \"" << json_escape(args.text_field) << "\",\n";
    out << "  \"data_glob\": \"" << json_escape(data_glob) << "\",\n";
    if (data_list.empty())
    {
        out << "  \"data_list\": null,\n";
    }
    else
    {
        out << "  \"data_list\": \"" << json_escape(data_list) << "\",\n";
    }
    out << "  \"num_input_files\": " << input_files.size() << ",\n";
    out << "  \"input_files\": [\n";
    for (std::size_t i = 0; i < input_files.size(); ++i)
    {
        out << "    \"" << json_escape(input_files[i]) << "\"";
        if (i + 1 < input_files.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"dtype\": \"" << dtype_name << "\",\n";
    bool has_eos_marker = eos_id >= 0 && args.add_eos;
    out << "  \"add_eos\": " << (args.add_eos ? "true" : "false") << ",\n";
    if (has_eos_marker)
    {
        out << "  \"eos_token\": \"" << json_escape(args.eos_token) << "\",\n";
        out << "  \"eos_id\": " << eos_id << ",\n";
    }
    else
    {
        out << "  \"eos_token\": null,\n";
        out << "  \"eos_id\": null,\n";
    }
    if (!args.bos_token.empty())
    {
        out << "  \"bos_token\": \"" << json_escape(args.bos_token) << "\",\n";
        out << "  \"bos_id\": " << bos_id << ",\n";
    }
    else
    {
        out << "  \"bos_token\": null,\n";
        out << "  \"bos_id\": null,\n";
    }
    out << "  \"max_tokens_per_shard\": " << args.max_tokens_per_shard << ",\n";
    out << "  \"max_memory_mb\": " << args.max_memory_mb << ",\n";
    out << "  \"num_docs\": " << num_docs << ",\n";
    out << "  \"num_skipped\": " << num_skipped << ",\n";
    out << "  \"total_tokens\": " << total_tokens << ",\n";
    out << "  \"shards\": [\n";
    for (std::size_t i = 0; i < shards.size(); ++i)
    {
        out << "    {\"file\": \"" << json_escape(shards[i].file) << "\", \"num_tokens\": " << shards[i].num_tokens
            << "}";
        if (i + 1 < shards.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"num_reused_files\": " << reused_files << ",\n";
    out << "  \"layout\": {\"shards\": \"shards\", \"completed\": \"cache/completed.list\", "
           "\"document_stream\": \"concatenated\", \"doc_boundary\": \""
        << (has_eos_marker ? "eos_token" : "none") << "\"}\n";
    out << "}\n";
    if (!out)
    {
        err = "failed to flush meta.json: " + meta_path.string();
        return false;
    }
    return true;
}

int run_tokenize(const Args &args)
{
    auto env = read_env_file(args.env_file);
    std::string data_glob = args.data_glob;
    std::string data_list = args.data_list;
    if (data_glob.empty())
    {
        auto it = env.find("DATA_PATH");
        if (it != env.end())
        {
            data_glob = it->second;
        }
    }
    if (data_list.empty())
    {
        auto it = env.find("DATA_LIST");
        if (it != env.end())
        {
            data_list = it->second;
        }
    }
    if (data_list.empty())
    {
        auto it = env.find("DATA_URL_LIST");
        if (it != env.end())
        {
            data_list = it->second;
        }
    }
    if (args.input_entries.empty() && data_glob.empty() && data_list.empty())
    {
        std::cerr << "DATA_PATH / DATA_LIST is missing. Set it in .env or pass --data-glob/--data-list.\n";
        return 1;
    }

    TokenizerEncoder tokenizer;
    std::string err;
    if (!tokenizer.load(args.tokenizer_path, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    std::uint32_t eos_id_u32 = 0;
    std::int64_t eos_id = -1;
    if (args.add_eos && !args.eos_token.empty())
    {
        if (!tokenizer.token_to_id(args.eos_token, eos_id_u32))
        {
            std::cerr << "eos token not found in tokenizer: " << args.eos_token << "\n";
            return 1;
        }
        eos_id = static_cast<std::int64_t>(eos_id_u32);
    }

    std::uint32_t bos_id_u32 = 0;
    std::int64_t bos_id = -1;
    if (!args.bos_token.empty())
    {
        if (!tokenizer.token_to_id(args.bos_token, bos_id_u32))
        {
            std::cerr << "bos token not found in tokenizer: " << args.bos_token << "\n";
            return 1;
        }
        bos_id = static_cast<std::int64_t>(bos_id_u32);
    }

    std::uint32_t dtype_bytes = tokenizer.vocab_size() <= std::numeric_limits<std::uint16_t>::max() ? 2u : 4u;
    std::string dtype_name = dtype_bytes == 2 ? "uint16" : "uint32";

    std::filesystem::path out_root = args.out_dir;
    std::filesystem::path shard_dir = out_root / "shards";
    std::filesystem::path cache_dir = out_root / "cache";
    std::filesystem::path completed_list_path = cache_dir / "completed.list";
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (ec)
    {
        std::cerr << "failed to create cache dir under: " << out_root.string() << "\n";
        return 1;
    }
    ec.clear();
    std::filesystem::create_directories(shard_dir, ec);
    if (ec)
    {
        std::cerr << "failed to create shard dir under: " << out_root.string() << "\n";
        return 1;
    }
    std::filesystem::path remote_cache_dir = cache_dir / "remote_inputs";

    std::string tokenizer_fp = detail::make_tokenizer_fingerprint(args.tokenizer_path);
    if (tokenizer_fp.empty())
    {
        std::cerr << "failed to fingerprint tokenizer file: " << args.tokenizer_path << "\n";
        return 1;
    }

    std::vector<InputSource> input_sources;
    if (!resolve_input_sources(args.input_entries, data_glob, data_list, remote_cache_dir.string(), input_sources, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (input_sources.empty())
    {
        std::cerr << "No files matched input settings.\n";
        return 1;
    }

    std::vector<FileTask> tasks;
    std::vector<std::string> input_files;
    tasks.reserve(input_sources.size());
    input_files.reserve(input_sources.size());
    for (std::size_t i = 0; i < input_sources.size(); ++i)
    {
        FileTask t;
        t.index = i;
        t.source = input_sources[i].source;
        t.path = input_sources[i].local_path;
        t.normalized_path = input_sources[i].normalized_id;
        t.file_size = input_sources[i].file_size;
        t.file_mtime = input_sources[i].file_mtime;
        tasks.push_back(std::move(t));
        input_files.push_back(input_sources[i].source);
    }

    PartSignature sig;
    sig.tokenizer_fingerprint = tokenizer_fp;
    sig.text_field = args.text_field;
    sig.min_chars = args.min_chars;
    sig.max_chars = args.max_chars;
    sig.bos_id = bos_id;
    sig.eos_id = eos_id;
    sig.dtype_bytes = dtype_bytes;
    std::string run_signature = detail::make_run_signature(sig);

    std::unordered_map<std::string, CompletionRecord> completion_records;
    if (!detail::init_completion_records(completed_list_path, run_signature, args.resume, completion_records, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    bool resume_existing_shards = args.resume && !completion_records.empty();
    ShardWriter shard_writer(shard_dir, dtype_bytes, args.max_tokens_per_shard);
    if (!shard_writer.initialize(resume_existing_shards, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (resume_existing_shards)
    {
        std::uint64_t completed_tokens = 0;
        for (const auto &kv : completion_records)
        {
            completed_tokens += kv.second.result.num_tokens;
        }
        if (completed_tokens != shard_writer.total_tokens())
        {
            std::cerr << "resume state mismatch: completed.list tokens=" << completed_tokens
                      << ", existing shards tokens=" << shard_writer.total_tokens()
                      << ". Please run with --no-resume or clean output dir.\n";
            return 1;
        }
    }

    std::size_t worker_threads = args.threads;
    if (worker_threads == 0)
    {
        worker_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    }
    bool single_large_file_mode = tasks.size() == 1 && worker_threads > 1;
    std::size_t file_worker_threads = single_large_file_mode ? 1 : worker_threads;
    std::size_t per_file_encode_threads = single_large_file_mode ? worker_threads : 1;

    std::uint64_t total_docs_est = 0;
    if (args.prescan_records)
    {
        err.clear();
        if (!detail::prescan_total_docs(tasks, args, total_docs_est, err))
        {
            std::cerr << err << "\n";
            return 1;
        }
    }

    std::cerr << "Files: " << input_files.size() << "\n";
    std::cerr << "Threads(file-level): " << file_worker_threads << "\n";
    if (per_file_encode_threads > 1)
    {
        std::cerr << "Threads(in-file encode): " << per_file_encode_threads << "\n";
    }
    std::cerr << "Token piece cache entries/worker: " << args.cache_max_entries << "\n";
    if (args.max_memory_mb > 0)
    {
        std::cerr << "Memory cap/worker: " << args.max_memory_mb << " MiB\n";
    }
    std::cerr << "Prescan docs: " << (args.prescan_records ? "on" : "off") << "\n";
    std::cerr << "Append EOS per doc: " << (args.add_eos ? "on" : "off") << "\n";
    if (args.prescan_records && total_docs_est > 0)
    {
        std::cerr << "Total docs (estimated): " << total_docs_est << "\n";
    }
    std::cerr << "Tokenizer model: " << tokenizer.model_name() << "\n";
    std::cerr << "Tokenizer vocab: " << tokenizer.vocab_size() << " (dtype=" << dtype_name << ")\n";
    std::cerr << "Output root: " << out_root.string() << "\n";
    std::cerr << "Shard dir: " << shard_dir.string() << "\n";
    std::cerr << "Completed list: " << completed_list_path.string() << "\n";
    std::cerr << "Resume completed files: " << completion_records.size() << "\n";
    std::string input_mode = !args.input_entries.empty() ? "python-list" : (data_list.empty() ? "glob" : "list");
    std::cerr << "Input mode: " << input_mode << "\n";

    auto start_time = std::chrono::steady_clock::now();
    std::vector<PartResult> results(tasks.size());
    std::atomic<std::size_t> next_idx{0};
    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::mutex completion_mu;
    std::string shared_err;
    ProgressTracker progress(tasks.size(), "tokenizing", 1000, "files");
    if (args.prescan_records && total_docs_est > 0)
    {
        progress.set_total_docs(total_docs_est);
    }

    auto worker = [&]() {
        auto report_docs = [&](std::uint64_t docs) {
            if (docs > 0)
            {
                progress.add(0, docs);
            }
        };
        while (true)
        {
            if (had_error.load(std::memory_order_relaxed))
            {
                break;
            }
            std::size_t idx = next_idx.fetch_add(1);
            if (idx >= tasks.size())
            {
                break;
            }
            PartResult r;
            std::string local_err;
            if (args.resume && detail::completion_is_reusable(tasks[idx], completion_records, r))
            {
                results[idx] = r;
                progress.add(1, r.num_docs);
                continue;
            }
            auto append_tokens = [&](const std::vector<std::uint32_t> &tokens, std::string &write_err) {
                return shard_writer.append_tokens(tokens, write_err);
            };
            if (!detail::process_file_to_shards(tasks[idx], args, tokenizer, sig, append_tokens,
                                                per_file_encode_threads, r, report_docs, local_err))
            {
                had_error.store(true);
                std::lock_guard<std::mutex> lock(err_mu);
                if (shared_err.empty())
                {
                    shared_err = local_err;
                }
                continue;
            }
            if (!detail::append_completion_record(completed_list_path, tasks[idx], r, completion_mu, local_err))
            {
                had_error.store(true);
                std::lock_guard<std::mutex> lock(err_mu);
                if (shared_err.empty())
                {
                    shared_err = local_err;
                }
                continue;
            }
            results[idx] = r;
            progress.add(1, 0);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(file_worker_threads);
    for (std::size_t t = 0; t < file_worker_threads; ++t)
    {
        threads.emplace_back(worker);
    }
    for (auto &t : threads)
    {
        t.join();
    }
    progress.finish();
    if (had_error.load())
    {
        std::cerr << (shared_err.empty() ? "tokenization failed" : shared_err) << "\n";
        return 1;
    }

    std::uint64_t num_docs = 0;
    std::uint64_t num_skipped = 0;
    std::uint64_t expected_tokens = 0;
    std::size_t reused_files = 0;
    for (const auto &r : results)
    {
        num_docs += r.num_docs;
        num_skipped += r.num_skipped;
        expected_tokens += r.num_tokens;
        if (r.reused)
        {
            ++reused_files;
        }
    }

    std::vector<ShardInfo> shards;
    std::uint64_t total_tokens = 0;
    err.clear();
    if (!shard_writer.finalize(shards, total_tokens, err))
    {
        std::cerr << err << "\n";
        return 1;
    }
    if (expected_tokens != total_tokens)
    {
        std::cerr << "token count mismatch: expected " << expected_tokens << " from completion records, got "
                  << total_tokens << " in shards\n";
        return 1;
    }

    std::filesystem::path meta_path = out_root / "meta.json";
    if (!write_meta_json(meta_path, args, data_glob, data_list, input_files, tokenizer.vocab_size(), dtype_name, eos_id,
                         bos_id, num_docs,
                         num_skipped, total_tokens, shards, reused_files, err))
    {
        std::cerr << err << "\n";
        return 1;
    }

    double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time)
            .count();
    if (elapsed < 1e-9)
    {
        elapsed = 1e-9;
    }
    std::cerr << "done. docs=" << num_docs << " skipped=" << num_skipped << " total_tokens=" << total_tokens << "\n";
    std::cerr << "shards=" << shards.size() << " dtype=" << dtype_name << " out=" << shard_dir.string() << "\n";
        std::cerr << "reused_files=" << reused_files << "/" << input_files.size() << "\n";
        std::cerr << "throughput docs/s=" << static_cast<double>(num_docs) / elapsed
                  << " tok/s=" << static_cast<double>(total_tokens) / elapsed << "\n";
    return 0;
}

} // namespace tokenflux::tokenize

