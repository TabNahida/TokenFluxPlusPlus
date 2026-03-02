#include "tokenize_pipeline_internal.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "tokenflux_lib.hpp"

namespace tokenflux::tokenize::detail
{
bool process_file_to_shards(
    const FileTask &task, const Args &args, const TokenizerEncoder &tokenizer, const PartSignature &sig,
    const std::function<bool(const std::vector<std::uint32_t> &, std::string &)> &append_tokens,
    std::size_t encode_threads, PartResult &result, const std::function<void(std::uint64_t)> &report_docs,
    std::string &err)
{
    std::size_t flush_threshold_tokens = 256 * 1024;
    std::size_t effective_cache_max_entries = args.cache_max_entries;
    if (args.max_memory_mb > 0)
    {
        std::uint64_t budget_bytes = static_cast<std::uint64_t>(args.max_memory_mb) * 1024ull * 1024ull;
        std::uint64_t buffer_budget = std::max<std::uint64_t>(budget_bytes / 4ull, 1ull << 20);
        std::size_t per_token_bytes = sizeof(std::uint32_t);
        std::size_t derived_flush = static_cast<std::size_t>(buffer_budget / std::max<std::size_t>(per_token_bytes, 1));
        flush_threshold_tokens = std::max<std::size_t>(64 * 1024, derived_flush);
        if (effective_cache_max_entries > 0)
        {
            std::size_t derived_cache_cap = static_cast<std::size_t>((budget_bytes * 3ull / 4ull) / 128ull);
            if (derived_cache_cap == 0)
            {
                derived_cache_cap = 1;
            }
            effective_cache_max_entries = std::min<std::size_t>(effective_cache_max_entries, derived_cache_cap);
        }
    }

    if (encode_threads <= 1)
    {
        std::vector<std::uint32_t> write_buffer;
        write_buffer.reserve(std::min<std::size_t>(flush_threshold_tokens, 1 << 20));
        std::unordered_map<std::string, std::vector<std::uint32_t>> cache;
        if (effective_cache_max_entries > 0)
        {
            const std::size_t reserve_n = std::min<std::size_t>(effective_cache_max_entries, 1 << 16);
            cache.reserve(reserve_n);
        }

        auto flush_buffer = [&]() -> bool {
            if (write_buffer.empty())
            {
                return true;
            }
            if (!append_tokens(write_buffer, err))
            {
                return false;
            }
            write_buffer.clear();
            return true;
        };

        constexpr std::uint64_t report_docs_batch = 128;
        std::uint64_t docs_pending_report = 0;
        bool callback_ok = true;
        bool read_ok = for_each_text_record(
            task.path, args.text_field,
            [&](const std::string &incoming_text) {
                if (!callback_ok)
                {
                    return;
                }
                if (incoming_text.empty())
                {
                    return;
                }
                std::string text = incoming_text;
                std::size_t chars = utf8_char_count(text);
                if (chars < args.min_chars)
                {
                    ++result.num_skipped;
                    return;
                }
                if (args.max_chars > 0 && chars > args.max_chars)
                {
                    text = truncate_utf8(text, args.max_chars);
                }

                std::size_t before = write_buffer.size();
                if (sig.bos_id >= 0)
                {
                    write_buffer.push_back(static_cast<std::uint32_t>(sig.bos_id));
                }
                tokenizer.encode_text_append(text, cache, write_buffer);
                if (sig.eos_id >= 0)
                {
                    write_buffer.push_back(static_cast<std::uint32_t>(sig.eos_id));
                }
                if (effective_cache_max_entries == 0 || cache.size() > effective_cache_max_entries)
                {
                    cache.clear();
                    cache.rehash(0);
                }

                ++result.num_docs;
                ++docs_pending_report;
                result.num_tokens += static_cast<std::uint64_t>(write_buffer.size() - before);
                if (docs_pending_report >= report_docs_batch && report_docs)
                {
                    report_docs(docs_pending_report);
                    docs_pending_report = 0;
                }
                if (write_buffer.size() >= flush_threshold_tokens && !flush_buffer())
                {
                    callback_ok = false;
                }
            },
            err);
        if (!read_ok)
        {
            if (err.empty())
            {
                err = "failed to read input file: " + task.source;
            }
            return false;
        }
        if (!callback_ok)
        {
            return false;
        }
        if (!flush_buffer())
        {
            return false;
        }
        if (docs_pending_report > 0 && report_docs)
        {
            report_docs(docs_pending_report);
        }

        result.reused = false;
        return true;
    }

    struct EncodeBatchIn
    {
        std::size_t id = 0;
        std::vector<std::string> docs;
    };
    struct EncodeBatchOut
    {
        std::size_t id = 0;
        std::vector<std::uint32_t> tokens;
        std::uint64_t num_docs = 0;
        std::uint64_t num_tokens = 0;
    };

    const std::size_t batch_docs = std::max<std::size_t>(args.encode_batch_size, 1);
    const std::size_t target_batch_bytes = derive_stream_target_bytes(task.file_size, encode_threads);
    const std::size_t in_queue_cap = std::max<std::size_t>(encode_threads * 4, 8);
    constexpr std::uint64_t report_docs_batch = 128;

    std::atomic<bool> had_error{false};
    std::mutex err_mu;
    std::string shared_err;

    std::deque<EncodeBatchIn> in_queue;
    std::mutex in_mu;
    std::condition_variable in_cv;
    bool input_done = false;

    std::unordered_map<std::size_t, EncodeBatchOut> out_ready;
    std::mutex out_mu;
    std::condition_variable out_cv;
    std::size_t workers_done = 0;

    std::uint64_t skipped_docs = 0;
    std::mutex skipped_mu;

    auto set_error = [&](const std::string &message) {
        had_error.store(true, std::memory_order_relaxed);
        if (!message.empty())
        {
            std::lock_guard<std::mutex> lock(err_mu);
            if (shared_err.empty())
            {
                shared_err = message;
            }
        }
        in_cv.notify_all();
        out_cv.notify_all();
    };

    auto producer = [&]() {
        std::size_t next_batch_id = 0;
        EncodeBatchIn pending;
        pending.id = next_batch_id;
        pending.docs.reserve(batch_docs);
        std::size_t pending_bytes = 0;
        std::uint64_t local_skipped = 0;

        auto push_pending = [&]() -> bool {
            if (pending.docs.empty())
            {
                return true;
            }
            std::unique_lock<std::mutex> lock(in_mu);
            in_cv.wait(lock, [&]() { return had_error.load(std::memory_order_relaxed) || in_queue.size() < in_queue_cap; });
            if (had_error.load(std::memory_order_relaxed))
            {
                return false;
            }
            in_queue.push_back(std::move(pending));
            pending = {};
            ++next_batch_id;
            pending.id = next_batch_id;
            pending.docs.reserve(batch_docs);
            pending_bytes = 0;
            lock.unlock();
            in_cv.notify_all();
            return true;
        };

        std::string local_err;
        bool callback_ok = true;
        bool read_ok = for_each_text_record(
            task.path, args.text_field,
            [&](const std::string &incoming_text) {
                if (!callback_ok || had_error.load(std::memory_order_relaxed))
                {
                    return;
                }
                if (incoming_text.empty())
                {
                    return;
                }
                std::string text = incoming_text;
                std::size_t chars = utf8_char_count(text);
                if (chars < args.min_chars)
                {
                    ++local_skipped;
                    return;
                }
                if (args.max_chars > 0 && chars > args.max_chars)
                {
                    text = truncate_utf8(text, args.max_chars);
                }
                pending_bytes += text.size();
                pending.docs.push_back(std::move(text));
                if ((pending.docs.size() >= batch_docs || pending_bytes >= target_batch_bytes) && !push_pending())
                {
                    callback_ok = false;
                }
            },
            local_err);

        if (read_ok && callback_ok && !had_error.load(std::memory_order_relaxed))
        {
            if (!push_pending())
            {
                callback_ok = false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(skipped_mu);
            skipped_docs = local_skipped;
        }

        if (!read_ok)
        {
            set_error(local_err.empty() ? ("failed to read input file: " + task.source) : local_err);
        }
        else if (!callback_ok && !had_error.load(std::memory_order_relaxed))
        {
            set_error("failed to enqueue encode batches");
        }

        {
            std::lock_guard<std::mutex> lock(in_mu);
            input_done = true;
        }
        in_cv.notify_all();
    };

    auto worker = [&]() {
        std::unordered_map<std::string, std::vector<std::uint32_t>> local_cache;
        if (effective_cache_max_entries > 0)
        {
            const std::size_t reserve_n = std::min<std::size_t>(effective_cache_max_entries, 1 << 16);
            local_cache.reserve(reserve_n);
        }

        while (true)
        {
            EncodeBatchIn in_batch;
            {
                std::unique_lock<std::mutex> lock(in_mu);
                in_cv.wait(lock, [&]() {
                    return had_error.load(std::memory_order_relaxed) || !in_queue.empty() || input_done;
                });
                if (had_error.load(std::memory_order_relaxed) && in_queue.empty())
                {
                    break;
                }
                if (in_queue.empty())
                {
                    if (input_done)
                    {
                        break;
                    }
                    continue;
                }
                in_batch = std::move(in_queue.front());
                in_queue.pop_front();
            }
            in_cv.notify_all();

            EncodeBatchOut out_batch;
            out_batch.id = in_batch.id;
            for (const auto &text : in_batch.docs)
            {
                if (sig.bos_id >= 0)
                {
                    out_batch.tokens.push_back(static_cast<std::uint32_t>(sig.bos_id));
                }
                tokenizer.encode_text_append(text, local_cache, out_batch.tokens);
                if (sig.eos_id >= 0)
                {
                    out_batch.tokens.push_back(static_cast<std::uint32_t>(sig.eos_id));
                }
                ++out_batch.num_docs;
            }
            out_batch.num_tokens = static_cast<std::uint64_t>(out_batch.tokens.size());
            if (effective_cache_max_entries == 0 || local_cache.size() > effective_cache_max_entries)
            {
                local_cache.clear();
                local_cache.rehash(0);
            }

            {
                std::lock_guard<std::mutex> lock(out_mu);
                out_ready.emplace(out_batch.id, std::move(out_batch));
            }
            out_cv.notify_all();
        }

        {
            std::lock_guard<std::mutex> lock(out_mu);
            ++workers_done;
        }
        out_cv.notify_all();
    };

    std::thread producer_thread(producer);

    std::vector<std::thread> encode_workers;
    encode_workers.reserve(encode_threads);
    for (std::size_t i = 0; i < encode_threads; ++i)
    {
        encode_workers.emplace_back(worker);
    }

    std::size_t next_write_batch = 0;
    std::uint64_t docs_pending_report = 0;
    while (true)
    {
        EncodeBatchOut out_batch;
        bool has_batch = false;
        {
            std::unique_lock<std::mutex> lock(out_mu);
            out_cv.wait(lock, [&]() {
                return had_error.load(std::memory_order_relaxed) || out_ready.find(next_write_batch) != out_ready.end() ||
                       workers_done == encode_threads;
            });

            auto it = out_ready.find(next_write_batch);
            if (it != out_ready.end())
            {
                out_batch = std::move(it->second);
                out_ready.erase(it);
                has_batch = true;
            }
            else if (workers_done == encode_threads)
            {
                break;
            }
        }

        if (!has_batch)
        {
            if (had_error.load(std::memory_order_relaxed))
            {
                break;
            }
            continue;
        }

        if (!append_tokens(out_batch.tokens, err))
        {
            set_error(err);
            break;
        }
        result.num_docs += out_batch.num_docs;
        result.num_tokens += out_batch.num_tokens;
        docs_pending_report += out_batch.num_docs;
        if (docs_pending_report >= report_docs_batch && report_docs)
        {
            report_docs(docs_pending_report);
            docs_pending_report = 0;
        }
        ++next_write_batch;
    }

    producer_thread.join();
    for (auto &t : encode_workers)
    {
        t.join();
    }

    if (docs_pending_report > 0 && report_docs)
    {
        report_docs(docs_pending_report);
    }
    {
        std::lock_guard<std::mutex> lock(skipped_mu);
        result.num_skipped = skipped_docs;
    }
    if (had_error.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(err_mu);
        if (!shared_err.empty())
        {
            err = shared_err;
        }
        else if (err.empty())
        {
            err = "failed to process file: " + task.source;
        }
        return false;
    }

    result.reused = false;
    return true;
}

} // namespace tokenflux::tokenize::detail

