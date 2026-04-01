#pragma once

// NFS/network filesystem optimizations for model loading:
// - Prefetch thread for mmap path (MADV_WILLNEED ahead of access)
// - O_DIRECT reader to bypass page cache for large one-shot reads
// - io_uring async reader for pipelined NFS reads

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>
#include <atomic>

#if defined(__linux__)
#define LLAMA_NFS_OPT_LINUX 1
#endif

// ─── Prefetch thread for mmap path ───────────────────────────────────────────
// Walks ahead through tensor regions calling madvise(MADV_WILLNEED) so the
// kernel starts fetching NFS pages before the main thread needs them.

struct llama_mmap_prefetcher {
    llama_mmap_prefetcher();
    ~llama_mmap_prefetcher();

    // Start prefetching tensor regions. regions = {offset, length} pairs,
    // already sorted by offset. addr = mmap base address.
    void start(void * addr, const std::vector<std::pair<size_t, size_t>> & regions);

    // Signal that the main thread has finished processing tensor at index i.
    // The prefetcher will stay ahead_count tensors ahead.
    void advance(size_t i);

    // Stop and join the thread.
    void stop();

    static const size_t ahead_count = 16; // prefetch this many tensors ahead

private:
    struct impl;
    impl * pimpl;
};

// ─── O_DIRECT file reader ────────────────────────────────────────────────────
// Opens a file with O_DIRECT to bypass page cache. Reads must use aligned
// buffers. For 250GB+ models that don't fit in page cache, this avoids
// cache thrashing and reduces memory pressure.

struct llama_direct_reader {
    llama_direct_reader(const char * path);
    ~llama_direct_reader();

    // Read len bytes at file offset into dst. dst must be aligned to 4096.
    // offset should ideally be aligned but will be handled if not.
    void pread_aligned(void * dst, size_t len, size_t offset);

    // Alignment requirement for O_DIRECT.
    static size_t alignment();

    // Whether O_DIRECT is supported on this file.
    bool is_direct() const;

private:
    struct impl;
    impl * pimpl;
};

// ─── io_uring async reader ───────────────────────────────────────────────────
// Submits batched read requests via io_uring, allowing NFS to pipeline
// multiple RPC requests instead of waiting for each one sequentially.

struct llama_uring_reader {
    llama_uring_reader(int fd, size_t queue_depth = 32);
    ~llama_uring_reader();

    // Submit an async read. Returns a token for wait_one().
    uint64_t submit_read(void * buf, size_t len, size_t offset);

    // Wait for the next completion. Returns {token, bytes_read}.
    // Returns {0, -1} on error.
    std::pair<uint64_t, ssize_t> wait_one();

    // Number of in-flight requests.
    size_t pending() const;

    static bool supported();

private:
    struct impl;
    impl * pimpl;
};
