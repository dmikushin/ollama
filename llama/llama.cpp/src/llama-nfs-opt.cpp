#include "llama-nfs-opt.h"
#include "llama-impl.h"

#include <cstring>
#include <cerrno>
#include <stdexcept>

#if LLAMA_NFS_OPT_LINUX
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Prefetch thread (mmap path)
// ═══════════════════════════════════════════════════════════════════════════════

#if LLAMA_NFS_OPT_LINUX

struct llama_mmap_prefetcher::impl {
    std::thread thread;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<size_t> main_idx{0};
    std::atomic<bool> should_stop{false};

    void * addr = nullptr;
    const std::vector<std::pair<size_t, size_t>> * regions = nullptr;

    void run() {
        LLAMA_LOG_DEBUG("prefetch thread started, %zu regions, ahead_count=%zu\n",
                        regions->size(), ahead_count);
        size_t prefetch_idx = 0;
        size_t total_bytes_prefetched = 0;
        while (!should_stop.load(std::memory_order_relaxed)) {
            size_t target = main_idx.load(std::memory_order_relaxed) + ahead_count;
            while (prefetch_idx < target && prefetch_idx < regions->size()) {
                auto [off, len] = (*regions)[prefetch_idx];
                int rc = posix_madvise(static_cast<char *>(addr) + off, len, POSIX_MADV_WILLNEED);
                if (rc != 0) {
                    LLAMA_LOG_WARN("prefetch madvise(WILLNEED) failed at offset %zu, len %zu: %s\n",
                                   off, len, strerror(rc));
                }
                total_bytes_prefetched += len;
                prefetch_idx++;
            }
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                return should_stop.load(std::memory_order_relaxed) ||
                       main_idx.load(std::memory_order_relaxed) + ahead_count > prefetch_idx;
            });
        }
        LLAMA_LOG_DEBUG("prefetch thread finished, prefetched %zu regions (%.1f MB)\n",
                        prefetch_idx, (double)total_bytes_prefetched / (1024.0 * 1024.0));
    }
};

llama_mmap_prefetcher::llama_mmap_prefetcher() : pimpl(new impl()) {}

llama_mmap_prefetcher::~llama_mmap_prefetcher() {
    stop();
    delete pimpl;
}

void llama_mmap_prefetcher::start(void * addr, const std::vector<std::pair<size_t, size_t>> & regions) {
    pimpl->addr = addr;
    pimpl->regions = &regions;
    pimpl->should_stop.store(false);
    pimpl->main_idx.store(0);
    pimpl->thread = std::thread(&impl::run, pimpl);
}

void llama_mmap_prefetcher::advance(size_t i) {
    pimpl->main_idx.store(i, std::memory_order_relaxed);
    pimpl->cv.notify_one();
}

void llama_mmap_prefetcher::stop() {
    if (pimpl && pimpl->thread.joinable()) {
        pimpl->should_stop.store(true, std::memory_order_relaxed);
        pimpl->cv.notify_one();
        pimpl->thread.join();
    }
}

#else

struct llama_mmap_prefetcher::impl {};
llama_mmap_prefetcher::llama_mmap_prefetcher() : pimpl(nullptr) {}
llama_mmap_prefetcher::~llama_mmap_prefetcher() {}
void llama_mmap_prefetcher::start(void *, const std::vector<std::pair<size_t, size_t>> &) {}
void llama_mmap_prefetcher::advance(size_t) {}
void llama_mmap_prefetcher::stop() {}

#endif

// ═══════════════════════════════════════════════════════════════════════════════
// O_DIRECT reader
// ═══════════════════════════════════════════════════════════════════════════════

#if LLAMA_NFS_OPT_LINUX

struct llama_direct_reader::impl {
    int fd = -1;
    bool direct = false;

    impl(const char * path) {
        fd = open(path, O_RDONLY | O_DIRECT);
        if (fd >= 0) {
            direct = true;
            LLAMA_LOG_INFO("O_DIRECT enabled for model file (bypassing page cache)\n");
        } else {
            int saved_errno = errno;
            fd = open(path, O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error(format("failed to open %s: %s", path, strerror(errno)));
            }
            direct = false;
            LLAMA_LOG_WARN("O_DIRECT not available (%s), using buffered I/O with fadvise\n",
                           strerror(saved_errno));
        }
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL) != 0) {
            LLAMA_LOG_DEBUG("posix_fadvise(SEQUENTIAL) failed: %s\n", strerror(errno));
        }
    }

    ~impl() {
        if (fd >= 0) {
            close(fd);
        }
    }

    void pread_aligned(void * dst, size_t len, size_t offset) {
        if (!direct) {
            pread_fallback(dst, len, offset);
            return;
        }

        const size_t align = 4096;
        size_t aligned_off = offset & ~(align - 1);
        size_t prefix = offset - aligned_off;
        size_t aligned_len = (prefix + len + align - 1) & ~(align - 1);

        // Fast path: offset and dst aligned, len aligned
        if (prefix == 0 && (len & (align - 1)) == 0 &&
            (reinterpret_cast<uintptr_t>(dst) & (align - 1)) == 0) {
            pread_full(dst, len, offset);
            return;
        }

        // Slow path: use aligned temp buffer
        void * tmp = nullptr;
        if (posix_memalign(&tmp, align, aligned_len) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        pread_full(tmp, aligned_len, aligned_off);
        memcpy(dst, static_cast<char *>(tmp) + prefix, len);
        free(tmp);
    }

    void pread_fallback(void * dst, size_t len, size_t offset) {
        size_t total = 0;
        while (total < len) {
            ssize_t n = ::pread(fd, static_cast<char *>(dst) + total, len - total, offset + total);
            if (n > 0) { total += n; continue; }
            if (n < 0 && errno == EINTR) { continue; }
            throw std::runtime_error(format("pread error at offset %zu: %s", offset + total, strerror(errno)));
        }
    }

    void pread_full(void * dst, size_t len, size_t offset) {
        size_t total = 0;
        while (total < len) {
            ssize_t n = ::pread(fd, static_cast<char *>(dst) + total, len - total, offset + total);
            if (n > 0) { total += n; continue; }
            if (n < 0 && errno == EINTR) { continue; }
            throw std::runtime_error(format("direct pread error: %s", strerror(errno)));
        }
    }
};

llama_direct_reader::llama_direct_reader(const char * path) : pimpl(new impl(path)) {}
llama_direct_reader::~llama_direct_reader() { delete pimpl; }
void llama_direct_reader::pread_aligned(void * dst, size_t len, size_t offset) { pimpl->pread_aligned(dst, len, offset); }
size_t llama_direct_reader::alignment() { return 4096; }
bool llama_direct_reader::is_direct() const { return pimpl->direct; }

#else

struct llama_direct_reader::impl {};
llama_direct_reader::llama_direct_reader(const char *) : pimpl(nullptr) {
    throw std::runtime_error("O_DIRECT not supported on this platform");
}
llama_direct_reader::~llama_direct_reader() {}
void llama_direct_reader::pread_aligned(void *, size_t, size_t) {}
size_t llama_direct_reader::alignment() { return 1; }
bool llama_direct_reader::is_direct() const { return false; }

#endif

// ═══════════════════════════════════════════════════════════════════════════════
// io_uring async reader (raw syscalls, no liburing)
// ═══════════════════════════════════════════════════════════════════════════════

#if LLAMA_NFS_OPT_LINUX

static int llama_io_uring_setup(unsigned entries, struct io_uring_params * p) {
    return (int) syscall(__NR_io_uring_setup, entries, p);
}

static int llama_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return (int) syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, NULL, 0);
}

struct llama_uring_reader::impl {
    int ring_fd = -1;

    // Mapped ring memory
    void * sq_ring_ptr = MAP_FAILED;
    size_t sq_ring_sz = 0;
    void * cq_ring_ptr = MAP_FAILED;
    size_t cq_ring_sz = 0;
    struct io_uring_sqe * sqes = nullptr;
    size_t sqes_sz = 0;

    // Kernel-provided offsets (from io_uring_params)
    struct io_sqring_offsets sq_off;
    struct io_cqring_offsets cq_off;

    int file_fd;

    std::atomic<size_t> n_pending{0};
    uint64_t next_token = 1;

    impl(int fd, size_t queue_depth) : file_fd(fd) {
        struct io_uring_params params;
        memset(&params, 0, sizeof(params));

        ring_fd = llama_io_uring_setup(queue_depth, &params);
        if (ring_fd < 0) {
            LLAMA_LOG_WARN("io_uring_setup failed: %s (falling back to sync I/O)\n", strerror(errno));
            return;
        }

        sq_off = params.sq_off;
        cq_off = params.cq_off;

        // Map SQ ring
        sq_ring_sz = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
        sq_ring_ptr = mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
        if (sq_ring_ptr == MAP_FAILED) { goto fail; }

        // Map SQEs
        sqes_sz = params.sq_entries * sizeof(struct io_uring_sqe);
        sqes = (struct io_uring_sqe *) mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE,
                                            MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
        if (sqes == MAP_FAILED) { sqes = nullptr; goto fail; }

        // Map CQ ring
        cq_ring_sz = params.cq_off.cqes + params.cq_entries * sizeof(struct io_uring_cqe);
        cq_ring_ptr = mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
        if (cq_ring_ptr == MAP_FAILED) { goto fail; }

        LLAMA_LOG_INFO("io_uring initialized with %u SQ entries, %u CQ entries\n",
                       params.sq_entries, params.cq_entries);
        return;

    fail:
        LLAMA_LOG_WARN("io_uring mmap failed: %s\n", strerror(errno));
        cleanup();
        ring_fd = -1;
    }

    void cleanup() {
        if (sq_ring_ptr != MAP_FAILED) { munmap(sq_ring_ptr, sq_ring_sz); sq_ring_ptr = MAP_FAILED; }
        if (sqes != nullptr)           { munmap(sqes, sqes_sz); sqes = nullptr; }
        if (cq_ring_ptr != MAP_FAILED) { munmap(cq_ring_ptr, cq_ring_sz); cq_ring_ptr = MAP_FAILED; }
        if (ring_fd >= 0)              { close(ring_fd); ring_fd = -1; }
    }

    ~impl() { cleanup(); }

    bool available() const { return ring_fd >= 0; }

    // Accessors into mapped ring memory via kernel-provided offsets
    uint32_t * sq_head_ptr()  { return (uint32_t *)((char *)sq_ring_ptr + sq_off.head); }
    uint32_t * sq_tail_ptr()  { return (uint32_t *)((char *)sq_ring_ptr + sq_off.tail); }
    uint32_t   sq_mask_val()  { return *(uint32_t *)((char *)sq_ring_ptr + sq_off.ring_mask); }
    uint32_t * sq_array_ptr() { return (uint32_t *)((char *)sq_ring_ptr + sq_off.array); }

    uint32_t * cq_head_ptr()  { return (uint32_t *)((char *)cq_ring_ptr + cq_off.head); }
    uint32_t * cq_tail_ptr()  { return (uint32_t *)((char *)cq_ring_ptr + cq_off.tail); }
    uint32_t   cq_mask_val()  { return *(uint32_t *)((char *)cq_ring_ptr + cq_off.ring_mask); }
    struct io_uring_cqe * cq_cqes() {
        return (struct io_uring_cqe *)((char *)cq_ring_ptr + cq_off.cqes);
    }

    uint64_t submit_read(void * buf, size_t len, size_t offset) {
        if (!available()) { return 0; }

        uint32_t tail = *sq_tail_ptr();
        uint32_t mask = sq_mask_val();
        uint32_t idx = tail & mask;

        struct io_uring_sqe * sqe = &sqes[idx];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = file_fd;
        sqe->addr = (uint64_t)(uintptr_t) buf;
        sqe->len = (__u32) len;
        sqe->off = (__u64) offset;

        uint64_t token = next_token++;
        sqe->user_data = token;

        sq_array_ptr()[idx] = idx;
        __atomic_store_n(sq_tail_ptr(), tail + 1, __ATOMIC_RELEASE);

        int ret = llama_io_uring_enter(ring_fd, 1, 0, 0);
        if (ret < 0) {
            LLAMA_LOG_WARN("io_uring_enter submit failed: %s\n", strerror(errno));
            return 0;
        }

        n_pending.fetch_add(1, std::memory_order_relaxed);
        return token;
    }

    std::pair<uint64_t, ssize_t> wait_one() {
        if (!available() || n_pending.load(std::memory_order_relaxed) == 0) {
            return {0, -1};
        }

        int ret = llama_io_uring_enter(ring_fd, 0, 1, IORING_ENTER_GETEVENTS);
        if (ret < 0) {
            LLAMA_LOG_WARN("io_uring_enter wait failed: %s\n", strerror(errno));
            return {0, -1};
        }

        uint32_t head = *cq_head_ptr();
        uint32_t tail_val = __atomic_load_n(cq_tail_ptr(), __ATOMIC_ACQUIRE);
        if (head == tail_val) { return {0, -1}; }

        uint32_t mask = cq_mask_val();
        struct io_uring_cqe * cqe = &cq_cqes()[head & mask];

        uint64_t token = cqe->user_data;
        ssize_t result = cqe->res;

        if (result < 0) {
            LLAMA_LOG_WARN("io_uring read completion error: token=%llu, error=%s\n",
                           (unsigned long long)token, strerror(-(int)result));
        }

        __atomic_store_n(cq_head_ptr(), head + 1, __ATOMIC_RELEASE);
        n_pending.fetch_sub(1, std::memory_order_relaxed);

        return {token, result};
    }
};

llama_uring_reader::llama_uring_reader(int fd, size_t queue_depth) : pimpl(new impl(fd, queue_depth)) {}
llama_uring_reader::~llama_uring_reader() { delete pimpl; }
uint64_t llama_uring_reader::submit_read(void * buf, size_t len, size_t offset) { return pimpl->submit_read(buf, len, offset); }
std::pair<uint64_t, ssize_t> llama_uring_reader::wait_one() { return pimpl->wait_one(); }
size_t llama_uring_reader::pending() const { return pimpl->n_pending.load(std::memory_order_relaxed); }

bool llama_uring_reader::supported() {
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    int fd = llama_io_uring_setup(1, &params);
    if (fd >= 0) { close(fd); return true; }
    return false;
}

#else

struct llama_uring_reader::impl {};
llama_uring_reader::llama_uring_reader(int, size_t) : pimpl(nullptr) {}
llama_uring_reader::~llama_uring_reader() {}
uint64_t llama_uring_reader::submit_read(void *, size_t, size_t) { return 0; }
std::pair<uint64_t, ssize_t> llama_uring_reader::wait_one() { return {0, -1}; }
size_t llama_uring_reader::pending() const { return 0; }
bool llama_uring_reader::supported() { return false; }

#endif
