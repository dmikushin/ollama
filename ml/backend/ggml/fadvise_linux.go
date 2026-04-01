package ggml

import (
	"log/slog"

	"golang.org/x/sys/unix"
)

// fadviseSequential hints the kernel that the file region will be read sequentially,
// enabling read-ahead which significantly improves NFS throughput.
func fadviseSequential(fd uintptr, offset, length int64) {
	if err := unix.Fadvise(int(fd), offset, length, unix.FADV_SEQUENTIAL); err != nil {
		slog.Debug("posix_fadvise(FADV_SEQUENTIAL) failed", "fd", fd, "error", err)
	}
}
