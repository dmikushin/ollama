package ggml

import "golang.org/x/sys/unix"

// fadviseSequential hints the kernel that the file region will be read sequentially,
// enabling read-ahead which significantly improves NFS throughput.
func fadviseSequential(fd uintptr, offset, length int64) {
	_ = unix.Fadvise(int(fd), offset, length, unix.FADV_SEQUENTIAL)
}
