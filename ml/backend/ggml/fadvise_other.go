//go:build !linux

package ggml

func fadviseSequential(fd uintptr, offset, length int64) {}
