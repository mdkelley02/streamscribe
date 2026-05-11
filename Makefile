WHISPER_DIR := $(abspath ./whisper.cpp)
NPROC       := $(shell nproc)
PREFIX      ?= /usr/local

# ── CUDA toolkit path (WSL2: install cuda-toolkit-12-x, NOT the full driver) ─
# Prerequisites:
#   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
#   sudo dpkg -i cuda-keyring_1.1-1_all.deb
#   sudo apt-get update
#   sudo apt-get install -y cuda-toolkit-12-8
#   export PATH=/usr/local/cuda/bin:$PATH

.PHONY: all build install-whisper uninstall-whisper whisper-clean clean test

all: build

# Build whisper.cpp with CUDA and install headers + shared libs into $(PREFIX)
# so any CGO consumer (this repo, downstream apps, IDEs, hot-reloaders) finds
# them via the default search paths — no per-project env wiring required.
#
# Run once per machine. Rerun after bumping whisper.cpp or changing the GPU
# arch (CMAKE_CUDA_ARCHITECTURES below). Build artifacts stay owned by the
# invoking user; only the install step elevates with sudo.
install-whisper:
	cmake -B $(WHISPER_DIR)/build -S $(WHISPER_DIR) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DGGML_CUDA=ON \
	    -DGGML_CUDA_FA=ON \
	    -DBUILD_SHARED_LIBS=ON \
	    -DCMAKE_CUDA_ARCHITECTURES=89
	cmake --build $(WHISPER_DIR)/build --config Release -j$(NPROC)
	sudo cmake --install $(WHISPER_DIR)/build --prefix=$(PREFIX)
	sudo ldconfig

# Remove the files install-whisper placed under $(PREFIX).
uninstall-whisper:
	@test -f $(WHISPER_DIR)/build/install_manifest.txt \
	    || { echo "no install_manifest.txt at $(WHISPER_DIR)/build; nothing to uninstall"; exit 0; }
	sudo xargs rm -f < $(WHISPER_DIR)/build/install_manifest.txt
	sudo ldconfig

# Wipe the cmake build directory so the next install-whisper reconfigures
# from scratch (needed after upstream whisper.cpp changes).
whisper-clean:
	rm -rf $(WHISPER_DIR)/build

build:
	go build -o streamscribe ./cmd/example

clean:
	rm -f streamscribe

TEST_FLAGS ?=

test:
	go test $(TEST_FLAGS) ./...
