WHISPER_DIR := $(abspath ./whisper.cpp)
NPROC       := $(shell nproc)

# ── CGO paths ────────────────────────────────────────────────────────────────
export C_INCLUDE_PATH := $(WHISPER_DIR)/include:$(WHISPER_DIR)/ggml/include
export LIBRARY_PATH   := $(WHISPER_DIR)/build/src:$(WHISPER_DIR)/build/ggml/src
export CGO_LDFLAGS    := -Wl,-rpath,$(WHISPER_DIR)/build/src \
                         -Wl,-rpath,$(WHISPER_DIR)/build/ggml/src

# ── CUDA toolkit path (WSL2: install cuda-toolkit-12-x, NOT the full driver) ─
# Prerequisites:
#   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
#   sudo dpkg -i cuda-keyring_1.1-1_all.deb
#   sudo apt-get update
#   sudo apt-get install -y cuda-toolkit-12-8
#   export PATH=/usr/local/cuda/bin:$PATH
CUDA_DIR     := /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

.PHONY: all build bulk whisper whisper-clean clean test

all: build

# Run cmake configure + incremental build (fast on subsequent runs)
whisper:
	cmake -B $(WHISPER_DIR)/build -S $(WHISPER_DIR) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DGGML_CUDA=ON \
	    -DGGML_CUDA_FA=ON \
	    -DBUILD_SHARED_LIBS=ON \
	    -DCMAKE_CUDA_ARCHITECTURES=89
	cmake --build $(WHISPER_DIR)/build --config Release -j$(NPROC)

build:
	@echo "Building streamscribe (CUDA)..."
	CGO_LDFLAGS="$(CGO_LDFLAGS) -Wl,-rpath,$(CUDA_LIB_DIR)" \
	LIBRARY_PATH="$(LIBRARY_PATH):$(CUDA_LIB_DIR)" \
	go build -o streamscribe ./cmd/example

bulk:
	@echo "Running bulk-transcriber (CUDA)..."
	CGO_LDFLAGS="$(CGO_LDFLAGS) -Wl,-rpath,$(CUDA_LIB_DIR)" \
	LIBRARY_PATH="$(LIBRARY_PATH):$(CUDA_LIB_DIR)" \
	go run ./cmd/bulk-transcriber

# ── Utilities ────────────────────────────────────────────────────────────────

# Full rebuild of whisper.cpp from scratch (only needed after upstream changes)
whisper-clean:
	rm -rf $(WHISPER_DIR)/build
	$(MAKE) whisper

clean:
	rm -f streamscribe

TEST_FLAGS ?=

test:
	@echo "Running tests..."
	CGO_LDFLAGS="$(CGO_LDFLAGS) -Wl,-rpath,$(CUDA_LIB_DIR)" \
	LIBRARY_PATH="$(LIBRARY_PATH):$(CUDA_LIB_DIR)" \
	go test $(TEST_FLAGS) ./...
