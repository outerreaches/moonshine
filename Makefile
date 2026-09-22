CC ?= cc
AR ?= ar
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_HOME ?= /opt/rocm
ROCM_ARCH ?= gfx1151
PYTHON ?= python3

CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -march=native \
	-Wall -Wextra -std=c99 -D_GNU_SOURCE
HIPFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -march=native \
	-pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument \
	-I$(ROCM_HOME)/include --offload-arch=$(ROCM_ARCH)
LDLIBS ?= -lm -pthread -lzstd
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt -lzstd
ICU_LDLIBS ?= $(shell pkg-config --libs icu-i18n 2>/dev/null || \
	echo -licui18n -licuuc -licudata)

MOONSHINE_MODEL ?=
MOONSHINE_CONTEXT ?= 8192
MOONSHINE_PREFILL_TOKENS ?= 512
MOONSHINE_CROSSOVER_TOKENS ?= 2 3 4 6 8 12 16 24 32 42
MOONSHINE_RETRIEVAL_TARGET ?= 16000
MOONSHINE_RETRIEVAL_BACKEND ?=
MOONSHINE_STATE_DIR ?= /tmp
MOONSHINE_DECODE_TRACE ?=
MOONSHINE_DECODE_LEDGER_TRACE ?=
MOONSHINE_DECODE_CACHE_TRACE ?=
MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY ?=
MOONSHINE_DECODE_CACHE_FRESH_EMPTY_SOURCE ?= 0
MOONSHINE_DECODE_CACHE_CAPACITIES ?= 24 26 28 30 31 32 34 36 40
MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN ?=
GLM53_OFFICIAL_ROOT ?=

GLM53_OBJS := \
	glm53_arch_math.o \
	glm53_architecture.o \
	glm53_dense_ops.o \
	glm53_engine_plan.o \
	glm53_expert_plan.o \
	glm53_expert_stream.o \
	glm53_fp8_dynamic.o \
	glm53_fp8_oracle.o \
	glm53_kda_aux_ops.o \
	glm53_kda_gate_ops.o \
	glm53_kda_ops.o \
	glm53_manifest.o \
	glm53_mhc_ops.o \
	glm53_official_tensor.o \
	glm53_phase5c.o \
	glm53_process_memory.o \
	glm53_residency.o \
	glm53_rocm_ops.o \
	glm53_state_oracle.o \
	glm53_vector_ops.o \
	glm53_static_bindings.o \
	glm53_static_layout.o \
	glm53_static_loader.o \
	glm53_weights.o

MIMO26_CPU_TESTS := \
	tests/test_mimo26_architecture \
	tests/test_mimo26_manifest \
	tests/test_mimo26_fp8 \
	tests/test_mimo26_router \
	tests/test_mimo26_ops \
	tests/test_mimo26_attention \
	tests/test_mimo26_kv \
	tests/test_mimo26_expert_cache

GLM53_CPU_TESTS := \
	tests/test_glm53_arch_math \
	tests/test_glm53_architecture \
	tests/test_glm53_engine_plan \
	tests/test_glm53_expert_plan \
	tests/test_glm53_expert_stream \
	tests/test_glm53_fp8_dynamic \
	tests/test_glm53_fp8_oracle \
	tests/test_glm53_manifest \
	tests/test_glm53_official_tensor \
	tests/test_glm53_process_memory \
	tests/test_glm53_residency \
	tests/test_glm53_state_oracle \
	tests/test_glm53_static_bindings \
	tests/test_glm53_static_layout \
	tests/test_glm53_weights

GLM53_ROCM_TESTS := \
	tests/test_glm53_dense_ops \
	tests/test_glm53_fp8_dynamic_rocm \
	tests/test_glm53_kda_aux_ops \
	tests/test_glm53_kda_gate_ops \
	tests/test_glm53_kda_ops \
	tests/test_glm53_mhc_ops \
	tests/test_glm53_phase5c \
	tests/test_glm53_rocm_ops \
	tests/test_glm53_static_loader \
	tests/test_glm53_vector_ops

GLM53_OFFICIAL_TESTS := \
	tests/test_glm53_architecture_official \
	tests/test_glm53_official_components \
	tests/test_glm53_official_kda \
	tests/test_glm53_official_projection \
	tests/test_glm53_phase5a_official \
	tests/test_glm53_phase5b_expert_official \
	tests/test_glm53_phase5b_loader_official \
	tests/test_glm53_phase5c_mhc_official

K3_OBJS := \
	k3_chat.o \
	k3_bundle.o \
	k3_engine.o \
	k3_prefill.o \
	k3_static_store.o \
	k3_static_layout.o \
	k3_expert_cache.o \
	k3_io_uring.o \
	k3_mzg.o \
	k3_mzg2.o \
	k3_json.o \
	k3_openai.o \
	k3_prefix_reuse.o \
	k3_prefix_catalog.o \
	k3_prefix_bundle.o \
	k3_prefill_route_index.o \
	k3_rocm_ops.o \
	k3_safetensors.o \
	k3_tokenizer.o

PORTABLE_CPU_TESTS := \
	tests/test_k3_expert_cache \
	tests/test_k3_bundle \
	tests/test_k3_prefix_reuse \
	tests/test_k3_prefix_bundle \
	tests/test_k3_prefix_catalog \
	tests/test_k3_prefill_route_index \
	tests/test_k3_prefill_timeline \
	tests/test_k3_q8_codec \
	tests/test_k3_json \
	tests/test_k3_openai \
	tests/test_k3_server_slot \
	tests/test_k3_safetensors_file

MODEL_CPU_TESTS := \
	tests/test_k3_prefill_plan \
	tests/test_k3_safetensors \
	tests/test_k3_mzg_store \
	tests/test_k3_io_qd

CPU_TESTS := $(PORTABLE_CPU_TESTS) $(MODEL_CPU_TESTS)

ASSET_TESTS := \
	tests/test_k3_tokenizer

ROCM_TESTS := \
	tests/test_k3_cache_registration \
	tests/test_k3_dense_mlp \
	tests/test_k3_embedding_output \
	tests/test_k3_engine_hello \
	tests/test_k3_engine_init \
	tests/test_k3_expert_smoke \
	tests/test_k3_kda_layer_smoke \
	tests/test_k3_kda_recurrent \
	tests/test_k3_mla_batch_determinism \
	tests/test_k3_mla_batch_kernels \
	tests/test_k3_mla_decode \
	tests/test_k3_mla_layer_smoke \
	tests/test_k3_moe_tail_profile \
	tests/test_k3_moe_smoke \
	tests/test_k3_mxfp4_envelope \
	tests/test_k3_prefill_512 \
	tests/test_k3_prefill_chunk \
	tests/test_k3_prefill_crossover \
	tests/test_k3_prefill_gemm_shapes \
	tests/test_k3_prefill_ops \
	tests/test_k3_q8_projection \
	tests/test_k3_residual_spine \
	tests/test_k3_rocm_components \
	tests/test_k3_state_checkpoint \
	tests/test_k3_static_store

CHAT_TESTS := \
	tests/test_k3_chat_session \
	tests/test_k3_prefix_checkpoint \
	tests/test_k3_long_context

ALL_TESTS := $(CPU_TESTS) $(ASSET_TESTS) $(ROCM_TESTS) $(CHAT_TESTS) \
	$(MIMO26_CPU_TESTS) \
	$(GLM53_CPU_TESTS) $(GLM53_ROCM_TESTS) $(GLM53_OFFICIAL_TESTS)

.PHONY: all help tests test test-cpu check-model \
	test-mimo26-schema test-mimo26-checkpoint mimo26-budget \
	mimo26-layer-parity \
	test-model-layout test-model-components test-engine-init \
	test-engine-hello test-chat-hello test-state-checkpoint test-tokenizer \
	test-prefill-2 test-prefill-scale test-prefill-kda-blas \
	test-prefill-crossover test-prefill-gemm-shapes \
	test-long-context-retrieval \
	test-prefix-checkpoint \
	test-mla-batch-determinism test-mla-batch-kernels \
	test-moe-tail-profile test-decode-cache-replay test-cache-analyzer \
	test-prefill-screen-analyzer test-anchor-recovery-analyzer \
	test-static-q8-screen test-mzg-transcoder test-mzg2-bundle \
	test-reduction-qualification test-openai-sdk \
	test-glm53-phase2 test-glm53-phase3 test-glm53-phase4 \
	test-glm53-phase4-official test-glm53-phase5a \
	test-glm53-phase5a-official test-glm53-phase5b \
	test-glm53-phase5b-official test-glm53-phase5b-full \
	test-glm53-phase5c test-glm53-phase5c-official \
	test-glm53-phase5d-kda clean

all: libmoonshine.a moonshine-chat moonshine-server

help:
	@echo "Moonshine targets:"
	@echo "  make                         Build library, chat client, and API server"
	@echo "  make moonshine-chat          Build the interactive deterministic client"
	@echo "  make moonshine-server        Build the OpenAI-compatible one-slot server"
	@echo "  make tests                   Build every test binary"
	@echo "  make test-cpu                Run portable tests without ROCm or weights"
	@echo "  make test                    Run model-free CPU/ROCm tests"
	@echo "  make test-openai-sdk         Run the optional official Python SDK SSE fixture"
	@echo "  make test-mzg-transcoder     Run MZG format/transcoder tests (python-zstandard)"
	@echo "  make test-static-q8-screen  Run model-free Q8 codec tests and build the offline screen"
	@echo "  make test-model-layout MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Validate the pinned 96-shard layout and plan"
	@echo "  make test-model-components MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Run bounded real-weight component oracles"
	@echo "  make test-engine-init MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_CONTEXT=8192"
	@echo "                               Allocate and validate the selected context"
	@echo "  make test-engine-hello MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_CONTEXT=8192"
	@echo "                               Run the locked greedy hello at the selected context"
	@echo "  make test-chat-hello MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Run native tokenizer-to-text chat end to end"
	@echo "  make test-state-checkpoint MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Prove checksummed export/import continuation"
	@echo "  make test-tokenizer MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Check native tokenizer and XTML parity"
	@echo "  make test-prefill-2 MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Compare two-token range and sequential state"
	@echo "  make test-prefill-scale MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_PREFILL_TOKENS=512"
	@echo "                               Run the default layer-major scale fixture"
	@echo "  make test-prefill-kda-blas MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_PREFILL_TOKENS=8192"
	@echo "                               Run the diagnostic KDA hipBLAS candidate"
	@echo "  make test-prefill-crossover MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_CROSSOVER_TOKENS='2 4 8 16'"
	@echo "                               Compare exact warm sequential/range suffixes"
	@echo "  make test-prefill-gemm-shapes"
	@echo "                               Screen exact MXFP4 expert tiles at production scale"
	@echo "  make test-long-context-retrieval MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_RETRIEVAL_TARGET=512|16000|32000"
	@echo "                               Run a deterministic natural-text gate"
	@echo "  make test-mla-batch-determinism"
	@echo "                               Compare looped and strided-batched MLA GEMMs"
	@echo "  make test-mla-batch-kernels  Check exact batched MLA launch primitives"
	@echo "  make test-moe-tail-profile  Profile model-shape MoE-tail kernels"
	@echo "  make test-decode-cache-replay MOONSHINE_DECODE_CACHE_TRACE=/path/to/cache.csv \\"
	@echo "       MOONSHINE_DECODE_LEDGER_TRACE=/path/to/ledger.csv \\"
	@echo "       MOONSHINE_DECODE_TRACE=/path/to/routes.csv MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY=32"
	@echo "                               Validate one capture and replay LRU capacities"
	@echo "       Set MOONSHINE_DECODE_CACHE_FRESH_EMPTY_SOURCE=1 only for a proven fresh empty seed"
	@echo "  make test-cache-analyzer"
	@echo "                               Run portable offline cache-analysis tests"
	@echo "       MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN=/path/to/accepted-run adds offline goldens"
	@echo "  make test-reduction-qualification MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Run the MXFP4 reduction-change gate bundle"
	@echo "  make clean                  Remove local build products"

libmoonshine.a: $(K3_OBJS) $(GLM53_OBJS)
	$(AR) rcs $@ $^

tests: $(ALL_TESTS)

%.o: %.c
	$(CC) $(CFLAGS) -I. -c -o $@ $<

%.o: %.cu
	$(HIPCC) $(HIPFLAGS) -I. -c -o $@ $<

k3_engine.o: k3_engine.cu k3_engine_state.inc k3_engine_prefill.inc k3_engine.h \
	k3_bundle.h k3_prefill.h k3_static_store.h k3_expert_cache.h \
	k3_io_uring.h k3_mzg.h k3_mzg2.h k3_rocm_ops.h k3_safetensors.h
k3_prefix_bundle.o: k3_prefix_bundle.c k3_prefix_bundle.h \
	k3_engine.h k3_tokenizer.h
k3_chat.o: k3_chat.c k3_chat.h k3_engine.h k3_tokenizer.h \
	k3_prefix_reuse.h k3_prefix_bundle.h
k3_prefix_catalog.o: k3_prefix_catalog.c k3_prefix_catalog.h \
	k3_engine.h k3_prefix_reuse.h
k3_prefill_route_index.o: k3_prefill_route_index.c \
	k3_prefill_route_index.h
k3_bundle.o: k3_bundle.c k3_bundle.h k3_json.h
k3_prefix_reuse.o: k3_prefix_reuse.c k3_prefix_reuse.h
k3_chat_cli.o: k3_chat_cli.c k3_chat.h k3_engine.h moonshine_version.h
k3_server.o: k3_server.c k3_server_slot.h k3_chat.h k3_json.h k3_openai.h \
	moonshine_version.h
k3_server_slot.o: k3_server_slot.c k3_server_slot.h
k3_prefill.o: k3_prefill.c k3_prefill.h k3_safetensors.h
k3_static_store.o: k3_static_store.cu k3_static_store.h \
	k3_rocm_ops.h k3_safetensors.h
k3_static_layout.o: k3_static_layout.c k3_static_store.h k3_safetensors.h
k3_q8_codec.o: k3_q8_codec.c k3_q8_codec.h
k3_q8_codec.o: CFLAGS += -fno-fast-math -frounding-math
k3_expert_cache.o: k3_expert_cache.c k3_expert_cache.h
k3_io_uring.o: k3_io_uring.c k3_io_uring.h
k3_mzg.o: k3_mzg.c k3_mzg.h
k3_mzg2.o: k3_mzg2.cu k3_mzg2.h
k3_json.o: k3_json.c k3_json.h
k3_openai.o: k3_openai.c k3_openai.h k3_json.h k3_chat.h
k3_rocm_ops.o: k3_rocm_ops.cu k3_rocm_ops.h
k3_safetensors.o: k3_safetensors.c k3_safetensors.h
k3_tokenizer.o: k3_tokenizer.c k3_tokenizer.h
glm53_arch_math.o: glm53_arch_math.c glm53_arch_math.h
glm53_arch_math.o: CFLAGS += -fno-fast-math
glm53_architecture.o: glm53_architecture.c glm53_architecture.h \
	k3_safetensors.h
glm53_engine_plan.o: glm53_engine_plan.c glm53_engine_plan.h
glm53_expert_stream.o: glm53_expert_stream.c glm53_expert_stream.h \
	glm53_expert_plan.h k3_safetensors.h
glm53_expert_plan.o: glm53_expert_plan.c glm53_expert_plan.h k3_safetensors.h
glm53_fp8_dynamic.o: glm53_fp8_dynamic.c glm53_fp8_dynamic.h
glm53_fp8_dynamic.o: CFLAGS += -fno-fast-math
glm53_fp8_oracle.o: glm53_fp8_oracle.c glm53_fp8_oracle.h
glm53_fp8_oracle.o: CFLAGS += -fno-fast-math
tests/test_mimo26_fp8.o: CFLAGS += -fno-fast-math
glm53_manifest.o: glm53_manifest.c glm53_manifest.h k3_json.h \
	k3_safetensors.h
glm53_official_tensor.o: glm53_official_tensor.c glm53_official_tensor.h \
	k3_safetensors.h
glm53_process_memory.o: glm53_process_memory.c glm53_process_memory.h
glm53_residency.o: glm53_residency.c glm53_residency.h
glm53_rocm_ops.o: glm53_rocm_ops.cu glm53_rocm_ops.h
glm53_rocm_ops.o tests/test_glm53_fp8_dynamic_rocm.o \
	tests/test_glm53_phase5b_expert_official.o \
	tests/test_glm53_rocm_ops.o tests/test_glm53_official_projection.o: \
	HIPFLAGS += -fno-fast-math
glm53_state_oracle.o: glm53_state_oracle.c glm53_state_oracle.h
glm53_state_oracle.o: CFLAGS += -fno-fast-math
glm53_static_layout.o: glm53_static_layout.c glm53_static_layout.h \
	glm53_weights.h
glm53_static_loader.o: glm53_static_loader.cu glm53_static_loader.h \
	glm53_static_layout.h k3_safetensors.h
glm53_weights.o: glm53_weights.c glm53_weights.h glm53_architecture.h \
	glm53_expert_plan.h glm53_manifest.h k3_safetensors.h

tests/test_glm53_arch_math.o: tests/test_glm53_arch_math.c glm53_arch_math.h
tests/test_glm53_architecture.o: tests/test_glm53_architecture.c \
	glm53_architecture.h
tests/test_glm53_engine_plan.o: tests/test_glm53_engine_plan.c \
	glm53_engine_plan.h
tests/test_glm53_expert_stream.o: tests/test_glm53_expert_stream.c \
	glm53_expert_stream.h
tests/test_glm53_architecture_official.o: \
	tests/test_glm53_architecture_official.c glm53_architecture.h \
	glm53_manifest.h k3_safetensors.h
tests/test_glm53_expert_plan.o: tests/test_glm53_expert_plan.c \
	glm53_expert_plan.h
tests/test_glm53_fp8_dynamic.o: tests/test_glm53_fp8_dynamic.c \
	glm53_fp8_dynamic.h
tests/test_glm53_fp8_dynamic.o: CFLAGS += -fno-fast-math
tests/test_glm53_fp8_oracle.o: tests/test_glm53_fp8_oracle.c \
	glm53_fp8_oracle.h
tests/test_glm53_manifest.o: tests/test_glm53_manifest.c glm53_manifest.h
tests/test_glm53_official_components.o: \
	tests/test_glm53_official_components.c glm53_arch_math.h
tests/test_glm53_official_kda.o: tests/test_glm53_official_kda.c \
	glm53_state_oracle.h
tests/test_glm53_official_projection.o: \
	tests/test_glm53_official_projection.cu glm53_fp8_oracle.h \
	glm53_manifest.h glm53_official_tensor.h glm53_rocm_ops.h k3_safetensors.h
tests/test_glm53_official_tensor.o: tests/test_glm53_official_tensor.c \
	glm53_official_tensor.h
tests/test_glm53_state_oracle.o: tests/test_glm53_state_oracle.c \
	glm53_state_oracle.h
tests/test_glm53_weights.o: tests/test_glm53_weights.c glm53_weights.h \
	glm53_architecture.h glm53_engine_plan.h
tests/test_glm53_process_memory.o: tests/test_glm53_process_memory.c \
	glm53_process_memory.h
tests/test_glm53_static_layout.o: tests/test_glm53_static_layout.c \
	glm53_static_layout.h glm53_weights.h
tests/test_glm53_static_loader.o: tests/test_glm53_static_loader.cu \
	glm53_static_loader.h glm53_static_layout.h
tests/test_glm53_static_bindings.o: tests/test_glm53_static_bindings.c \
	glm53_static_bindings.h glm53_static_loader.h glm53_static_layout.h \
	glm53_weights.h
tests/test_glm53_phase5b_loader_official.o: \
	tests/test_glm53_phase5b_loader_official.cu glm53_static_loader.h \
	glm53_static_layout.h glm53_process_memory.h glm53_weights.h
tests/test_glm53_phase5c_mhc_official.o: \
	tests/test_glm53_phase5c_mhc_official.cu glm53_mhc_ops.h \
	glm53_vector_ops.h glm53_process_memory.h glm53_weights.h
tests/test_glm53_phase5a_official.o: \
	tests/test_glm53_phase5a_official.c glm53_weights.h \
	glm53_expert_stream.h glm53_engine_plan.h
tests/test_glm53_residency.o: tests/test_glm53_residency.c glm53_residency.h
tests/test_glm53_rocm_ops.o: tests/test_glm53_rocm_ops.cu \
	glm53_rocm_ops.h glm53_fp8_oracle.h k3_rocm_ops.h
tests/test_glm53_vector_ops.o: tests/test_glm53_vector_ops.cu \
	glm53_vector_ops.h
tests/test_glm53_kda_aux_ops.o: tests/test_glm53_kda_aux_ops.cu \
	glm53_kda_aux_ops.h
tests/test_glm53_kda_gate_ops.o: tests/test_glm53_kda_gate_ops.cu \
	glm53_kda_gate_ops.h
tests/test_glm53_kda_ops.o: tests/test_glm53_kda_ops.cu glm53_kda_ops.h
glm53_kda_aux_ops.o: glm53_kda_aux_ops.cu glm53_kda_aux_ops.h
glm53_kda_gate_ops.o: glm53_kda_gate_ops.cu glm53_kda_gate_ops.h
glm53_kda_ops.o: glm53_kda_ops.cu glm53_kda_ops.h

glm53_kda_aux_ops.o glm53_kda_gate_ops.o glm53_kda_ops.o \
	tests/test_glm53_kda_aux_ops.o tests/test_glm53_kda_gate_ops.o \
	tests/test_glm53_kda_ops.o: HIPFLAGS += -fno-fast-math
glm53_mhc_ops.o: glm53_mhc_ops.cu glm53_mhc_ops.h
tests/test_glm53_dense_ops.o: tests/test_glm53_dense_ops.cu \
	glm53_dense_ops.h glm53_rocm_ops.h glm53_vector_ops.h
tests/test_glm53_phase5c.o: tests/test_glm53_phase5c.cu glm53_phase5c.h \
	glm53_static_bindings.h

tests/test_k3_bundle.o: tests/test_k3_bundle.c k3_bundle.h
tests/test_k3_chat_session.o: tests/test_k3_chat_session.c k3_chat.h
tests/test_k3_long_context.o: tests/test_k3_long_context.c k3_chat.h
tests/test_k3_openai.o: tests/test_k3_openai.c k3_openai.h k3_chat.h
tests/test_k3_prefix_checkpoint.o: tests/test_k3_prefix_checkpoint.c k3_chat.h
tests/test_k3_server_slot.o: tests/test_k3_server_slot.c k3_server_slot.h
tests/test_k3_safetensors_file.o: tests/test_k3_safetensors_file.c \
	k3_safetensors.h
tests/test_k3_mzg_store.o: tests/test_k3_mzg_store.c k3_mzg.h k3_safetensors.h
tests/test_k3_prefix_reuse.o: tests/test_k3_prefix_reuse.c k3_prefix_reuse.h
tests/test_k3_prefix_catalog.o: tests/test_k3_prefix_catalog.c \
	k3_prefix_catalog.h k3_engine.h
tests/test_k3_prefix_bundle.o: tests/test_k3_prefix_bundle.c \
	k3_prefix_bundle.h
tests/test_k3_state_checkpoint.o: tests/test_k3_state_checkpoint.cu k3_engine.h
tests/test_k3_prefill_route_index.o: tests/test_k3_prefill_route_index.c \
	k3_prefill_route_index.h
tests/test_k3_prefill_timeline.o: tests/test_k3_prefill_timeline.c \
	k3_prefill.h
tests/test_k3_tokenizer.o: tests/test_k3_tokenizer.c k3_tokenizer.h
tools/transcode_mzg2_layer.o: tools/transcode_mzg2_layer.cu \
	k3_mzg2.h k3_safetensors.h
tools/screen_static_q8.o: tools/screen_static_q8.c \
	k3_q8_codec.h k3_static_store.h k3_safetensors.h

tools/transcode_mzg2_layer: tools/transcode_mzg2_layer.o k3_safetensors.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tools/screen_static_q8: tools/screen_static_q8.o k3_q8_codec.o \
		k3_static_layout.o k3_safetensors.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_glm53_arch_math: tests/test_glm53_arch_math.o glm53_arch_math.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_architecture.o: mimo26_architecture.c mimo26_architecture.h \
	k3_safetensors.h
tests/test_mimo26_architecture.o: tests/test_mimo26_architecture.c \
	mimo26_architecture.h k3_safetensors.h
tests/test_mimo26_official.o: tests/test_mimo26_official.c \
	mimo26_manifest.h mimo26_architecture.h k3_safetensors.h
tests/test_mimo26_official: tests/test_mimo26_official.o \
		mimo26_manifest.o mimo26_architecture.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tools/mimo26_dump_ops.o: tools/mimo26_dump_ops.c \
	mimo26_attention.h mimo26_ops.h mimo26_router.h
tools/mimo26_dump_ops.o: CFLAGS += -fno-fast-math
tools/mimo26_dump_ops: tools/mimo26_dump_ops.o mimo26_ops.o \
		mimo26_router.o mimo26_attention.o mimo26_architecture.o \
		k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tools/mimo26_dump_rope.o: tools/mimo26_dump_rope.c \
	mimo26_attention.h mimo26_ops.h
tools/mimo26_dump_rope.o: CFLAGS += -fno-fast-math
tools/mimo26_dump_rope: tools/mimo26_dump_rope.o mimo26_attention.o \
		mimo26_ops.o mimo26_architecture.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_mimo26_expert_cache.o: tests/test_mimo26_expert_cache.c \
	k3_expert_cache.h mimo26_architecture.h mimo26_manifest.h \
	mimo26_router.h
tests/test_mimo26_expert_cache: tests/test_mimo26_expert_cache.o \
		k3_expert_cache.o mimo26_manifest.o mimo26_architecture.o \
		mimo26_router.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tools/mimo26_dump_weights.o: tools/mimo26_dump_weights.c \
	mimo26_weights.h mimo26_manifest.h mimo26_architecture.h \
	mimo26_router.h
tools/mimo26_dump_weights: tools/mimo26_dump_weights.o \
		mimo26_weights.o mimo26_manifest.o mimo26_architecture.o \
		mimo26_attention.o mimo26_ops.o mimo26_router.o \
		glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tools/mimo26_run.o: tools/mimo26_run.c mimo26_worker.h
tools/mimo26_run: tools/mimo26_run.o mimo26_worker.o mimo26_layer.o \
		mimo26_weights.o mimo26_kv.o mimo26_manifest.o \
		mimo26_architecture.o mimo26_attention.o mimo26_ops.o \
		mimo26_router.o k3_expert_cache.o glm53_fp8_oracle.o \
		k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_worker.o: mimo26_worker.c mimo26_worker.h mimo26_layer.h \
	mimo26_weights.h mimo26_manifest.h mimo26_ops.h k3_expert_cache.h
mimo26_worker.o: CFLAGS += -fno-fast-math
mimo26_layer.o: mimo26_layer.c mimo26_layer.h mimo26_weights.h \
	mimo26_kv.h mimo26_router.h mimo26_ops.h mimo26_architecture.h
mimo26_layer.o: CFLAGS += -fno-fast-math
tests/test_mimo26_layer_parity.o: tests/test_mimo26_layer_parity.c \
	mimo26_layer.h mimo26_weights.h mimo26_manifest.h mimo26_ops.h \
	mimo26_architecture.h
tests/test_mimo26_layer_parity.o: CFLAGS += -fno-fast-math
tests/test_mimo26_layer_parity: tests/test_mimo26_layer_parity.o \
		mimo26_layer.o mimo26_weights.o mimo26_kv.o mimo26_manifest.o \
		mimo26_architecture.o mimo26_attention.o mimo26_ops.o \
		mimo26_router.o glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_server_slot.o: mimo26_server_slot.c mimo26_server_slot.h
tests/test_mimo26_server_slot.o: tests/test_mimo26_server_slot.c \
	mimo26_server_slot.h
tests/test_mimo26_server_slot: tests/test_mimo26_server_slot.o \
		mimo26_server_slot.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_mimo26_qualify.o: tests/test_mimo26_qualify.c mimo26_worker.h
tests/test_mimo26_qualify.o: CFLAGS += -fno-fast-math
tests/test_mimo26_qualify: tests/test_mimo26_qualify.o \
		mimo26_worker.o mimo26_layer.o mimo26_weights.o mimo26_kv.o \
		mimo26_manifest.o mimo26_architecture.o mimo26_attention.o \
		mimo26_ops.o mimo26_router.o k3_expert_cache.o \
		glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_weights.o: mimo26_weights.c mimo26_weights.h \
	mimo26_architecture.h mimo26_attention.h mimo26_ops.h \
	mimo26_router.h glm53_fp8_oracle.h k3_safetensors.h
mimo26_weights.o: CFLAGS += -fno-fast-math
mimo26_kv.o: mimo26_kv.c mimo26_kv.h mimo26_attention.h \
	mimo26_architecture.h
tests/test_mimo26_kv.o: tests/test_mimo26_kv.c mimo26_kv.h \
	mimo26_architecture.h mimo26_ops.h
tests/test_mimo26_kv: tests/test_mimo26_kv.o mimo26_kv.o \
		mimo26_attention.o mimo26_ops.o mimo26_architecture.o \
		k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_attention.o: mimo26_attention.c mimo26_attention.h \
	mimo26_architecture.h mimo26_ops.h
mimo26_attention.o: CFLAGS += -fno-fast-math
tests/test_mimo26_attention.o: tests/test_mimo26_attention.c \
	mimo26_attention.h mimo26_ops.h
tests/test_mimo26_attention.o: CFLAGS += -fno-fast-math
tests/test_mimo26_attention: tests/test_mimo26_attention.o \
		mimo26_attention.o mimo26_ops.o mimo26_architecture.o \
		k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_ops.o: mimo26_ops.c mimo26_ops.h
mimo26_ops.o: CFLAGS += -fno-fast-math
tests/test_mimo26_ops.o: tests/test_mimo26_ops.c mimo26_ops.h \
	glm53_arch_math.h
tests/test_mimo26_ops.o: CFLAGS += -fno-fast-math
tests/test_mimo26_ops: tests/test_mimo26_ops.o mimo26_ops.o \
		glm53_arch_math.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_router.o: mimo26_router.c mimo26_router.h
mimo26_router.o: CFLAGS += -fno-fast-math
tests/test_mimo26_router.o: tests/test_mimo26_router.c \
	mimo26_router.h glm53_arch_math.h
tests/test_mimo26_router.o: CFLAGS += -fno-fast-math
tests/test_mimo26_router: tests/test_mimo26_router.o mimo26_router.o \
		glm53_arch_math.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_mimo26_fp8.o: tests/test_mimo26_fp8.c \
	glm53_fp8_oracle.h mimo26_architecture.h
tests/test_mimo26_fp8: tests/test_mimo26_fp8.o glm53_fp8_oracle.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
mimo26_manifest.o: mimo26_manifest.c mimo26_manifest.h \
	mimo26_architecture.h k3_safetensors.h k3_json.h
tests/test_mimo26_manifest.o: tests/test_mimo26_manifest.c \
	mimo26_manifest.h mimo26_architecture.h k3_safetensors.h
tests/test_mimo26_manifest: tests/test_mimo26_manifest.o \
		mimo26_manifest.o mimo26_architecture.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_mimo26_architecture: tests/test_mimo26_architecture.o \
		mimo26_architecture.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_architecture: tests/test_glm53_architecture.o \
		glm53_architecture.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_architecture_official: \
		tests/test_glm53_architecture_official.o glm53_architecture.o \
		glm53_manifest.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_engine_plan: tests/test_glm53_engine_plan.o \
		glm53_engine_plan.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_expert_plan: tests/test_glm53_expert_plan.o \
		glm53_expert_plan.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_expert_stream: tests/test_glm53_expert_stream.o \
		glm53_expert_stream.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_fp8_dynamic: tests/test_glm53_fp8_dynamic.o \
		glm53_fp8_dynamic.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_fp8_oracle: tests/test_glm53_fp8_oracle.o \
		glm53_fp8_oracle.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_manifest: tests/test_glm53_manifest.o glm53_manifest.o \
		k3_json.o k3_safetensors.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_official_tensor: tests/test_glm53_official_tensor.o \
		glm53_official_tensor.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_process_memory: tests/test_glm53_process_memory.o \
		glm53_process_memory.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_residency: tests/test_glm53_residency.o glm53_residency.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_state_oracle: tests/test_glm53_state_oracle.o \
		glm53_state_oracle.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_static_bindings: tests/test_glm53_static_bindings.o \
		glm53_static_bindings.o glm53_static_loader.o glm53_static_layout.o \
		glm53_weights.o glm53_architecture.o glm53_expert_plan.o \
		glm53_manifest.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_vector_ops: tests/test_glm53_vector_ops.o glm53_vector_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_kda_aux_ops: tests/test_glm53_kda_aux_ops.o glm53_kda_aux_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_kda_gate_ops: tests/test_glm53_kda_gate_ops.o glm53_kda_gate_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_kda_ops: tests/test_glm53_kda_ops.o glm53_kda_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_mhc_ops: tests/test_glm53_mhc_ops.o glm53_mhc_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_dense_ops: tests/test_glm53_dense_ops.o glm53_dense_ops.o \
		glm53_rocm_ops.o glm53_vector_ops.o glm53_fp8_dynamic.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_phase5c: tests/test_glm53_phase5c.o glm53_phase5c.o \
		glm53_dense_ops.o glm53_mhc_ops.o glm53_rocm_ops.o \
		glm53_vector_ops.o glm53_fp8_dynamic.o k3_rocm_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_static_layout: tests/test_glm53_static_layout.o \
		glm53_static_layout.o glm53_weights.o glm53_architecture.o \
		glm53_expert_plan.o glm53_manifest.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_weights: tests/test_glm53_weights.o glm53_weights.o \
		glm53_architecture.o glm53_expert_plan.o glm53_manifest.o \
		k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_phase5a_official: \
		tests/test_glm53_phase5a_official.o glm53_weights.o \
		glm53_expert_stream.o glm53_engine_plan.o glm53_architecture.o \
		glm53_expert_plan.o glm53_manifest.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_fp8_dynamic_rocm: \
		tests/test_glm53_fp8_dynamic_rocm.o glm53_rocm_ops.o \
		glm53_fp8_dynamic.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_rocm_ops: tests/test_glm53_rocm_ops.o glm53_rocm_ops.o \
		glm53_fp8_oracle.o k3_rocm_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_static_loader: tests/test_glm53_static_loader.o \
		glm53_static_loader.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_phase5b_expert_official: \
		tests/test_glm53_phase5b_expert_official.o glm53_rocm_ops.o \
		glm53_fp8_dynamic.o glm53_weights.o glm53_expert_stream.o \
		glm53_architecture.o glm53_expert_plan.o glm53_manifest.o \
		k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_phase5b_loader_official: \
		tests/test_glm53_phase5b_loader_official.o glm53_static_loader.o \
		glm53_static_bindings.o glm53_static_layout.o glm53_phase5c.o \
		glm53_dense_ops.o glm53_mhc_ops.o glm53_rocm_ops.o \
		glm53_vector_ops.o glm53_fp8_dynamic.o glm53_process_memory.o \
		glm53_weights.o glm53_architecture.o glm53_expert_plan.o \
		glm53_manifest.o k3_rocm_ops.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_phase5c_mhc_official: \
		tests/test_glm53_phase5c_mhc_official.o glm53_mhc_ops.o \
		glm53_vector_ops.o glm53_process_memory.o glm53_weights.o \
		glm53_architecture.o glm53_expert_plan.o glm53_manifest.o \
		k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)
tests/test_glm53_official_components: \
		tests/test_glm53_official_components.o glm53_arch_math.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_official_kda: tests/test_glm53_official_kda.o \
		glm53_state_oracle.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_glm53_official_projection: \
		tests/test_glm53_official_projection.o glm53_rocm_ops.o \
		glm53_fp8_oracle.o glm53_official_tensor.o glm53_manifest.o \
		k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/test_k3_bundle: tests/test_k3_bundle.o k3_bundle.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_k3_expert_cache: tests/test_k3_expert_cache.o k3_expert_cache.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_k3_json: tests/test_k3_json.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_k3_prefix_reuse: tests/test_k3_prefix_reuse.o k3_prefix_reuse.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_k3_prefix_bundle: tests/test_k3_prefix_bundle.o \
		k3_prefix_bundle.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_k3_prefix_catalog: tests/test_k3_prefix_catalog.o \
		k3_prefix_catalog.o k3_prefix_reuse.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_k3_server_slot: tests/test_k3_server_slot.o k3_server_slot.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_k3_safetensors_file: tests/test_k3_safetensors_file.o \
		k3_safetensors.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_k3_prefill_route_index: tests/test_k3_prefill_route_index.o \
		k3_prefill_route_index.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_k3_prefill_timeline: tests/test_k3_prefill_timeline.o k3_prefill.o \
		k3_safetensors.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
tests/test_k3_q8_codec: tests/test_k3_q8_codec.o k3_q8_codec.o \
		k3_static_layout.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)


tests/test_k3_openai: tests/test_k3_openai.o k3_openai.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(MODEL_CPU_TESTS): %: %.o libmoonshine.a
	$(CC) $(CFLAGS) -o $@ $< libmoonshine.a $(LDLIBS)

tests/test_k3_tokenizer: tests/test_k3_tokenizer.o libmoonshine.a
	$(CC) $(CFLAGS) -o $@ $< libmoonshine.a $(LDLIBS) $(ICU_LDLIBS)

$(CHAT_TESTS): %: %.o libmoonshine.a
	$(HIPCC) $(HIPFLAGS) -o $@ $< libmoonshine.a \
		$(ROCM_LDLIBS) $(ICU_LDLIBS)

moonshine-chat: k3_chat_cli.o libmoonshine.a
	$(HIPCC) $(HIPFLAGS) -o $@ k3_chat_cli.o libmoonshine.a \
		$(ROCM_LDLIBS) $(ICU_LDLIBS)

moonshine-server: k3_server.o k3_server_slot.o libmoonshine.a
	$(HIPCC) $(HIPFLAGS) -o $@ k3_server.o k3_server_slot.o libmoonshine.a \
		$(ROCM_LDLIBS) $(ICU_LDLIBS)

$(ROCM_TESTS): %: %.o libmoonshine.a
	$(HIPCC) $(HIPFLAGS) -o $@ $< libmoonshine.a $(ROCM_LDLIBS)

test-glm53-phase2: $(GLM53_CPU_TESTS)
	./tests/test_glm53_manifest
	./tests/test_glm53_expert_plan
	./tests/test_glm53_residency

test-glm53-phase3: tests/test_glm53_fp8_oracle $(GLM53_ROCM_TESTS)
	./tests/test_glm53_fp8_oracle
	./tests/test_glm53_rocm_ops

test-glm53-phase4: tests/test_glm53_arch_math \
		tests/test_glm53_architecture tests/test_glm53_official_tensor \
		tests/test_glm53_state_oracle
	./tests/test_glm53_arch_math
	./tests/test_glm53_architecture
	./tests/test_glm53_official_tensor
	./tests/test_glm53_state_oracle

test-glm53-phase5a: tests/test_glm53_engine_plan \
		tests/test_glm53_expert_stream tests/test_glm53_weights
	./tests/test_glm53_engine_plan
	./tests/test_glm53_expert_stream
	./tests/test_glm53_weights

test-glm53-phase5a-official: tests/test_glm53_phase5a_official
	@test -n "$(GLM53_OFFICIAL_ROOT)" || \
		(echo "set GLM53_OFFICIAL_ROOT to the verified official artifact" >&2; exit 2)
	./tests/test_glm53_phase5a_official "$(GLM53_OFFICIAL_ROOT)"

test-glm53-phase5d-kda: tests/test_glm53_kda_aux_ops \
		tests/test_glm53_kda_gate_ops tests/test_glm53_kda_ops
	./tests/test_glm53_kda_aux_ops
	./tests/test_glm53_kda_gate_ops
	./tests/test_glm53_kda_ops

test-glm53-phase5c: tests/test_glm53_static_bindings \
		tests/test_glm53_vector_ops tests/test_glm53_mhc_ops \
		tests/test_glm53_dense_ops tests/test_glm53_phase5c
	./tests/test_glm53_static_bindings
	./tests/test_glm53_vector_ops
	./tests/test_glm53_mhc_ops
	./tests/test_glm53_dense_ops
	./tests/test_glm53_phase5c

test-glm53-phase5c-official: tests/test_glm53_phase5c_mhc_official
	@test -n "$(GLM53_OFFICIAL_ROOT)" || \
		(echo "set GLM53_OFFICIAL_ROOT to the verified official artifact" >&2; exit 2)
	./tests/test_glm53_phase5c_mhc_official "$(GLM53_OFFICIAL_ROOT)"

test-glm53-phase5b: tests/test_glm53_fp8_dynamic \
		tests/test_glm53_fp8_dynamic_rocm tests/test_glm53_process_memory \
		tests/test_glm53_static_layout tests/test_glm53_static_loader
	./tests/test_glm53_fp8_dynamic
	./tests/test_glm53_fp8_dynamic_rocm
	./tests/test_glm53_process_memory
	./tests/test_glm53_static_layout
	./tests/test_glm53_static_loader

test-glm53-phase5b-official: tests/test_glm53_phase5b_expert_official \
		tests/test_glm53_phase5b_loader_official
	@test -n "$(GLM53_OFFICIAL_ROOT)" || \
		(echo "set GLM53_OFFICIAL_ROOT to the verified official artifact" >&2; exit 2)
	./tests/test_glm53_phase5b_expert_official "$(GLM53_OFFICIAL_ROOT)"
	./tests/test_glm53_phase5b_loader_official "$(GLM53_OFFICIAL_ROOT)"

test-glm53-phase5b-full: tests/test_glm53_phase5b_loader_official
	@test -n "$(GLM53_OFFICIAL_ROOT)" || \
		(echo "set GLM53_OFFICIAL_ROOT to the verified official artifact" >&2; exit 2)
	./tests/test_glm53_phase5b_loader_official "$(GLM53_OFFICIAL_ROOT)" full

test-glm53-phase4-official: tests/test_glm53_architecture_official \
		tests/test_glm53_official_components tests/test_glm53_official_kda \
		tests/test_glm53_official_projection \
		tests/verify_glm53_phase4_reference.py \
		tests/fixtures/glm53_phase4_reference.json \
		tests/fixtures/glm53_official_projection_f32.bin \
		tests/fixtures/glm53_phase4_kda_v1.json \
		tests/fixtures/glm53_phase4_kda_v1.bin \
		tests/fixtures/glm53_phase4_components.json \
		tests/fixtures/glm53_phase4_components.bin
	@test -n "$(GLM53_OFFICIAL_ROOT)" || \
		(echo "set GLM53_OFFICIAL_ROOT to the verified official artifact" >&2; exit 2)
	$(PYTHON) tests/verify_glm53_phase4_reference.py \
		"$(GLM53_OFFICIAL_ROOT)" \
		tests/fixtures/glm53_phase4_reference.json \
		tests/fixtures/glm53_official_projection_f32.bin \
		tests/fixtures/glm53_phase4_kda_v1.json \
		tests/fixtures/glm53_phase4_kda_v1.bin \
		tests/fixtures/glm53_phase4_components.json \
		tests/fixtures/glm53_phase4_components.bin
	./tests/test_glm53_architecture_official "$(GLM53_OFFICIAL_ROOT)"
	./tests/test_glm53_official_components \
		tests/fixtures/glm53_phase4_components.bin
	./tests/test_glm53_official_kda \
		tests/fixtures/glm53_phase4_kda_v1.bin
	./tests/test_glm53_official_projection "$(GLM53_OFFICIAL_ROOT)" \
		tests/fixtures/glm53_official_projection_f32.bin

test-cache-analyzer:
	PYTHONDONTWRITEBYTECODE=1 \
		MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN="$(MOONSHINE_CACHE_ANALYZER_GOLDEN_RUN)" \
		$(PYTHON) -m unittest discover -s tests \
		-p 'test_analyze_decode_cache.py' -v

test-prefill-screen-analyzer:
	PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) -m unittest -v tests/test_analyze_prefill_screen.py


test-anchor-recovery-analyzer:
	PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) -m unittest -v tests/test_analyze_anchor_recovery.py

test-static-q8-screen: tests/test_k3_q8_codec tools/screen_static_q8
	./tests/test_k3_q8_codec
	./tools/screen_static_q8 --self-test

test-mzg-transcoder:
	PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) -m unittest -v tests/test_transcode_mzg.py

test-mimo26-schema: tools/mimo26_dump_rope tools/mimo26_dump_ops tests/test_mimo26_architecture \
		tests/test_mimo26_manifest tests/test_mimo26_server_slot
	./tests/test_mimo26_architecture
	./tests/test_mimo26_manifest
	./tests/test_mimo26_server_slot
	./tests/test_mimo26_fp8
	./tests/test_mimo26_router
	./tests/test_mimo26_ops
	./tests/test_mimo26_attention
	./tests/test_mimo26_kv
	./tests/test_mimo26_expert_cache
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/test_mimo26_rope_vs_torch.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/test_mimo26_ops_vs_reference.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/test_mimo26_audit.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/test_mimo26_tokenizer.py

# Full-metadata audit against a real checkpoint. Header-only and read-only;
# requires MIMO26_ROOT to point at an official download.
test-mimo26-checkpoint: tests/test_mimo26_official \
		tools/mimo26_dump_weights
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/audit_mimo26_checkpoint.py \
		$(MIMO26_ROOT)
	./tests/test_mimo26_official $(MIMO26_ROOT)
	MIMO26_ROOT=$(MIMO26_ROOT) PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) tests/test_mimo26_tokenizer.py
	MIMO26_ROOT=$(MIMO26_ROOT) PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) tests/test_mimo26_weights_vs_reference.py

# Layer parity against the reference. Needs a checkpoint and a fixture
# from tools/mimo26_reference_layer.py, so it is separate from the
# checkpoint schema target.
mimo26-layer-parity: tests/test_mimo26_layer_parity
	@for layer in 0 1 5; do \
	  PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/mimo26_reference_layer.py \
	    --root $(MIMO26_ROOT) --layer $$layer --tokens 1 \
	    --out mimo26-layer$$layer.bin >/dev/null 2>&1 || exit 1; \
	  ./tests/test_mimo26_layer_parity $(MIMO26_ROOT) \
	    mimo26-layer$$layer.bin || exit 1; \
	  rm -f mimo26-layer$$layer.bin; \
	done

# M4 operational qualification: determinism, cache-independence, rollback,
# reset, fault containment and the memory guard, against a real checkpoint.
# About a minute per decoded token on CPU, so it is not part of test-mimo26.
mimo26-qualify: tests/test_mimo26_qualify
	MIMO26_ROOT=$(MIMO26_ROOT) ./tests/test_mimo26_qualify

tools/mimo26_eval.o: tools/mimo26_eval.c mimo26_worker.h
tools/mimo26_eval.o: CFLAGS += -fno-fast-math
tools/mimo26_eval: tools/mimo26_eval.o \
		mimo26_worker.o mimo26_layer.o mimo26_weights.o mimo26_kv.o \
		mimo26_manifest.o mimo26_architecture.o mimo26_attention.o \
		mimo26_ops.o mimo26_router.o k3_expert_cache.o \
		glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_mimo26_gpu_mxfp4.o: tests/test_mimo26_gpu_mxfp4.cu \
	k3_rocm_ops.h k3_safetensors.h mimo26_weights.h mimo26_ops.h \
	mimo26_manifest.h mimo26_architecture.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_mxfp4: tests/test_mimo26_gpu_mxfp4.o \
		k3_rocm_ops.o mimo26_weights.o mimo26_manifest.o \
		mimo26_architecture.o mimo26_attention.o mimo26_ops.o \
		mimo26_router.o glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

mimo26_rocm_ops.o: mimo26_rocm_ops.cu mimo26_rocm_ops.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_ops.o: tests/test_mimo26_gpu_ops.cu \
	mimo26_rocm_ops.h k3_rocm_ops.h mimo26_ops.h mimo26_router.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_ops: tests/test_mimo26_gpu_ops.o \
		mimo26_rocm_ops.o k3_rocm_ops.o mimo26_ops.o mimo26_router.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/test_mimo26_gpu_attention.o: tests/test_mimo26_gpu_attention.cu \
	mimo26_rocm_ops.h mimo26_attention.h mimo26_ops.h mimo26_architecture.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_attention: tests/test_mimo26_gpu_attention.o \
		mimo26_rocm_ops.o mimo26_attention.o mimo26_ops.o \
		mimo26_architecture.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

# G3: MiMo attention on the GPU, bit-exact against the CPU oracle.
mimo26-gpu-attention: tests/test_mimo26_gpu_attention
	./tests/test_mimo26_gpu_attention

# G2: MiMo's own GPU primitives, where its contracts diverge from K3's.
mimo26-gpu-ops: tests/test_mimo26_gpu_ops
	./tests/test_mimo26_gpu_ops

tools/mimo26_gpu_bench.o: tools/mimo26_gpu_bench.cu k3_rocm_ops.h
	$(HIPCC) $(HIPFLAGS) -I. -c -o $@ $<
tools/mimo26_gpu_bench: tools/mimo26_gpu_bench.o k3_rocm_ops.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

mimo26_rocm_layer.o: mimo26_rocm_layer.cu mimo26_rocm_layer.h \
	mimo26_rocm_ops.h k3_rocm_ops.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_layer.o: tests/test_mimo26_gpu_layer.cu \
	mimo26_rocm_layer.h mimo26_rocm_ops.h mimo26_layer.h mimo26_weights.h \
	mimo26_kv.h mimo26_manifest.h mimo26_attention.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_layer: tests/test_mimo26_gpu_layer.o \
		mimo26_rocm_layer.o mimo26_rocm_ops.o k3_rocm_ops.o \
		mimo26_layer.o mimo26_weights.o mimo26_kv.o mimo26_manifest.o \
		mimo26_architecture.o mimo26_attention.o mimo26_ops.o \
		mimo26_router.o glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

# G4: a whole layer on the GPU against the CPU layer, on real weights.
mimo26-gpu-layer: tests/test_mimo26_gpu_layer
	MIMO26_ROOT=$(MIMO26_ROOT) ./tests/test_mimo26_gpu_layer

mimo26_gpu_worker.o: mimo26_gpu_worker.cu mimo26_gpu_worker.h \
	mimo26_rocm_layer.h mimo26_rocm_ops.h k3_rocm_ops.h mimo26_weights.h \
	mimo26_kv.h mimo26_manifest.h mimo26_attention.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tools/mimo26_gpu_run.o: tools/mimo26_gpu_run.cu mimo26_gpu_worker.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tools/mimo26_gpu_run: tools/mimo26_gpu_run.o mimo26_gpu_worker.o \
		mimo26_rocm_layer.o mimo26_rocm_ops.o k3_rocm_ops.o \
		mimo26_weights.o mimo26_kv.o mimo26_manifest.o \
		mimo26_architecture.o mimo26_attention.o mimo26_ops.o \
		mimo26_router.o glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/test_mimo26_gpu_qualify.o: tests/test_mimo26_gpu_qualify.cu \
	mimo26_gpu_worker.h
	$(HIPCC) $(HIPFLAGS) -fno-fast-math -I. -c -o $@ $<
tests/test_mimo26_gpu_qualify: tests/test_mimo26_gpu_qualify.o \
		mimo26_gpu_worker.o mimo26_rocm_layer.o mimo26_rocm_ops.o \
		k3_rocm_ops.o mimo26_weights.o mimo26_kv.o mimo26_manifest.o \
		mimo26_architecture.o mimo26_attention.o mimo26_ops.o \
		mimo26_router.o glm53_fp8_oracle.o k3_safetensors.o k3_json.o
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(ROCM_LDLIBS)

# GPU operational qualification, mirroring mimo26-qualify on the CPU.
mimo26-gpu-qualify: tests/test_mimo26_gpu_qualify
	MIMO26_ROOT=$(MIMO26_ROOT) ./tests/test_mimo26_gpu_qualify

# The whole GPU gate set, in dependency order.
mimo26-gpu: mimo26-gpu-ops mimo26-gpu-attention mimo26-gpu-mxfp4 \
	mimo26-gpu-layer mimo26-gpu-qualify

# G1: the MiMo expert path on the GPU against the verified CPU dequantizer.
mimo26-gpu-mxfp4: tests/test_mimo26_gpu_mxfp4
	MIMO26_ROOT=$(MIMO26_ROOT) ./tests/test_mimo26_gpu_mxfp4

# M4 functional-quality gate. Teacher-forced over the corpus frozen in
# tests/mimo26_eval_spec.json, scored against thresholds frozen with it.
# Hours on CPU at roughly a minute per token.
mimo26-eval: tools/mimo26_eval
	MIMO26_ROOT=$(MIMO26_ROOT) PYTHONDONTWRITEBYTECODE=1 \
	  $(PYTHON) tests/mimo26_eval_tokens.py > mimo26-eval-tokens.txt
	./tools/mimo26_eval $(MIMO26_ROOT) < mimo26-eval-tokens.txt \
	  > mimo26-eval.jsonl
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/score_mimo26_eval.py \
	  mimo26-eval.jsonl

mimo26-budget:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tools/mimo26_budget.py \
		$(MIMO26_ROOT)

test-mzg2-bundle:
	PYTHONDONTWRITEBYTECODE=1 \
		$(PYTHON) -m unittest -v tests/test_build_mzg2_bundle.py

test-cpu: $(PORTABLE_CPU_TESTS) test-glm53-phase4 test-glm53-phase5a \
		tests/test_glm53_fp8_dynamic tests/test_glm53_process_memory \
		tests/test_glm53_static_layout tests/test_glm53_static_bindings \
		test-cache-analyzer \
	test-prefill-screen-analyzer test-anchor-recovery-analyzer \
	test-mzg2-bundle
	./tests/test_k3_expert_cache
	./tests/test_k3_prefix_bundle
	./tests/test_k3_prefix_reuse
	./tests/test_k3_prefix_catalog
	./tests/test_k3_prefill_route_index
	./tests/test_k3_prefill_timeline
	./tests/test_k3_q8_codec
	./tests/test_k3_json
	./tests/test_k3_openai
	./tests/test_k3_server_slot
	./tests/test_glm53_fp8_dynamic
	./tests/test_glm53_process_memory
	./tests/test_glm53_static_layout
	./tests/test_glm53_static_bindings

test-decode-cache-replay: tests/test_k3_expert_cache
	@test -n "$(MOONSHINE_DECODE_TRACE)" || \
		{ echo "error: set MOONSHINE_DECODE_TRACE"; exit 2; }
	@test -n "$(MOONSHINE_DECODE_CACHE_TRACE)" || \
		{ echo "error: set MOONSHINE_DECODE_CACHE_TRACE"; exit 2; }
	@test -n "$(MOONSHINE_DECODE_LEDGER_TRACE)" || \
		{ echo "error: set MOONSHINE_DECODE_LEDGER_TRACE"; exit 2; }
	@test -n "$(MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY)" || \
		{ echo "error: set MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY"; exit 2; }
	@test "$(MOONSHINE_DECODE_CACHE_FRESH_EMPTY_SOURCE)" = 0 -o \
		"$(MOONSHINE_DECODE_CACHE_FRESH_EMPTY_SOURCE)" = 1 || \
		{ echo "error: fresh-empty source flag must be 0 or 1"; exit 2; }
	@set -e; source_seen=0; provenance=; \
	if test "$(MOONSHINE_DECODE_CACHE_FRESH_EMPTY_SOURCE)" = 1; then \
		provenance=fresh-empty-source; \
	fi; \
	for capacity in $(MOONSHINE_DECODE_CACHE_CAPACITIES); do \
		if test "$$capacity" = \
			"$(MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY)"; then \
			source_seen=1; \
		fi; \
		./tests/test_k3_expert_cache \
			"$(MOONSHINE_DECODE_CACHE_TRACE)" \
			"$(MOONSHINE_DECODE_LEDGER_TRACE)" \
			"$(MOONSHINE_DECODE_TRACE)" \
			"$(MOONSHINE_DECODE_CACHE_SOURCE_CAPACITY)" "$$capacity" \
			$$provenance; \
	done; \
	test "$$source_seen" = 1 || \
		{ echo "error: capacity sweep must include source capacity"; exit 2; }

test: \
	test-glm53-phase4 \
	test-glm53-phase5a \
	test-glm53-phase5b \
	test-glm53-phase5c \
	test-glm53-phase5d-kda \
	tests/test_k3_expert_cache \
	tests/test_k3_prefix_reuse \
	tests/test_k3_json \
	tests/test_k3_openai \
	tests/test_k3_rocm_components \
	tests/test_k3_mxfp4_envelope \
	tests/test_k3_kda_recurrent \
	tests/test_k3_mla_decode \
	tests/test_k3_prefill_ops
	./tests/test_k3_expert_cache
	./tests/test_k3_prefix_reuse
	./tests/test_k3_json
	./tests/test_k3_openai
	./tests/test_k3_rocm_components
	./tests/test_k3_mxfp4_envelope
	./tests/test_k3_kda_recurrent
	./tests/test_k3_mla_decode
	./tests/test_k3_prefill_ops

test-openai-sdk:
	@$(PYTHON) -c 'import openai' 2>/dev/null || \
		{ echo "error: install tests/requirements-sdk.txt in an isolated environment"; exit 2; }
	$(PYTHON) tests/test_openai_sdk.py

check-model:
	@test -n "$(MOONSHINE_MODEL)" || \
		{ echo "error: set MOONSHINE_MODEL=/path/to/moonshotai__Kimi-K3"; exit 2; }
	@test -f "$(MOONSHINE_MODEL)/model-00001-of-000096.safetensors" || \
		{ echo "error: MOONSHINE_MODEL is not the expected 96-shard SafeTensors tree"; exit 2; }

test-model-layout: check-model \
	tests/test_k3_safetensors tests/test_k3_prefill_plan
	./tests/test_k3_safetensors "$(MOONSHINE_MODEL)"
	./tests/test_k3_prefill_plan "$(MOONSHINE_MODEL)"

test-model-components: check-model \
	tests/test_k3_q8_projection \
	tests/test_k3_kda_layer_smoke \
	tests/test_k3_mla_layer_smoke \
	tests/test_k3_embedding_output \
	tests/test_k3_residual_spine \
	tests/test_k3_dense_mlp \
	tests/test_k3_expert_smoke \
	tests/test_k3_moe_smoke
	./tests/test_k3_q8_projection "$(MOONSHINE_MODEL)"
	./tests/test_k3_kda_layer_smoke "$(MOONSHINE_MODEL)"
	./tests/test_k3_mla_layer_smoke "$(MOONSHINE_MODEL)"
	./tests/test_k3_embedding_output "$(MOONSHINE_MODEL)"
	./tests/test_k3_residual_spine "$(MOONSHINE_MODEL)"
	./tests/test_k3_dense_mlp "$(MOONSHINE_MODEL)"
	./tests/test_k3_expert_smoke "$(MOONSHINE_MODEL)"
	./tests/test_k3_moe_smoke "$(MOONSHINE_MODEL)"

test-prefix-checkpoint: check-model tests/test_k3_prefix_checkpoint
	./tests/test_k3_prefix_checkpoint "$(MOONSHINE_MODEL)"

test-engine-init: check-model tests/test_k3_engine_init
	./tests/test_k3_engine_init "$(MOONSHINE_MODEL)" "$(MOONSHINE_CONTEXT)"

test-engine-hello: check-model tests/test_k3_engine_hello
	./tests/test_k3_engine_hello \
		"$(MOONSHINE_MODEL)" q8 32 "$(MOONSHINE_CONTEXT)"

test-chat-hello: check-model tests/test_k3_chat_session
	./tests/test_k3_chat_session "$(MOONSHINE_MODEL)"

test-state-checkpoint: check-model tests/test_k3_state_checkpoint
	./tests/test_k3_state_checkpoint "$(MOONSHINE_MODEL)" "$(MOONSHINE_STATE_DIR)"

test-tokenizer: check-model tests/test_k3_tokenizer
	./tests/test_k3_tokenizer "$(MOONSHINE_MODEL)"

test-prefill-2: check-model tests/test_k3_prefill_chunk
	./tests/test_k3_prefill_chunk "$(MOONSHINE_MODEL)"

test-prefill-scale: check-model tests/test_k3_prefill_512
	./tests/test_k3_prefill_512 \
		"$(MOONSHINE_MODEL)" "$(MOONSHINE_PREFILL_TOKENS)" \
		"$(MOONSHINE_CONTEXT)"

test-prefill-kda-blas: check-model tests/test_k3_prefill_512
	./tests/test_k3_prefill_512 \
		"$(MOONSHINE_MODEL)" "$(MOONSHINE_PREFILL_TOKENS)" \
		"$(MOONSHINE_CONTEXT)" kda-blas

test-prefill-crossover: check-model tests/test_k3_prefill_crossover
	./tests/test_k3_prefill_crossover \
		"$(MOONSHINE_MODEL)" $(MOONSHINE_CROSSOVER_TOKENS)

test-prefill-gemm-shapes: tests/test_k3_prefill_gemm_shapes
	./tests/test_k3_prefill_gemm_shapes

test-long-context-retrieval: check-model tests/test_k3_long_context
	MOONSHINE_RETRIEVAL_TARGET="$(MOONSHINE_RETRIEVAL_TARGET)" \
	MOONSHINE_RETRIEVAL_BACKEND="$(MOONSHINE_RETRIEVAL_BACKEND)" \
		./tests/test_k3_long_context "$(MOONSHINE_MODEL)"

test-mla-batch-determinism: tests/test_k3_mla_batch_determinism
	./tests/test_k3_mla_batch_determinism

test-mla-batch-kernels: tests/test_k3_mla_batch_kernels
	./tests/test_k3_mla_batch_kernels

test-moe-tail-profile: tests/test_k3_moe_tail_profile
	./tests/test_k3_moe_tail_profile

test-reduction-qualification: check-model \
	tests/test_k3_rocm_components \
	tests/test_k3_mxfp4_envelope \
	tests/test_k3_expert_smoke \
	tests/test_k3_moe_smoke \
	tests/test_k3_engine_init \
	tests/test_k3_engine_hello \
	tests/test_k3_tokenizer
	./tests/test_k3_rocm_components
	./tests/test_k3_mxfp4_envelope
	./tests/test_k3_expert_smoke "$(MOONSHINE_MODEL)"
	./tests/test_k3_moe_smoke "$(MOONSHINE_MODEL)"
	./tests/test_k3_engine_init "$(MOONSHINE_MODEL)" 8192
	./tests/test_k3_tokenizer "$(MOONSHINE_MODEL)"
	./tests/test_k3_engine_hello "$(MOONSHINE_MODEL)" q8 32 8192

clean:
	rm -f libmoonshine.a moonshine-chat moonshine-server \
		k3_chat_cli.o k3_server.o k3_server_slot.o k3_q8_codec.o \
		tools/transcode_mzg2_layer tools/transcode_mzg2_layer.o \
		tools/screen_static_q8 tools/screen_static_q8.o \
		$(K3_OBJS) $(GLM53_OBJS) tests/*.o $(ALL_TESTS)
