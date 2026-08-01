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
LDLIBS ?= -lm -pthread
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
ICU_LDLIBS ?= $(shell pkg-config --libs icu-i18n 2>/dev/null || \
	echo -licui18n -licuuc -licudata)

MOONSHINE_MODEL ?=
MOONSHINE_CONTEXT ?= 8192
MOONSHINE_PREFILL_TOKENS ?= 512
MOONSHINE_CROSSOVER_TOKENS ?= 2 3 4 6 8 12 16 24 32 42
MOONSHINE_RETRIEVAL_TARGET ?= 16000
MOONSHINE_STATE_DIR ?= /tmp

K3_OBJS := \
	k3_chat.o \
	k3_engine.o \
	k3_prefill.o \
	k3_static_store.o \
	k3_expert_cache.o \
	k3_io_uring.o \
	k3_json.o \
	k3_openai.o \
	k3_rocm_ops.o \
	k3_safetensors.o \
	k3_tokenizer.o

PORTABLE_CPU_TESTS := \
	tests/test_k3_expert_cache \
	tests/test_k3_json \
	tests/test_k3_openai

MODEL_CPU_TESTS := \
	tests/test_k3_prefill_plan \
	tests/test_k3_safetensors \
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
	tests/test_k3_mla_decode \
	tests/test_k3_mla_layer_smoke \
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
	tests/test_k3_long_context

ALL_TESTS := $(CPU_TESTS) $(ASSET_TESTS) $(ROCM_TESTS) $(CHAT_TESTS)

.PHONY: all help tests test test-cpu check-model \
	test-model-layout test-model-components test-engine-init \
	test-engine-hello test-chat-hello test-state-checkpoint test-tokenizer \
	test-prefill-2 test-prefill-scale test-prefill-kda-blas \
	test-prefill-crossover test-long-context-retrieval \
	test-reduction-qualification \
	test-openai-sdk clean

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
	@echo "  make test-long-context-retrieval MOONSHINE_MODEL=/path/to/Kimi-K3 MOONSHINE_RETRIEVAL_TARGET=512|16000|32000"
	@echo "                               Run a deterministic natural-text gate"
	@echo "  make test-reduction-qualification MOONSHINE_MODEL=/path/to/Kimi-K3"
	@echo "                               Run the MXFP4 reduction-change gate bundle"
	@echo "  make clean                  Remove local build products"

libmoonshine.a: $(K3_OBJS)
	$(AR) rcs $@ $^

tests: $(ALL_TESTS)

%.o: %.c
	$(CC) $(CFLAGS) -I. -c -o $@ $<

%.o: %.cu
	$(HIPCC) $(HIPFLAGS) -I. -c -o $@ $<

k3_engine.o: k3_engine.cu k3_engine_state.inc k3_engine_prefill.inc k3_engine.h \
	k3_prefill.h k3_static_store.h k3_expert_cache.h k3_io_uring.h \
	k3_rocm_ops.h k3_safetensors.h
k3_chat.o: k3_chat.c k3_chat.h k3_engine.h k3_tokenizer.h
k3_chat_cli.o: k3_chat_cli.c k3_chat.h moonshine_version.h
k3_server.o: k3_server.c k3_chat.h k3_json.h k3_openai.h \
	moonshine_version.h
k3_prefill.o: k3_prefill.c k3_prefill.h k3_safetensors.h
k3_static_store.o: k3_static_store.cu k3_static_store.h \
	k3_rocm_ops.h k3_safetensors.h
k3_expert_cache.o: k3_expert_cache.c k3_expert_cache.h
k3_io_uring.o: k3_io_uring.c k3_io_uring.h
k3_json.o: k3_json.c k3_json.h
k3_openai.o: k3_openai.c k3_openai.h k3_json.h k3_chat.h
k3_rocm_ops.o: k3_rocm_ops.cu k3_rocm_ops.h
k3_safetensors.o: k3_safetensors.c k3_safetensors.h
k3_tokenizer.o: k3_tokenizer.c k3_tokenizer.h
tests/test_k3_chat_session.o: tests/test_k3_chat_session.c k3_chat.h
tests/test_k3_long_context.o: tests/test_k3_long_context.c k3_chat.h
tests/test_k3_openai.o: tests/test_k3_openai.c k3_openai.h k3_chat.h
tests/test_k3_tokenizer.o: tests/test_k3_tokenizer.c k3_tokenizer.h

tests/test_k3_expert_cache: tests/test_k3_expert_cache.o k3_expert_cache.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_k3_json: tests/test_k3_json.o k3_json.o
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

moonshine-server: k3_server.o libmoonshine.a
	$(HIPCC) $(HIPFLAGS) -o $@ k3_server.o libmoonshine.a \
		$(ROCM_LDLIBS) $(ICU_LDLIBS)

$(ROCM_TESTS): %: %.o libmoonshine.a
	$(HIPCC) $(HIPFLAGS) -o $@ $< libmoonshine.a $(ROCM_LDLIBS)

test-cpu: $(PORTABLE_CPU_TESTS)
	./tests/test_k3_expert_cache
	./tests/test_k3_json
	./tests/test_k3_openai

test: \
	tests/test_k3_expert_cache \
	tests/test_k3_json \
	tests/test_k3_openai \
	tests/test_k3_rocm_components \
	tests/test_k3_mxfp4_envelope \
	tests/test_k3_kda_recurrent \
	tests/test_k3_mla_decode \
	tests/test_k3_prefill_ops
	./tests/test_k3_expert_cache
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

test-long-context-retrieval: check-model tests/test_k3_long_context
	MOONSHINE_RETRIEVAL_TARGET="$(MOONSHINE_RETRIEVAL_TARGET)" \
		./tests/test_k3_long_context "$(MOONSHINE_MODEL)"

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
		k3_chat_cli.o k3_server.o \
		$(K3_OBJS) tests/*.o $(ALL_TESTS)
