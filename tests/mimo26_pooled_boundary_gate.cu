// Corrected-worker comparison: decode oracle, full chunks, partial tail,
// continuation across the sliding-window boundary, then forced decode.
#include "mimo26_gpu_worker.h"
#include "mimo26_tokenizer.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
extern "C" void mimo26_test_allocation_guards();
static void save(const std::string &path, const void *data, size_t bytes) {
    FILE *f = fopen(path.c_str(), "wbx"); assert(f);
    assert(fwrite(data, 1, bytes, f) == bytes && !fclose(f));
}
int main(int argc, char **argv) {
    assert(argc == 5); setvbuf(stdout, nullptr, _IOLBF, 0);
    unsigned slots = std::stoul(argv[3]), chunk = std::stoul(argv[4]);
    assert((slots == 16 || slots == 128) && (chunk == 0 || chunk == 32 || chunk == 64 || chunk == 128));
    char error[512]{}; mimo26_tokenizer *tokenizer = nullptr;
    assert(mimo26_tokenizer_create(&tokenizer, argv[1], error, sizeof error));
    std::string text;
    for (unsigned i = 0; i < 8; ++i)
        text += "Review a Python window_sum implementation for an off-by-one error. "
                "A warehouse has 17, 24 and 39 boxes, each containing six packets. "
                "La bibliothèque ouvrira à neuf heures samedi, sauf en cas de panne. "
                "Distinguish certain facts from conditional statements. ";
    mimo26_token_buffer ids{};
    assert(mimo26_tokenizer_encode(tokenizer, text.c_str(), false, &ids, error, sizeof error));
    assert(ids.count >= 257);
    save(std::string(argv[2])+"/tokens.bin", ids.ids, 257 * sizeof(uint32_t));
    mimo26_gpu_worker_config c; mimo26_gpu_worker_config_defaults(&c);
    c.expert_slots_per_layer = slots; c.global_kv_capacity = 512;
    c.prefill_chunk = chunk; c.expert_lookahead = chunk != 0;
    mimo26_gpu_worker *worker = nullptr;
    assert(mimo26_gpu_worker_create(&worker, argv[1], &c, error, sizeof error) == MIMO26_GPU_WORKER_OK);
    std::vector<float> logits(152576);
    size_t position = 0;
    for (size_t count : {128u, 1u, 128u}) {
        assert(mimo26_gpu_worker_prefill(worker, ids.ids + position, count, logits.data(),
                    nullptr, nullptr, error, sizeof error) == MIMO26_GPU_WORKER_OK);
        position += count;
        assert(mimo26_gpu_worker_position(worker) == position);
        mimo26_test_allocation_guards();
        save(std::string(argv[2])+"/prefill-"+std::to_string(position)+".bin", logits.data(), logits.size()*4);
        printf("PASS boundary %zu\n", position);
    }
    for (uint32_t token : {13u, 42u}) {
        assert(mimo26_gpu_worker_decode(worker, token, logits.data(), error, sizeof error) == MIMO26_GPU_WORKER_OK);
        mimo26_test_allocation_guards();
        save(std::string(argv[2])+"/decode-"+std::to_string(token)+".bin", logits.data(), logits.size()*4);
    }
    mimo26_gpu_worker_destroy(worker);
    mimo26_token_buffer_free(&ids); mimo26_tokenizer_destroy(tokenizer);
    puts("PASS corrected-worker boundary gate");
}
