// Full-vector equality gate for prefix reuse.
//
// The qualification evidence for reuse was byte-identical assistant TEXT
// across turns, which is weaker than the gate expert-major had to pass: text
// equality survives a small logit perturbation that greedy decoding rounds
// away, and it says nothing about the distribution behind the chosen token.
// This compares raw float32 logit vectors with memcmp and no tolerance.
//
// Three ways of arriving at the same position must agree exactly:
//
//   FRESH    prefill tokens[0..M) in one call
//   CONTINUE prefill tokens[0..N), then prefill tokens[N..M) on the same
//            worker -- the in-memory tier's path
//   RESTORE  prefill tokens[0..N), export, reset, import, then prefill
//            tokens[N..M) -- the disk tier's path
//
// RESTORE is the one that matters: it is the only path where committed
// history crosses a file, and a loss there would show as a plausible
// completion computed against damaged history rather than as a crash.
//
// Decodes after the split are captured too, because a restore that got the
// position or the windowed layers' first_position wrong would still produce
// correct logits for the very next token and diverge only as generation walks
// past the boundary.
//
//   mimo26_prefix_reuse_gate ROOT OUT_PREFIX SLOTS CHUNK CONTEXT SPLIT DECODES
#include "mimo26_gpu_worker.h"
#include "mimo26_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char *TEXT =
    "The harbour office keeps tide tables, mooring records and weather logs. "
    "Each morning the duty officer checks the barometer and signs the register. "
    "On Thursday the keeper found a folder describing how the breakwater was laid. "
    "Its diagrams showed the bay in autumn, when swell pushes against the wall. "
    "Surveyors compared those diagrams with sonar taken after the winter storms. "
    "They chose to keep the original blocks while replacing the capping stones. "
    "A visiting class asked how anyone knows which parts are original at all. "
    "The keeper explained that the records disagree, and that the disagreement "
    "is itself the finding rather than a gap to be filled in with a guess. "
    "By evening the folder had been photographed, catalogued and shelved again. ";

static std::vector<float> capture(const float *logits, size_t count)
{
    return std::vector<float>(logits, logits + count);
}

static bool identical(const std::vector<float> &a, const std::vector<float> &b)
{
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s ROOT OUT SLOTS CHUNK CONTEXT SPLIT DECODES\n",
                     argv[0]);
        return 2;
    }
    const char *root = argv[1];
    const std::string out = argv[2];
    const unsigned slots = (unsigned)std::strtoul(argv[3], nullptr, 10);
    const unsigned chunk = (unsigned)std::strtoul(argv[4], nullptr, 10);
    const unsigned context = (unsigned)std::strtoul(argv[5], nullptr, 10);
    const unsigned split = (unsigned)std::strtoul(argv[6], nullptr, 10);
    const unsigned decodes = (unsigned)std::strtoul(argv[7], nullptr, 10);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    char error[512] = {0};
    mimo26_tokenizer *tokenizer = nullptr;
    if (!mimo26_tokenizer_create(&tokenizer, root, error, sizeof error)) {
        std::fprintf(stderr, "tokenizer: %s\n", error);
        return 1;
    }
    mimo26_token_buffer prompt{};
    std::string text;
    while (text.size() < 6000) text += TEXT;
    if (!mimo26_tokenizer_encode(tokenizer, text.c_str(), false, &prompt, error,
                                 sizeof error)) {
        std::fprintf(stderr, "encode: %s\n", error);
        return 1;
    }
    if (prompt.count <= split + 2u || split == 0u) {
        std::fprintf(stderr, "prompt of %zu tokens cannot split at %u\n",
                     prompt.count, split);
        return 2;
    }
    const size_t total = prompt.count < context ? prompt.count : context - decodes - 1u;
    std::printf("prompt %zu tokens, using %zu, split at %u, %u decodes\n",
                prompt.count, total, split, decodes);

    mimo26_gpu_worker_config config{};
    mimo26_gpu_worker_config_defaults(&config);
    config.expert_slots_per_layer = (uint16_t)slots;
    config.prefill_chunk = (uint16_t)chunk;
    config.global_kv_capacity = context;
    mimo26_gpu_worker *worker = nullptr;
    if (mimo26_gpu_worker_create(&worker, root, &config, error, sizeof error) !=
        MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "worker: %s\n", error);
        return 1;
    }
    std::vector<float> logits(152576u);
    const std::string state = out + ".state";

    // Capture the prefill vector plus `decodes` following vectors, so a
    // boundary error that only shows as generation walks past the split is
    // visible rather than latent.
    auto run_tail = [&](const char *label, size_t from) {
        std::vector<std::vector<float>> captured;
        if (mimo26_gpu_worker_prefill(worker, prompt.ids + from,
                                      (uint32_t)(total - from), logits.data(),
                                      nullptr, nullptr, error,
                                      sizeof error) != MIMO26_GPU_WORKER_OK) {
            std::fprintf(stderr, "%s prefill: %s\n", label, error);
            std::exit(1);
        }
        captured.push_back(capture(logits.data(), logits.size()));
        uint32_t next = mimo26_gpu_worker_argmax(logits.data());
        for (unsigned d = 0; d < decodes; d++) {
            if (mimo26_gpu_worker_decode(worker, next, logits.data(), error,
                                         sizeof error) != MIMO26_GPU_WORKER_OK) {
                std::fprintf(stderr, "%s decode %u: %s\n", label, d, error);
                std::exit(1);
            }
            captured.push_back(capture(logits.data(), logits.size()));
            next = mimo26_gpu_worker_argmax(logits.data());
        }
        return captured;
    };

    // FRESH: the whole prompt in one prefill.
    mimo26_gpu_worker_reset(worker);
    auto fresh = run_tail("fresh", 0);

    // CONTINUE: split across two prefills on the same worker.
    mimo26_gpu_worker_reset(worker);
    if (mimo26_gpu_worker_prefill(worker, prompt.ids, split, logits.data(),
                                  nullptr, nullptr, error, sizeof error) !=
        MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "head prefill: %s\n", error);
        return 1;
    }
    auto cont = run_tail("continue", split);

    // RESTORE: same split, but the head crosses a file.
    mimo26_gpu_worker_reset(worker);
    if (mimo26_gpu_worker_prefill(worker, prompt.ids, split, logits.data(),
                                  nullptr, nullptr, error, sizeof error) !=
        MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "head prefill: %s\n", error);
        return 1;
    }
    mimo26_kv_state_info wrote{};
    if (mimo26_gpu_worker_export_state(worker, state.c_str(), &wrote, error,
                                       sizeof error) != MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "export: %s\n", error);
        return 1;
    }
    mimo26_gpu_worker_reset(worker);
    mimo26_kv_state_info read{};
    const double before = (double)wrote.length;
    if (mimo26_gpu_worker_import_state(worker, state.c_str(), &read, error,
                                       sizeof error) != MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "import: %s\n", error);
        return 1;
    }
    if (read.length != (uint64_t)before ||
        mimo26_gpu_worker_position(worker) != read.length) {
        std::fprintf(stderr, "restored length %llu != %llu or position %llu\n",
                     (unsigned long long)read.length,
                     (unsigned long long)before,
                     (unsigned long long)mimo26_gpu_worker_position(worker));
        return 1;
    }
    std::printf("checkpoint %.1f MiB, export %.3f s, import %.3f s\n",
                (double)wrote.file_bytes / (1024.0 * 1024.0),
                wrote.wall_seconds, read.wall_seconds);
    auto restored = run_tail("restore", split);

    unsigned bad = 0;
    for (size_t i = 0; i < fresh.size(); i++) {
        const bool c = identical(fresh[i], cont[i]);
        const bool r = identical(fresh[i], restored[i]);
        if (!c || !r) {
            bad++;
            std::printf("  vector %zu (%s): continue %s, restore %s\n", i,
                        i == 0 ? "prefill" : "decode",
                        c ? "identical" : "DIFFERS",
                        r ? "identical" : "DIFFERS");
        }
    }
    std::printf("%zu vectors compared (1 prefill + %u decodes), %u differing\n",
                fresh.size(), decodes, bad);
    std::printf("%s full-vector prefix reuse: fresh == continue == restore\n",
                bad == 0 ? "PASS" : "FAIL");
    std::remove(state.c_str());
    mimo26_gpu_worker_destroy(worker);
    mimo26_token_buffer_free(&prompt);
    mimo26_tokenizer_destroy(tokenizer);
    return bad == 0 ? 0 : 1;
}
