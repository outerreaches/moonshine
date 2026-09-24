// Full-vector equality gate for expert-major MoE execution, at a chosen
// profile rather than the fixed one the route screen uses.
//
// Expert-major only changes the prefill batch path, but prefill leaves state
// behind -- KV history, position, the expert cache -- so the gate captures the
// prefill logits AND several subsequent decode vectors. A grouping bug that
// corrupted KV or bled the output stash would pass a prefill-only comparison
// and fail here.
//
// Writes raw float32 logit vectors so the caller can compare with memcmp; no
// tolerance, no argmax-only check. Text is long enough to span several chunks
// at chunk 128, which is the case per-token execution never exercises.
//
//   mimo26_expert_major_gate ROOT OUT_PREFIX SLOTS CHUNK CONTEXT DECODES [TOKENS]
#include "mimo26_gpu_worker.h"
#include "mimo26_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char *TEXT =
    "The town library keeps maps, letters, and records of local weather. "
    "Each morning a volunteer checks the catalogue and returns borrowed books. "
    "On Tuesday the archivist found a notebook describing how the bridge was built. "
    "Its drawings showed the river in winter, when ice pressed against the stone. "
    "Engineers compared those drawings with recent measurements before repairs. "
    "They chose to preserve the original stones while replacing the walkway. "
    "A school class visited the archive to learn how evidence changes explanations. "
    "The students recorded questions, checked dates, and separated observation from guess. "
    "After lunch they wrote a report explaining which claims the documents supported. "
    "The teacher asked them to keep uncertain details visible rather than fill gaps. "
    "In the afternoon a surveyor arrived with instruments to measure the arch. "
    "She explained that small errors compound across a long span of masonry. "
    "The archivist offered tea and a chair beside the window facing the water. "
    "They talked about how records survive floods, fires, and simple neglect. "
    "By evening the notebook had been photographed, catalogued, and returned. "
    "A copy went to the council, another to the county record office upstairs. "
    "The bridge itself stood unchanged, older than every document describing it. "
    "Nobody could say which of its stones had been replaced and which were first. ";

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr,
                     "usage: %s ROOT OUT SLOTS CHUNK CONTEXT DECODES [TOKENS]\n",
                     argv[0]);
        return 2;
    }
    const char *root = argv[1];
    const std::string out = argv[2];
    const unsigned slots = (unsigned)std::strtoul(argv[3], nullptr, 10);
    const unsigned chunk = (unsigned)std::strtoul(argv[4], nullptr, 10);
    const unsigned context = (unsigned)std::strtoul(argv[5], nullptr, 10);
    const unsigned decodes = (unsigned)std::strtoul(argv[6], nullptr, 10);
    const unsigned forced = argc == 8
        ? (unsigned)std::strtoul(argv[7], nullptr, 10) : 0u;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    char error[512] = {0};
    mimo26_tokenizer *tokenizer = nullptr;
    if (!mimo26_tokenizer_create(&tokenizer, root, error, sizeof error)) {
        std::fprintf(stderr, "tokenizer: %s\n", error);
        return 1;
    }
    std::string text;
    for (int i = 0; i < 3; i++) {
        text += TEXT;
    }
    mimo26_token_buffer ids{};
    if (!mimo26_tokenizer_encode(tokenizer, text.c_str(), false, &ids, error,
                                 sizeof error)) {
        std::fprintf(stderr, "encode: %s\n", error);
        return 1;
    }
    /* Trim so prefill spans whole chunks plus a partial one: the boundary is
     * where a grouping bug is most likely to show. */
    size_t count = ids.count;
    const size_t want = forced ? (size_t)forced : (size_t)chunk * 3u + chunk / 2u;
    if (count > want) {
        count = want;
    }
    if (count < 2u) {
        std::fprintf(stderr, "prompt too short: %zu\n", count);
        return 1;
    }

    mimo26_gpu_worker_config config;
    mimo26_gpu_worker_config_defaults(&config);
    config.expert_slots_per_layer = (uint16_t)slots;
    config.prefill_chunk = (uint16_t)chunk;
    config.global_kv_capacity = context;

    mimo26_gpu_worker *worker = nullptr;
    if (mimo26_gpu_worker_create(&worker, root, &config, error,
                                 sizeof error) != MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "worker: %s\n", error);
        return 1;
    }
    std::vector<float> logits(152576);
    auto dump = [&](const std::string &name) -> bool {
        FILE *f = std::fopen((out + "-" + name + ".bin").c_str(), "wbx");
        if (!f) return false;
        const bool ok =
            std::fwrite(logits.data(), 4, logits.size(), f) == logits.size();
        return std::fclose(f) == 0 && ok;
    };

    if (mimo26_gpu_worker_prefill(worker, ids.ids, count, logits.data(),
                                  nullptr, nullptr, error,
                                  sizeof error) != MIMO26_GPU_WORKER_OK) {
        std::fprintf(stderr, "prefill: %s\n", error);
        return 1;
    }
    if (!dump("prefill")) {
        std::fprintf(stderr, "write prefill\n");
        return 1;
    }
    uint32_t next = mimo26_gpu_worker_argmax(logits.data());
    std::printf("{\"phase\":\"prefill\",\"tokens\":%zu,\"slots\":%u,"
                "\"chunk\":%u,\"argmax\":%u}\n", count, slots, chunk, next);

    for (unsigned d = 0; d < decodes; d++) {
        if (mimo26_gpu_worker_decode(worker, next, logits.data(), error,
                                     sizeof error) != MIMO26_GPU_WORKER_OK) {
            std::fprintf(stderr, "decode %u: %s\n", d, error);
            return 1;
        }
        if (!dump("decode" + std::to_string(d))) {
            std::fprintf(stderr, "write decode %u\n", d);
            return 1;
        }
        next = mimo26_gpu_worker_argmax(logits.data());
        std::printf("{\"phase\":\"decode\",\"step\":%u,\"argmax\":%u}\n", d,
                    next);
    }

    mimo26_gpu_worker_stats stats{};
    mimo26_gpu_worker_get_stats(worker, &stats);
    std::printf("{\"accesses\":%llu,\"hits\":%llu,\"uploads\":%llu,"
                "\"resident_bytes\":%llu}\n",
                (unsigned long long)stats.expert_accesses,
                (unsigned long long)stats.expert_hits,
                (unsigned long long)stats.expert_uploads,
                (unsigned long long)mimo26_gpu_worker_resident_bytes(worker));
    mimo26_gpu_worker_destroy(worker);
    mimo26_token_buffer_free(&ids);
    mimo26_tokenizer_destroy(tokenizer);
    return 0;
}
