// Isolated load-only probe. Never changes worker allocation or VM policy.
#include "mimo26_gpu_worker.h"
#include <cstdio>
#include <cstdlib>
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    unsigned slots = unsigned(std::strtoul(argv[2], nullptr, 10));
    if (slots != 48 && slots != 96 && slots != 144) return 2;
    mimo26_gpu_worker_config config;
    mimo26_gpu_worker_config_defaults(&config);
    config.global_kv_capacity = 256;
    config.prefill_chunk = 32;
    config.expert_slots_per_layer = slots;
    printf("{\"phase\":\"start\",\"slots\":%u,\"planned_bytes\":%llu}\n",
           slots, (unsigned long long)mimo26_gpu_worker_planned_bytes(&config));
    mimo26_gpu_worker *worker = nullptr;
    char error[512]{};
    auto status = mimo26_gpu_worker_create(&worker, argv[1], &config, error, sizeof error);
    printf("{\"phase\":\"loaded\",\"status\":%d,\"resident_bytes\":%llu}\n",
           int(status), (unsigned long long)mimo26_gpu_worker_resident_bytes(worker));
    if (status != MIMO26_GPU_WORKER_OK) fprintf(stderr, "LOAD_FAILED %s\n", error);
    mimo26_gpu_worker_destroy(worker);
    return status == MIMO26_GPU_WORKER_OK ? 0 : 1;
}
