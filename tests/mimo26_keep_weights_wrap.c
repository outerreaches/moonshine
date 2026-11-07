/* Test-only clean request reset experiment. Not for recovery/quarantine paths.
 * The normal worker reset still clears KV and resolved pointers. This suppresses
 * only the weight-cache reset. Same-process counters are intentionally cumulative.
 */
#include "k3_expert_cache.h"
#include <stdio.h>
bool __wrap_k3_expert_cache_reset(k3_expert_cache *cache,char *error,size_t size){
    if(!cache)return false;
    if(error&&size)error[0]=0;
    fputs("TEST_ONLY retained weight cache on clean reset\n",stderr);
    return true;
}
