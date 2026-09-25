#include "../mimo26_server_options.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Mirrors what main() gets from mimo26_gpu_worker_config_defaults before it
 * parses. Duplicated rather than called because this test is deliberately
 * CPU-only and does not link the HIP worker; keep it in step with
 * mimo26_gpu_worker_config_defaults. */
static void server_defaults(mimo26_server_options *o) {
    o->worker.expert_slots_per_layer = 160u;
    o->worker.global_kv_capacity = 131072u;
    o->worker.prefill_chunk = 128u;
    o->worker.expert_lookahead = true;
    o->worker.expert_major = true;
}

static mimo26_server_options parsed;
static unsigned checks;
static void check(bool expected, int argc, char **argv) {
    memset(&parsed,0,sizeof parsed);
    parsed.worker.global_kv_capacity=2048;
    parsed.worker.expert_slots_per_layer=16;
    parsed.worker.prefill_chunk=32;
    parsed.worker.memory_limit_bytes=12345;
    mimo26_server_options before;
    memcpy(&before,&parsed,sizeof before);
    char error[256]={0};
    const bool got_it = mimo26_server_parse_options(argc,argv,&parsed,error,sizeof error);
    if (got_it != expected) {
        fprintf(stderr, "case %u expected %d got %d:", checks, (int)expected, (int)got_it);
        for (int i = 1; i < argc; i++) fprintf(stderr, " %s", argv[i]);
        fprintf(stderr, "  error=%s\n", error);
    }
    assert(got_it==expected);
    if (!expected) assert(error[0]&&!memcmp(&before,&parsed,sizeof before));
    else assert(parsed.worker.memory_limit_bytes==12345);
    ++checks;
}
#define CHECK(expected, ...) do { char *args[]={"server",__VA_ARGS__}; check(expected,(int)(sizeof args/sizeof *args),args); } while(0)
int main(void) {
    CHECK(true,"root");
    assert(!strcmp(parsed.host,"127.0.0.1")&&parsed.port==8640);
    assert(!parsed.worker.expert_lookahead&&parsed.worker.prefill_chunk==32);
    assert(parsed.retain_experts);   /* on by default since 2026-09-25 */
    assert(parsed.worker.expert_slots_per_layer==16&&parsed.worker.global_kv_capacity==2048);
    CHECK(true,"root","--expert-lookahead","on","--prefill-chunk","64","--slots","48","--context","1024","--port","8765","--host","127.0.0.1");
    assert(parsed.worker.expert_lookahead&&parsed.worker.prefill_chunk==64&&parsed.worker.expert_slots_per_layer==48);
    assert(parsed.worker.global_kv_capacity==1024&&parsed.port==8765);
    CHECK(true,"root","--expert-lookahead","off","--prefill-chunk","0");
    CHECK(true,"root","--retain-experts","on"); assert(parsed.retain_experts);
    CHECK(true,"root","--retain-experts","off"); assert(!parsed.retain_experts);
    CHECK(false,"root","--retain-experts","true");
    CHECK(false,"root","--retain-experts");
    CHECK(false,"root","--retain-experts","on","--retain-experts","off");
    CHECK(false,"root","--expert-lookahead","on","--prefill-chunk","0");
    CHECK(false,"root","--expert-lookahead","true");
    CHECK(false,"root","--expert-lookahead");
    CHECK(false,"root","--expert-lookahead","on","--expert-lookahead","off");
    CHECK(false,"root","--unknown","on");
    CHECK(false,"root","--slots","48","orphan");
    CHECK(false,"");
    CHECK(false,"--expert-lookahead");
    CHECK(false,"root","--host","localhost");
    CHECK(false,"root","--host","256.0.0.1");
    CHECK(true,"root","--port","65535","--slots","256","--prefill-chunk","128","--context","4294967295");
    CHECK(true,"root","--port","1","--slots","8","--context","1","--prefill-chunk","1");
    CHECK(false,"root","--port","0"); CHECK(false,"root","--port","65536");
    CHECK(false,"root","--slots","7"); CHECK(false,"root","--slots","257");
    CHECK(false,"root","--prefill-chunk","129");
    CHECK(false,"root","--context","0"); CHECK(false,"root","--context","4294967296");
    /* The memory guard's floor: 0 disables it, and it must reject junk the
     * same way every other numeric option does. Default is 8 GiB. */
    CHECK(true,"root","--min-headroom-gib","0");
    CHECK(true,"root","--min-headroom-gib","512");
    CHECK(false,"root","--min-headroom-gib","513");
    CHECK(false,"root","--min-headroom-gib","-1");
    CHECK(false,"root","--min-headroom-gib","20junk");
    CHECK(false,"root","--min-headroom-gib","");
    {
        mimo26_server_options got; char error[256];
        memset(&got,0,sizeof got);
        server_defaults(&got);
        const char *argv[]={"x","root"};
        assert(mimo26_server_parse_options(2,(char**)argv,&got,error,sizeof error));
        assert(got.min_headroom_gib==8u);
        const char *argv2[]={"x","root","--min-headroom-gib","12"};
        assert(mimo26_server_parse_options(4,(char**)argv2,&got,error,sizeof error));
        assert(got.min_headroom_gib==12u);
    }
    /*
     * The request deadline is the real cap on prompt length, so it has to be
     * reachable from the command line. It was a hardcoded 600 until 2026-09-24,
     * which silently limited cold prompts to ~11K tokens.
     */
    {
        mimo26_server_options got; char error[256];
        memset(&got,0,sizeof got);
        server_defaults(&got);
        const char *argv[]={"x","root"};
        assert(mimo26_server_parse_options(2,(char**)argv,&got,error,sizeof error));
        assert(got.request_deadline_seconds==600u);
        const char *argv2[]={"x","root","--request-deadline-seconds","5400"};
        assert(mimo26_server_parse_options(4,(char**)argv2,&got,error,sizeof error));
        assert(got.request_deadline_seconds==5400u);
        /* Zero would mean a request that can never run; refuse it. */
        const char *argv3[]={"x","root","--request-deadline-seconds","0"};
        assert(!mimo26_server_parse_options(4,(char**)argv3,&got,error,sizeof error));
        const char *argv4[]={"x","root","--request-deadline-seconds","86401"};
        assert(!mimo26_server_parse_options(4,(char**)argv4,&got,error,sizeof error));
    }
    /* Grouped prefill follows lookahead by default, but the explicit
     * contradiction is refused rather than quietly downgraded. */
    CHECK(true,"root","--expert-major","on","--expert-lookahead","on");
    CHECK(true,"root","--expert-major","off");
    CHECK(false,"root","--expert-major","yes");
    CHECK(false,"root","--expert-major","");
    CHECK(true,"root","--expert-lookahead","off");
    CHECK(false,"root","--expert-major","on","--expert-lookahead","off");
    CHECK(true,"root","--expert-major","off","--expert-lookahead","off");
    {
        mimo26_server_options got; char error[256];
        memset(&got,0,sizeof got);
        server_defaults(&got);
        const char *d[]={"x","root"};
        assert(mimo26_server_parse_options(2,(char**)d,&got,error,sizeof error));
        assert(got.worker.expert_major && got.worker.expert_lookahead);
        assert(got.worker.expert_slots_per_layer==160u);
        assert(got.worker.global_kv_capacity==131072u);
        server_defaults(&got);
        const char *off[]={"x","root","--expert-lookahead","off"};
        assert(mimo26_server_parse_options(4,(char**)off,&got,error,sizeof error));
        assert(!got.worker.expert_major && !got.worker.expert_lookahead);
    }
    /*
     * Bearer credential and the bind policy around it, ported from K3's
     * server. The policy is the load-bearing part: a reachable bind with no
     * credential must be refused, not warned about.
     */
    CHECK(true,"root","--api-key","secret");
    assert(parsed.api_key && !strcmp(parsed.api_key,"secret"));
    CHECK(true,"root","--api-key","");          /* empty means no key */
    assert(parsed.api_key == NULL);
    CHECK(false,"root","--api-key");            /* value required */
    CHECK(false,"root","--api-key","a","--api-key","b");
    /* Loopback may run open; anything reachable may not. */
    CHECK(true,"root","--host","127.0.0.1");
    CHECK(true,"root","--host","127.0.0.5");
    CHECK(false,"root","--host","10.42.0.1");
    CHECK(true,"root","--host","10.42.0.1","--api-key","secret");
    CHECK(false,"root","--host","10.42.0.1","--api-key","");
    /* Per-request output ceiling. */
    CHECK(true,"root","--max-output-tokens","1");
    CHECK(true,"root","--max-output-tokens","65536");
    CHECK(false,"root","--max-output-tokens","0");
    CHECK(false,"root","--max-output-tokens","65537");
    CHECK(false,"root","--max-output-tokens","junk");
    {
        mimo26_server_options got; char error[256];
        memset(&got,0,sizeof got);
        server_defaults(&got);
        const char *d[]={"x","root"};
        assert(mimo26_server_parse_options(2,(char**)d,&got,error,sizeof error));
        assert(got.max_output_tokens==8192u);
        /* With no flag the environment supplies the key, so it need not
         * appear in a process listing. */
        setenv("MIMO26_API_KEY","fromenv",1);
        memset(&got,0,sizeof got);
        server_defaults(&got);
        const char *e[]={"x","root","--host","10.42.0.1"};
        assert(mimo26_server_parse_options(4,(char**)e,&got,error,sizeof error));
        assert(got.api_key && !strcmp(got.api_key,"fromenv"));
        unsetenv("MIMO26_API_KEY");
    }
    const char *bad[]={"-1","+1"," 16","16 ","16junk","","18446744073709551616","999999999999999999999999999"};
    for(unsigned i=0;i<sizeof bad/sizeof *bad;++i) { CHECK(false,"root","--slots",(char*)bad[i]); }
    printf("PASS %u strict server option cases; no default mutation on failure\n",checks);
}
