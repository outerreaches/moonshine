#include "../mimo26_server_options.h"
#include <assert.h>
#include <stdlib.h>

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
    assert(mimo26_server_parse_options(argc,argv,&parsed,error,sizeof error)==expected);
    if (!expected) assert(error[0]&&!memcmp(&before,&parsed,sizeof before));
    else assert(parsed.worker.memory_limit_bytes==12345);
    ++checks;
}
#define CHECK(expected, ...) do { char *args[]={"server",__VA_ARGS__}; check(expected,(int)(sizeof args/sizeof *args),args); } while(0)
int main(void) {
    CHECK(true,"root");
    assert(!strcmp(parsed.host,"127.0.0.1")&&parsed.port==8640);
    assert(!parsed.worker.expert_lookahead&&parsed.worker.prefill_chunk==32);
    assert(parsed.worker.expert_slots_per_layer==16&&parsed.worker.global_kv_capacity==2048);
    CHECK(true,"root","--expert-lookahead","on","--prefill-chunk","64","--slots","48","--context","1024","--port","8765","--host","127.0.0.1");
    assert(parsed.worker.expert_lookahead&&parsed.worker.prefill_chunk==64&&parsed.worker.expert_slots_per_layer==48);
    assert(parsed.worker.global_kv_capacity==1024&&parsed.port==8765);
    CHECK(true,"root","--expert-lookahead","off","--prefill-chunk","0");
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
    const char *bad[]={"-1","+1"," 16","16 ","16junk","","18446744073709551616","999999999999999999999999999"};
    for(unsigned i=0;i<sizeof bad/sizeof *bad;++i) { CHECK(false,"root","--slots",(char*)bad[i]); }
    printf("PASS %u strict server option cases; no default mutation on failure\n",checks);
}
