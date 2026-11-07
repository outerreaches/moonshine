/* Test-only failure after a real batch is submitted; never linked by Makefile. */
#include "../k3_io_uring.h"
#include <stdio.h>
#include <stdlib.h>
bool __real_k3_io_uring_submit(k3_io_uring*,const k3_io_request*,uint16_t,char*,size_t);
bool __wrap_k3_io_uring_submit(k3_io_uring*r,const k3_io_request*q,uint16_t n,char*e,size_t z) {
    static bool fired=false;
    bool ok=__real_k3_io_uring_submit(r,q,n,e,z);
    if(ok && !fired && getenv("MIMO26_TEST_HTTP_SUBMIT_FAILURE")) {
        fired=true;
        fprintf(stderr,"TEST_HTTP_FAULT pending=%u\n",k3_io_uring_outstanding(r));
        snprintf(e,z,"test-only failure after real submitted batch");
        return false;
    }
    return ok;
}
