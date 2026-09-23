// Reproduce current transport gaps in the actual parser; not a passing safety gate.
#include <initializer_list>
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>
#include <sys/wait.h>

int main() {
    const char* cases[]={
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2junk\r\n\r\n{}",
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 9\r\n\r\n{}",
        "POST /v1/chat/completions HTTP/1.1\r\nX-Content-Length: 2\r\n\r\n{}",
        "POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\n\r\n{}"
    };
    for(const char* wire:cases) {
        int pair[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
        assert(send(pair[1],wire,strlen(wire),0)==ssize_t(strlen(wire)));
        shutdown(pair[1],SHUT_WR);
        http_request request{};char error[256]{};
        bool accepted=read_request(pair[0],&request,error,sizeof(error));
        assert(accepted && request.body_size==2); // reproduces unsafe acceptance today
        request_free(&request);close(pair[0]);close(pair[1]);
    }
    int pair[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
    pid_t child=fork();assert(child>=0);
    if(!child) {
        close(pair[1]);http_request request{};char error[256];
        bool ok=read_request(pair[0],&request,error,sizeof(error));
        request_free(&request);close(pair[0]);_exit(ok?0:1);
    }
    close(pair[0]);int status=0;
    struct pollfd pause{-1,0,0};assert(poll(&pause,0,150)==0);
    assert(waitpid(child,&status,WNOHANG)==0);
    // Closing the owned peer unblocks the child without killing it.
    close(pair[1]);assert(waitpid(child,&status,0)==child);
    assert(WIFEXITED(status)&&WEXITSTATUS(status)==1);
    puts("REPRODUCED: four ambiguous/invalid framings accepted; empty peer blocks until closed. NOT a production safety pass.");
}
