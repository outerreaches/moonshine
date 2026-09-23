// Actual server transport, CPU-only. Short deadline only in this test binary.
#include <initializer_list>
#define MIMO26_HTTP_TIMEOUT_SECONDS .12
#define main mimo26_unused_server_main
#include "../mimo26_server.cu"
#undef main
#include <cassert>
#include <thread>
#include <string>
#include <sys/wait.h>
#include <random>

static void parse(const std::string& wire,int expected) {
    int pair[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
    assert(send(pair[1],wire.data(),wire.size(),MSG_NOSIGNAL)==(ssize_t)wire.size());
    shutdown(pair[1],SHUT_WR);
    http_request request{};char error[256];
    bool accepted=read_request(pair[0],&request,error,sizeof(error));
    if(expected>=0)assert(accepted==bool(expected));
    if(accepted) {
        assert(request.method && request.path && request.body);
        assert(request.body_size<=MAX_REQUEST_BYTES && request.body[request.body_size]=='\0');
    }
    request_free(&request);close(pair[0]);close(pair[1]);
}
int main() {
    parse("GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",true);
    parse("POST /v1/chat/completions HTTP/1.1\r\ncontent-length:\t2 \r\n\r\n{}",true);
    for(const char* headers:{"Content-Length: 2junk","Content-Length: +2","Content-Length: -2",
        "Content-Length:","Content-Length: 9999999999999999999999999",
        "Content-Length: 2\r\nContent-Length: 9","Content-Length: 2\r\nContent-Length: 2",
        "X-Content-Length: 2","Content-Length : 2"," Content-Length: 2",
        "Transfer-Encoding: chunked\r\nContent-Length: 2","Expect: 100-continue\r\nContent-Length: 2",
        "Content-Length: 2\nX: value"})
        parse(std::string("POST / HTTP/1.1\r\n")+headers+"\r\n\r\n{}",false);
    parse("POST / HTTP/1.1\r\nContent-Length: 3\r\n\r\n{}",false);
    parse("POST / HTTP/1.1\r\nContent-Length: 1\r\n\r\n{}",false);
    parse("GET / HTTP/2.0\r\n\r\n",false);
    parse("GET / HTTP/1.1\r\nBad Header: x\r\n\r\n",false);
    parse(std::string("GET / HTTP/1.1\r\nX: a")+std::string(1,'\0')+"b\r\n\r\n",false);
    parse("POST / HTTP/1.1\r\nContent-Length: 0\r\n\r\nGET / HTTP/1.1\r\n\r\n",false);
    parse("GET /health HTTP/1.0\r\n\r\n",true);
    parse(std::string("GET / HTTP/1.1\r\nX: ")+std::string(17000,'x')+"\r\n\r\n",false);
    // Deterministic parser memory-safety smoke, not exhaustive protocol fuzzing.
    std::mt19937 random(0x26f1a5u);
    const std::string seed="POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}";
    for(unsigned i=0;i<1000;++i) {
        std::string wire=seed;
        for(unsigned j=0;j<1+i%4;++j)wire[random()%wire.size()]=char(random()%256);
        if(i%3==0)wire.resize(random()%wire.size());
        parse(wire,-1);
    }
    for(bool body:{false,true}) {
        int pair[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
        if(body) {
            const char* h="POST / HTTP/1.1\r\nContent-Length: 20\r\n\r\n";
            assert(send(pair[1],h,strlen(h),0)==(ssize_t)strlen(h));
        }
        std::thread drip([&](){for(int i=0;i<8;++i){usleep(30000);(void)send(pair[1],"x",1,MSG_NOSIGNAL);}});
        double start=now_seconds();http_request request{};char error[256];
        assert(!read_request(pair[0],&request,error,sizeof(error)));
        assert(now_seconds()-start>=.10 && now_seconds()-start<.4);
        request_free(&request);drip.join();close(pair[0]);close(pair[1]);
    }
    int pair[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
    int small=4096;assert(!setsockopt(pair[0],SOL_SOCKET,SO_SNDBUF,&small,sizeof(small)));
    std::string large(4*1024*1024,'x');double start=now_seconds();
    assert(!send_all(pair[0],large.data(),large.size()));assert(now_seconds()-start<.4);
    g_shutdown=1;http_request request{};char error[256];start=now_seconds();
    assert(!read_request(pair[0],&request,error,sizeof(error)));
    assert(!send_all(pair[0],"x",1));assert(now_seconds()-start<.05);g_shutdown=0;
    close(pair[0]);close(pair[1]);
    assert(!socketpair(AF_UNIX,SOCK_STREAM,0,pair));
    pid_t child=fork();assert(child>=0);
    if(!child) {
        close(pair[1]);signal(SIGTERM,on_signal);
        http_request pending{};char message[256];
        bool accepted=read_request(pair[0],&pending,message,sizeof(message));
        request_free(&pending);close(pair[0]);_exit(!accepted && g_shutdown ? 0 : 1);
    }
    close(pair[0]);usleep(30000);assert(!kill(child,SIGTERM));
    int status=0;assert(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
    close(pair[1]);
    puts("PASS strict framing; 1000 deterministic parser mutations; absolute slow-header/body deadlines; blocked-send deadline; shutdown refusal");
}
