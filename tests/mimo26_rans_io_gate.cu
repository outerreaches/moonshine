// Bounded SSD admission screen. Includes the exact prior test-only codec;
// no production store ABI, worker-cache integration, or arithmetic changes.
#include "mimo26_rans_gpu_gate.cu"
#include "../k3_io_uring.h"
#include "../k3_rocm_ops.h"
#include "../mimo26_rocm_ops.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fstream>
#include <random>
#include <string>

namespace {
constexpr unsigned Experts=6, ComputeCount=128;
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point from) {
    return std::chrono::duration<double,std::milli>(Clock::now()-from).count();
}
uint64_t read_bytes() {
    std::ifstream file("/proc/self/io"); std::string key; uint64_t value;
    while(file>>key>>value)if(key=="read_bytes:")return value;
    throw std::runtime_error("missing process I/O counter");
}
bool mapped_metadata(const uint8_t* block,uint32_t length,uint32_t tile) {
    if(length<64)return false;
    Header h; memcpy(&h,block,64);
    if(h.magic!=0x36524d4d || h.version!=1 || h.bytes!=length || h.raw!=ExpertBytes ||
       h.tile!=tile || h.count!=ExpertBytes/tile)return false;
    for(uint32_t v:h.reserved)if(v)return false;
    uint32_t begin=align(64+h.count*sizeof(Descriptor),4096),cursor=begin;
    if(begin>length)return false;
    for(uint32_t i=0;i<h.count;++i) {
        Descriptor d; memcpy(&d,block+64+i*sizeof(d),sizeof(d));
        if(!valid_tile(i,d,ExpertBytes,tile,length,begin,2) || d.payload!=cursor)return false;
        cursor+=align(d.stored,4);
    }
    return align(cursor,4096)==length;
}
struct IoResources {
    k3_io_uring* ring=nullptr;
    uint8_t* host[Experts]{}, *device_alias[Experts]{}, *dest=nullptr, *weights=nullptr;
    uint16_t *input=nullptr,*gate=nullptr,*up=nullptr,*active=nullptr,*expert_out=nullptr,*final=nullptr;
    float* accumulator=nullptr;
    Tables* tables=nullptr; uint32_t* errors=nullptr;
    hipStream_t stream=nullptr,compute=nullptr;
    hipEvent_t first=nullptr,last=nullptr;
    std::vector<int> fds;
    ~IoResources() {
        // A failure may leave submissions outstanding. Destroy the owner first.
        if(ring)k3_io_uring_destroy(ring);
        if(stream)(void)hipStreamSynchronize(stream);
        if(compute)(void)hipStreamSynchronize(compute);
        for(auto* p:host)if(p)(void)hipHostFree(p);
        for(void* p:std::vector<void*>{dest,weights,input,gate,up,active,expert_out,final,accumulator,tables,errors})
            if(p)(void)hipFree(p);
        if(first)(void)hipEventDestroy(first); if(last)(void)hipEventDestroy(last);
        if(stream)(void)hipStreamDestroy(stream); if(compute)(void)hipStreamDestroy(compute);
        for(int fd:fds)(void)close(fd);
    }
    int open_read(const std::string& path) {
        int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_DIRECT);
        require(fd>=0,"O_DIRECT open failed"); fds.push_back(fd); return fd;
    }
};
struct Mode {
    const char* name; uint32_t tile,waves;
    std::array<int,Experts> fd{};
    std::array<uint64_t,Experts> offset{};
    std::array<uint32_t,Experts> bytes{},skip{};
};
void write_fixture(const std::string& path,const std::vector<uint8_t>& bytes) {
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    require(fd>=0,"fixture must be a fresh file");
    size_t done=0;
    while(done<bytes.size()) {
        ssize_t n=write(fd,bytes.data()+done,bytes.size()-done);
        if(n<0 && errno==EINTR)continue;
        if(n<=0) { (void)close(fd); throw std::runtime_error("fixture write failed"); }
        done+=size_t(n);
    }
    int synced=fsync(fd),closed=close(fd); require(!synced && !closed,"fixture sync/close failed");
}
void launch_compute(IoResources& r) {
    hip(hipEventRecord(r.first,r.compute));
    require(mimo26_rocm_zero_f32(r.accumulator,4096,r.compute),"compute accumulator init");
    for(unsigned i=0;i<ComputeCount;++i) {
        const uint8_t* w=r.weights+(i%Experts)*ExpertBytes;
        require(k3_rocm_mxfp4_gemv_bf16(r.gate,w+4456448,w+8650752,r.input,2048,4096,r.compute),"compute gate");
        require(k3_rocm_mxfp4_gemv_bf16(r.up,w+8912896,w+13107200,r.input,2048,4096,r.compute),"compute up");
        require(mimo26_rocm_silu_product_bf16(r.active,r.gate,r.up,2048,r.compute),"compute silu");
        require(k3_rocm_mxfp4_gemv_bf16(r.expert_out,w,w+4194304,r.active,4096,2048,r.compute),"compute down");
        require(mimo26_rocm_expert_accumulate_f32(r.accumulator,r.expert_out,.125f,4096,r.compute),"compute accumulate");
    }
    require(mimo26_rocm_expert_finalize_bf16(r.final,r.accumulator,4096,r.compute),"compute finalize");
    hip(hipEventRecord(r.last,r.compute));
}
}

extern "C" int mimo26_rans_io_gate(const uint8_t* raw,const uint64_t* packed_counts,
        const uint64_t* scale_counts,const char* directory,const char* const* sources,
        const uint64_t* source_offsets,const char* log_path,uint32_t rounds) {
    try {
        require(raw && packed_counts && scale_counts && directory && sources && source_offsets && log_path,"null input");
        require(rounds>=2 && rounds<=16,"round budget");
        IoResources r; Model models[2]; Tables tables{};
        require(model(packed_counts,models[0],16) && model(scale_counts,models[1],256),"model build");
        for(unsigned k=0;k<2;++k) {
            for(unsigned i=0;i<256;++i) { tables.frequency[k][i]=models[k].frequency[i]; tables.cumulative[k][i]=models[k].cumulative[i]; }
            memcpy(tables.lut[k],models[k].lut.data(),4096);
        }
        require(valid_tables(tables),"table validation");
        std::vector<Mode> modes;
        Mode original{"original",0,0};
        for(unsigned e=0;e<Experts;++e) {
            original.fd[e]=r.open_read(sources[e]);
            original.offset[e]=source_offsets[e]&~uint64_t(4095);
            original.skip[e]=uint32_t(source_offsets[e]-original.offset[e]);
            original.bytes[e]=align(original.skip[e]+ExpertBytes,4096);
        }
        modes.push_back(original);
        std::string root(directory);
        Mode snapshot{"raw_fixture",0,0};
        write_fixture(root+"/raw.bin",std::vector<uint8_t>(raw,raw+Experts*ExpertBytes));
        int raw_fd=r.open_read(root+"/raw.bin");
        for(unsigned e=0;e<Experts;++e) {
            snapshot.fd[e]=raw_fd; snapshot.offset[e]=uint64_t(e)*ExpertBytes; snapshot.bytes[e]=ExpertBytes;
        }
        modes.push_back(snapshot);
        for(uint32_t tile:{16384u,65536u}) {
            Mode compressed{tile==16384?"rans16_w4":"rans64_w1",tile,tile==16384?4u:1u};
            std::vector<uint8_t> payload;
            for(unsigned e=0;e<Experts;++e) {
                uint64_t escapes=0;
                auto block=encode(raw+e*ExpertBytes,ExpertBytes,tile,2,models,escapes);
                require(validate(block,ExpertBytes,tile,2) && mapped_metadata(block.data(),block.size(),tile),"fixture validation");
                compressed.offset[e]=payload.size(); compressed.bytes[e]=block.size();
                payload.insert(payload.end(),block.begin(),block.end());
            }
            std::string path=root+(tile==16384?"/rans16.bin":"/rans64.bin");
            write_fixture(path,payload); int fd=r.open_read(path);
            compressed.fd.fill(fd); modes.push_back(compressed);
            if(tile==16384) { compressed.name="rans16_w8"; compressed.waves=8; modes.push_back(compressed); }
        }
        hipDeviceProp_t props{}; hip(hipGetDeviceProperties(&props,0)); require(props.warpSize==32,"wave32 required");
        hip(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
        hip(hipStreamCreateWithFlags(&r.compute,hipStreamNonBlocking));
        hip(hipEventCreate(&r.first)); hip(hipEventCreate(&r.last));
        iovec buffers[Experts];
        for(unsigned i=0;i<Experts;++i) {
            hip(hipHostMalloc(&r.host[i],ExpertBytes+4096,hipHostMallocMapped));
            hip(hipHostGetDevicePointer(reinterpret_cast<void**>(&r.device_alias[i]),r.host[i],0));
            require(uintptr_t(r.host[i])%4096==0,"staging not aligned");
            buffers[i]={r.host[i],ExpertBytes+4096};
        }
        char error[512]{};
        require(k3_io_uring_create(&r.ring,buffers,Experts,error,sizeof(error)),error);
        constexpr uint32_t Stride=ExpertBytes+256;
        hip(hipMalloc(&r.dest,Experts*Stride)); hip(hipMalloc(&r.weights,Experts*ExpertBytes));
        hip(hipMemcpy(r.weights,raw,Experts*ExpertBytes,hipMemcpyHostToDevice));
        hip(hipMalloc(&r.tables,sizeof(tables))); hip(hipMemcpy(r.tables,&tables,sizeof(tables),hipMemcpyHostToDevice));
        hip(hipMalloc(&r.errors,4)); hip(hipMalloc(&r.input,4096*2));
        hip(hipMalloc(&r.gate,2048*2)); hip(hipMalloc(&r.up,2048*2)); hip(hipMalloc(&r.active,2048*2));
        hip(hipMalloc(&r.expert_out,4096*2)); hip(hipMalloc(&r.final,4096*2)); hip(hipMalloc(&r.accumulator,4096*4));
        std::vector<uint16_t> input(4096),compute_reference(4096),compute_result(4096);
        for(unsigned i=0;i<4096;++i)input[i]=uint16_t(0x3c00+(i%64)*2+(i%2?0x8000:0));
        hip(hipMemcpy(r.input,input.data(),8192,hipMemcpyHostToDevice));
        launch_compute(r); hip(hipStreamSynchronize(r.compute));
        hip(hipMemcpy(compute_reference.data(),r.final,8192,hipMemcpyDeviceToHost));
        for(auto x:compute_reference)require((x&0x7f80)!=0x7f80,"nonfinite compute reference");
        std::vector<uint8_t> recovered(Experts*Stride);
        std::ofstream log(log_path,std::ios::out|std::ios::app); require(bool(log),"open trial log");
        std::mt19937 random(0x26f1a5u);
        std::array<unsigned,Experts> identities{0,1,2,3,4,5};
        // Warm-up round is recorded but excluded by the analyzer.
        for(uint32_t round=0;round<=rounds;++round)for(unsigned contention:{0u,1u})for(unsigned qd:{1u,2u,6u}) {
            std::shuffle(identities.begin(),identities.end(),random);
            std::vector<unsigned> order{0,1,2,3,4}; std::shuffle(order.begin(),order.end(),random);
            for(unsigned mode_index:order) {
                const Mode& mode=modes[mode_index];
                require(k3_io_uring_outstanding(r.ring)==0,"leftover reads");
                hip(hipMemset(r.dest,0xa5,Experts*Stride));
                uint64_t before=read_bytes(),requested=0; auto start=Clock::now();
                if(contention)launch_compute(r);
                double queue_ms=milliseconds(start); auto admission_start=Clock::now();
                for(unsigned first=0;first<Experts;first+=qd) {
                    k3_io_request requests[Experts]{};
                    for(unsigned i=0;i<qd;++i) {
                        unsigned e=identities[first+i];
                        requests[i]={mode.fd[e],mode.offset[e],mode.bytes[e],uint16_t(i),i};
                        requested+=mode.bytes[e];
                    }
                    require(k3_io_uring_submit(r.ring,requests,qd,error,sizeof(error)),error);
                    unsigned completed=0; bool seen[Experts]{};
                    while(completed<qd) {
                        k3_io_completion completions[Experts]{}; uint16_t got=0;
                        require(k3_io_uring_wait(r.ring,completions,Experts,&got,error,sizeof(error)),error);
                        require(got>0 && got<=qd-completed,"completion count");
                        for(unsigned c=0;c<got;++c) {
                            const auto& done=completions[c]; unsigned i=done.user_data;
                            require(done.user_data<qd && done.buffer_index==i && !seen[i],"completion identity"); seen[i]=true;
                            unsigned e=identities[first+i];
                            require(done.result==int32_t(mode.bytes[e]),"short/error direct read");
                            uint8_t* dest=r.dest+e*Stride+128;
                            if(!mode.tile) {
                                hip(hipMemcpyAsync(dest,r.host[i]+mode.skip[e],ExpertBytes,hipMemcpyHostToDevice,r.stream));
                            } else {
                                require(mapped_metadata(r.host[i],mode.bytes[e],mode.tile),"read metadata validation");
                                hip(hipMemsetAsync(r.errors,0,4,r.stream));
                                uint32_t count=ExpertBytes/mode.tile,begin=align(64+count*32,4096);
                                decode<<<(count+mode.waves-1)/mode.waves,mode.waves*32,0,r.stream>>>(
                                    r.device_alias[i],mode.bytes[e],ExpertBytes,mode.tile,count,begin,2,r.tables,dest,r.errors);
                                hip(hipGetLastError());
                                uint32_t flags=0; hip(hipMemcpyAsync(&flags,r.errors,4,hipMemcpyDeviceToHost,r.stream));
                                hip(hipStreamSynchronize(r.stream)); require(!flags,"decoded integrity failure");
                            }
                            hip(hipStreamSynchronize(r.stream)); ++completed;
                        }
                    }
                }
                double admission_ms=milliseconds(admission_start);
                bool compute_busy=false; float compute_ms=0;
                if(contention) {
                    hipError_t state=hipEventQuery(r.last);
                    require(state==hipSuccess || state==hipErrorNotReady,"compute event query");
                    compute_busy=state==hipErrorNotReady;
                    hip(hipStreamSynchronize(r.compute)); hip(hipEventElapsedTime(&compute_ms,r.first,r.last));
                }
                double makespan_ms=milliseconds(start); uint64_t physical=read_bytes()-before;
                require(physical>=requested,"direct I/O accounting below request bytes");
                require(k3_io_uring_outstanding(r.ring)==0,"undrained reads");
                hip(hipMemcpy(recovered.data(),r.dest,Experts*Stride,hipMemcpyDeviceToHost));
                for(unsigned e=0;e<Experts;++e) {
                    const uint8_t* p=recovered.data()+e*Stride;
                    for(unsigned j=0;j<128;++j)require(p[j]==0xa5 && p[128+ExpertBytes+j]==0xa5,"admission canary");
                    require(!memcmp(p+128,raw+e*ExpertBytes,ExpertBytes),"admitted expert mismatch");
                }
                if(contention) {
                    hip(hipMemcpy(compute_result.data(),r.final,8192,hipMemcpyDeviceToHost));
                    require(compute_result==compute_reference,"concurrent compute mismatch");
                }
                log<<"{\"round\":"<<round<<",\"mode\":\""<<mode.name<<"\",\"qd\":"<<qd
                   <<",\"contention\":"<<contention<<",\"compute_experts\":"<<(contention?ComputeCount:0)
                   <<",\"admission_ms\":"<<admission_ms<<",\"queue_ms\":"<<queue_ms<<",\"compute_ms\":"<<compute_ms
                   <<",\"makespan_ms\":"<<makespan_ms<<",\"compute_busy_at_admission_end\":"<<(compute_busy?"true":"false")
                   <<",\"requested_bytes\":"<<requested<<",\"read_bytes\":"<<physical
                   <<",\"exact_bytes\":"<<Experts*ExpertBytes<<",\"expert_order\":[";
                for(unsigned i=0;i<Experts;++i)log<<(i?",":"")<<identities[i];
                log<<"]}"<<std::endl; require(bool(log),"write trial log");
            }
        }
        return 1;
    } catch(const std::exception& e) { fprintf(stderr,"mimo26_rans_io_gate: %s\n",e.what()); return 0; }
}
