// Research-only MiMo codec gate, not linked into any server or K3 MZG2 ABI.
// Reuses Moonshine's CPU screen and local MZG2 wave ordering/checksum design.
// Preserves all 16 packed codes and all 256 scale bytes. See docs/provenance.md.
#include "mimo26_rans_screen.cpp"
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace {
constexpr uint32_t ExpertBytes=13369344, WeightBytes=4194304, ScaleBytes=262144;
constexpr uint32_t Bounds=1, State=2, Terminal=4, Checksum=8, Host=16;
struct Header { uint32_t magic, version, bytes, raw, tile, count, reserved[10]; };
struct Descriptor { uint32_t offset, bytes, payload, stored, kind, checksum, reserved[2]; };
struct Tables { uint32_t frequency[2][256], cumulative[2][256]; uint8_t lut[2][4096]; };
static_assert(sizeof(Header)==64 && sizeof(Descriptor)==32, "prototype metadata accounting");
__host__ __device__ uint32_t checkword(uint32_t word,uint32_t index) {
    uint32_t r=index&31u;
    return (r?((word<<r)|(word>>(32-r))):word) ^ (index*0x9e3779b9u);
}
__host__ __device__ uint32_t expected_kind(uint32_t offset,uint32_t layout) {
    return layout==2 ? (offset%(WeightBytes+ScaleBytes)>=WeightBytes) : layout;
}
__host__ __device__ bool valid_tile(uint32_t index, Descriptor d, uint32_t n,
                                   uint32_t tile, uint32_t block, uint32_t begin, uint32_t layout) {
    uint32_t offset=index*tile;
    return offset<n && d.offset==offset && d.bytes==(n-offset<tile?n-offset:tile) &&
        (d.kind&~3u)==0 && (d.kind&1u)==expected_kind(offset,layout) &&
        !d.reserved[0] && !d.reserved[1] && d.payload>=begin && !(d.payload&3u) &&
        d.payload<=block && d.stored<=block-d.payload &&
        ((d.kind&2u)?d.stored==d.bytes:(d.stored>=128 && !(d.stored&1u)));
}

__global__ void decode(const uint8_t* input,uint32_t block,uint32_t n,uint32_t tile,
                       uint32_t count,uint32_t begin,uint32_t layout,const Tables* tables,
                       uint8_t* output,uint32_t* errors) {
    __shared__ Tables table;
    for(uint32_t i=threadIdx.x;i<8192;i+=blockDim.x) table.lut[i/4096][i%4096]=tables->lut[i/4096][i%4096];
    for(uint32_t i=threadIdx.x;i<512;i+=blockDim.x) {
        table.frequency[i/256][i%256]=tables->frequency[i/256][i%256];
        table.cumulative[i/256][i%256]=tables->cumulative[i/256][i%256];
    }
    __syncthreads();
    uint32_t lane=threadIdx.x%32, index=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(index>=count)return;
    Descriptor d=reinterpret_cast<const Descriptor*>(input+sizeof(Header))[index];
    if(!valid_tile(index,d,n,tile,block,begin,layout)) {
        if(!lane)atomicOr(errors,Bounds);
        return;
    }
    const uint8_t* payload=input+d.payload;
    uint8_t* destination=output+d.offset;
    uint32_t checksum=0;
    if(d.kind&2u) {
        for(uint32_t at=lane*4;at<d.bytes;at+=128) {
            uint32_t word=*reinterpret_cast<const uint32_t*>(payload+at);
            *reinterpret_cast<uint32_t*>(destination+at)=word;
            checksum^=checkword(word,at/4);
        }
    } else {
        uint32_t state=reinterpret_cast<const uint32_t*>(payload)[lane], cursor=0, assembled=0;
        const uint16_t* words=reinterpret_cast<const uint16_t*>(payload+128);
        uint32_t word_count=(d.stored-128)/2, kind=d.kind&1u, perword=kind?4:8;
        bool valid=!__ballot(state<Lower);
        uint64_t lane_mask=lane?((uint64_t(1)<<lane)-1):0;
        for(uint32_t round=0;round<d.bytes/4*perword/32 && valid;++round) {
            uint32_t slot=state&4095u, symbol=table.lut[kind][slot];
            // Tables are host-validated before launch. Still fail closed on illegal symbols.
            if(__ballot(symbol>=(kind?256u:16u))) { valid=false; break; }
            state=table.frequency[kind][symbol]*(state>>12)+slot-table.cumulative[kind][symbol];
            bool need=state<Lower;
            uint64_t mask=__ballot(need);
            uint32_t needed=__popcll(mask);
            if(needed>word_count-cursor) { valid=false; break; }
            if(need)state=(state<<16)|words[cursor+__popcll(mask&lane_mask)];
            cursor+=needed;
            assembled|=symbol<<((round%perword)*(kind?8:4));
            if(round%perword==perword-1) {
                uint32_t at=(round/perword*32+lane)*4;
                *reinterpret_cast<uint32_t*>(destination+at)=assembled;
                checksum^=checkword(assembled,at/4); assembled=0;
            }
        }
        if(__ballot(!valid || cursor!=word_count || state!=Lower)) {
            if(!lane)atomicOr(errors,valid?Terminal:State);
            return;
        }
    }
    for(uint32_t delta=16;delta;delta>>=1)checksum^=__shfl_xor(checksum,delta,32);
    if(!lane && checksum!=d.checksum)atomicOr(errors,Checksum);
}

void require(bool good,const char* message) { if(!good)throw std::runtime_error(message); }
void hip(hipError_t error) { if(error!=hipSuccess)throw std::runtime_error(hipGetErrorString(error)); }
uint32_t align(uint32_t n,uint32_t a) { return (n+a-1)/a*a; }
std::vector<uint8_t> encode(const uint8_t* raw,uint32_t n,uint32_t tile,uint32_t layout,
                            const Model* models,uint64_t& escapes) {
    uint32_t count=(n+tile-1)/tile, begin=align(sizeof(Header)+count*sizeof(Descriptor),4096);
    std::vector<uint8_t> block(begin,0);
    std::vector<Descriptor> descriptors;
    for(uint32_t at=0;at<n;at+=tile) {
        uint32_t bytes=std::min(tile,n-at),kind=expected_kind(at,layout);
        const Model& m=models[kind];
        std::array<uint32_t,32> states; states.fill(Lower);
        std::vector<uint16_t> words;
        for(uint32_t r=bytes*(kind?1:2)/32;r-->0;)for(uint32_t lane=32;lane-->0;) {
            uint32_t s=value(raw+at,r,lane,kind==0), f=m.frequency[s], state=states[lane];
            if(uint64_t(state)>=(uint64_t(f)<<20)) { words.push_back(uint16_t(state)); state>>=16; }
            uint64_t next=(uint64_t(state/f)<<12)+state%f+m.cumulative[s];
            require(next<=UINT32_MAX,"encoder state overflow"); states[lane]=uint32_t(next);
        }
        std::reverse(words.begin(),words.end());
        uint32_t stored=128+2*uint32_t(words.size()), checksum=0;
        for(uint32_t i=0;i<bytes;i+=4) { uint32_t word; memcpy(&word,raw+at+i,4); checksum^=checkword(word,i/4); }
        Descriptor d{at,bytes,uint32_t(block.size()),stored,kind,checksum,{0,0}};
        if(stored>=bytes) { d.kind|=2; d.stored=bytes; ++escapes; }
        size_t offset=block.size(); block.resize(offset+align(d.stored,4),0);
        if(d.kind&2)memcpy(block.data()+offset,raw+at,bytes);
        else {
            memcpy(block.data()+offset,states.data(),128);
            if(!words.empty())memcpy(block.data()+offset+128,words.data(),words.size()*2);
        }
        descriptors.push_back(d);
    }
    block.resize(align(uint32_t(block.size()),4096),0);
    Header header{0x36524d4d,1,uint32_t(block.size()),n,tile,count,{}};
    memcpy(block.data(),&header,sizeof(header));
    memcpy(block.data()+sizeof(header),descriptors.data(),descriptors.size()*sizeof(Descriptor));
    return block;
}
bool validate(const std::vector<uint8_t>& block,uint32_t n,uint32_t tile,uint32_t layout) {
    if(block.size()<sizeof(Header))return false;
    Header h; memcpy(&h,block.data(),sizeof(h));
    if(h.magic!=0x36524d4d || h.version!=1 || h.bytes!=block.size() || h.raw!=n ||
       h.tile!=tile || h.count!=(n+tile-1)/tile)return false;
    for(uint32_t r:h.reserved)if(r)return false;
    uint32_t begin=align(sizeof(Header)+h.count*sizeof(Descriptor),4096),cursor=begin;
    if(begin>block.size())return false;
    for(uint32_t i=0;i<h.count;++i) {
        Descriptor d; memcpy(&d,block.data()+sizeof(Header)+i*sizeof(d),sizeof(d));
        if(!valid_tile(i,d,n,tile,h.bytes,begin,layout) || d.payload!=cursor)return false;
        cursor+=align(d.stored,4);
    }
    return align(cursor,4096)==h.bytes;
}
bool valid_tables(const Tables& t) {
    for(uint32_t k=0;k<2;++k) {
        uint32_t total=0;
        for(uint32_t s=0;s<(k?256u:16u);++s) {
            uint32_t f=t.frequency[k][s];
            if(!f || f>4096-total || t.cumulative[k][s]!=total)return false;
            for(uint32_t i=total;i<total+f;++i)if(t.lut[k][i]!=s)return false;
            total+=f;
        }
        if(total!=4096)return false;
    }
    return true;
}
struct Resources {
    uint8_t *mapped=nullptr,*mapped_device=nullptr,*storage=nullptr,*raw_host=nullptr;
    Tables* tables=nullptr; uint32_t* errors=nullptr;
    hipStream_t stream=nullptr; hipEvent_t first=nullptr,last=nullptr;
    ~Resources() {
        if(stream)(void)hipStreamSynchronize(stream);
        if(mapped)(void)hipHostFree(mapped); if(raw_host)(void)hipHostFree(raw_host);
        if(storage)(void)hipFree(storage); if(tables)(void)hipFree(tables); if(errors)(void)hipFree(errors);
        if(first)(void)hipEventDestroy(first); if(last)(void)hipEventDestroy(last); if(stream)(void)hipStreamDestroy(stream);
    }
};
double median(std::vector<double> values) { std::sort(values.begin(),values.end()); return values[values.size()/2]; }
}

struct MimoRansGpuReport {
    uint64_t raw_bytes, block_bytes, tiles, raw_tiles, fault_checks, host_faults, device_faults;
    double decode_ms, raw_copy_ms, checked_admission_ms;
    double decode_samples[5], copy_samples[5], admission_samples[5];
};
extern "C" int mimo26_rans_gpu_gate(const uint8_t* raw,uint32_t n,const uint64_t* packed_counts,
        const uint64_t* scale_counts,uint32_t tile,uint32_t waves,uint32_t layout,uint32_t faults,
        MimoRansGpuReport* result) {
    try {
        require(raw && packed_counts && scale_counts && result,"null input");
        require(n && n<=ExpertBytes && n%128==0 && layout<=2 && (layout!=2 || n==ExpertBytes),"input shape");
        require((tile==16384 || tile==32768 || tile==65536) && (waves==1 || waves==4 || waves==8),"launch shape");
        *result={}; Model models[2]; Tables tables{};
        require(model(packed_counts,models[0],16) && model(scale_counts,models[1],256),"model build");
        for(uint32_t k=0;k<2;++k) {
            for(uint32_t i=0;i<256;++i) { tables.frequency[k][i]=models[k].frequency[i]; tables.cumulative[k][i]=models[k].cumulative[i]; }
            memcpy(tables.lut[k],models[k].lut.data(),4096);
        }
        require(valid_tables(tables),"invalid tables");
        uint64_t escapes=0; auto block=encode(raw,n,tile,layout,models,escapes);
        require(validate(block,n,tile,layout),"encoded metadata invalid");
        Header h; memcpy(&h,block.data(),sizeof(h));
        uint32_t begin=align(sizeof(Header)+h.count*sizeof(Descriptor),4096);
        Resources r; hipDeviceProp_t properties{}; hip(hipGetDeviceProperties(&properties,0));
        require(properties.warpSize==32,"wave32 required");
        hip(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
        hip(hipEventCreate(&r.first)); hip(hipEventCreate(&r.last));
        hip(hipHostMalloc(&r.mapped,block.size(),hipHostMallocMapped));
        hip(hipHostGetDevicePointer(reinterpret_cast<void**>(&r.mapped_device),r.mapped,0));
        hip(hipHostMalloc(&r.raw_host,n,hipHostMallocMapped)); memcpy(r.raw_host,raw,n);
        hip(hipMalloc(&r.storage,n+256)); hip(hipMalloc(&r.tables,sizeof(tables))); hip(hipMalloc(&r.errors,4));
        hip(hipMemcpy(r.tables,&tables,sizeof(tables),hipMemcpyHostToDevice));
        std::vector<uint8_t> output(n+256); uint64_t published=0;
        auto run=[&](const std::vector<uint8_t>& candidate,bool bypass_metadata,bool publish,double* ms) {
            // Test-only admission seam: corrupted/partial scratch is never published.
            if(!bypass_metadata && !validate(candidate,n,tile,layout))return Host;
            require(candidate.size()==block.size(),"allocation size changed");
            memcpy(r.mapped,candidate.data(),candidate.size());
            hip(hipMemsetAsync(r.storage,0xa5,n+256,r.stream)); hip(hipMemsetAsync(r.errors,0,4,r.stream));
            hip(hipEventRecord(r.first,r.stream));
            decode<<<(h.count+waves-1)/waves,waves*32,0,r.stream>>>(r.mapped_device,h.bytes,n,tile,h.count,begin,layout,r.tables,r.storage+128,r.errors);
            hip(hipGetLastError()); hip(hipEventRecord(r.last,r.stream));
            uint32_t errors=0;
            hip(hipMemcpyAsync(&errors,r.errors,4,hipMemcpyDeviceToHost,r.stream)); hip(hipStreamSynchronize(r.stream));
            float elapsed=0; hip(hipEventElapsedTime(&elapsed,r.first,r.last)); if(ms)*ms=elapsed;
            hip(hipMemcpy(output.data(),r.storage,n+256,hipMemcpyDeviceToHost));
            for(uint32_t i=0;i<128;++i)require(output[i]==0xa5 && output[n+128+i]==0xa5,"output canary overwritten");
            if(!errors && publish)++published;
            return errors;
        };
        require(run(block,false,true,nullptr)==0 && published==1,"valid admission failed");
        require(!memcmp(output.data()+128,raw,n),"GPU byte mismatch");
        std::vector<double> decode_ms,copy_ms,admission_ms;
        for(uint32_t i=0;i<5;++i) {
            // Admission wall measurement below excludes the debug output readback/canary check.
            memcpy(r.mapped,block.data(),block.size());
            hip(hipMemsetAsync(r.errors,0,4,r.stream)); hip(hipStreamSynchronize(r.stream));
            auto start=std::chrono::steady_clock::now();
            require(validate(block,n,tile,layout),"timed metadata validation");
            hip(hipEventRecord(r.first,r.stream));
            decode<<<(h.count+waves-1)/waves,waves*32,0,r.stream>>>(r.mapped_device,h.bytes,n,tile,h.count,begin,layout,r.tables,r.storage+128,r.errors);
            hip(hipGetLastError()); hip(hipEventRecord(r.last,r.stream));
            uint32_t errors=0; hip(hipMemcpyAsync(&errors,r.errors,4,hipMemcpyDeviceToHost,r.stream));
            hip(hipStreamSynchronize(r.stream)); require(!errors,"timed decode error");
            admission_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            float ms=0; hip(hipEventElapsedTime(&ms,r.first,r.last)); decode_ms.push_back(ms);
            hip(hipEventRecord(r.first,r.stream));
            hip(hipMemcpyAsync(r.storage+128,r.raw_host,n,hipMemcpyHostToDevice,r.stream));
            hip(hipEventRecord(r.last,r.stream)); hip(hipStreamSynchronize(r.stream));
            hip(hipEventElapsedTime(&ms,r.first,r.last)); copy_ms.push_back(ms);
        }
        if(faults) {
            auto expect=[&](std::vector<uint8_t> bad,bool bypass,uint32_t wanted) {
                uint64_t before=published; uint32_t errors=run(bad,bypass,true,nullptr);
                require(errors && (!wanted || errors==wanted) && published==before,"fault accepted or wrong failure class");
                ++result->fault_checks; if(errors==Host)++result->host_faults; else ++result->device_faults;
            };
            for(uint32_t field=0;field<16;++field) {
                auto bad=block; reinterpret_cast<uint32_t*>(bad.data())[field]^=1; expect(bad,false,Host);
            }
            auto truncated=block; truncated.pop_back(); expect(truncated,false,Host);
            truncated.resize(63); expect(truncated,false,Host);
            const Descriptor original=*reinterpret_cast<const Descriptor*>(block.data()+64);
            for(uint32_t field: {0u,1u,2u,3u,4u,6u,7u}) {
                auto bad=block; auto* d=reinterpret_cast<Descriptor*>(bad.data()+64);
                if(field==0)d->offset=UINT32_MAX;
                if(field==1)d->bytes=UINT32_MAX;
                if(field==2)d->payload=h.bytes+4;
                if(field==3)d->stored=UINT32_MAX;
                if(field==4)d->kind|=4;
                if(field==6)d->reserved[0]=1;
                if(field==7)d->reserved[1]=1;
                expect(bad,false,Host); expect(bad,true,Bounds);
            }
            auto bad=block; reinterpret_cast<Descriptor*>(bad.data()+64)->checksum^=1;
            expect(bad,false,Checksum);
            bad=block; reinterpret_cast<Descriptor*>(bad.data()+64)->kind^=1;
            expect(bad,false,Host); expect(bad,true,Bounds);
            bad=block; reinterpret_cast<Descriptor*>(bad.data()+64)->payload=64;
            expect(bad,false,Host); expect(bad,true,Bounds);
            bad=block; reinterpret_cast<Descriptor*>(bad.data()+64)->payload+=1;
            expect(bad,false,Host); expect(bad,true,Bounds);
            if(original.kind&2u) {
                bad=block; bad[original.payload]^=1; expect(bad,false,Checksum);
            } else {
                bad=block; memset(bad.data()+original.payload,0,4); expect(bad,false,State);
                bad=block; reinterpret_cast<Descriptor*>(bad.data()+64)->stored=128;
                expect(bad,true,State); // actual held-out and compressible synthetic tiles need renormalization
                // An extra word must fail exact consumption even when decoded bytes are unchanged.
                if(original.stored+2<=h.bytes-original.payload) {
                    bad=block; reinterpret_cast<Descriptor*>(bad.data()+64)->stored+=2;
                    expect(bad,true,Terminal);
                }
                if(original.stored>128) {
                    bad=block; bad[original.payload+128+(original.stored-128)/2]^=1;
                    expect(bad,false,0); // corrupted renormalization data: any nonzero device error
                }
            }
            Tables bad_tables=tables; bad_tables.frequency[0][0]=0;
            require(!valid_tables(bad_tables),"bad model frequency accepted"); ++result->fault_checks; ++result->host_faults;
            bad_tables=tables; bad_tables.lut[1][0]^=1;
            require(!valid_tables(bad_tables),"bad model LUT accepted"); ++result->fault_checks; ++result->host_faults;
            require(run(block,false,true,nullptr)==0 && !memcmp(output.data()+128,raw,n),"post-fault recovery mismatch");
        }
        result->raw_bytes=n; result->block_bytes=block.size(); result->tiles=h.count; result->raw_tiles=escapes;
        result->decode_ms=median(decode_ms); result->raw_copy_ms=median(copy_ms); result->checked_admission_ms=median(admission_ms);
        for(uint32_t i=0;i<5;++i) {
            result->decode_samples[i]=decode_ms[i]; result->copy_samples[i]=copy_ms[i]; result->admission_samples[i]=admission_ms[i];
        }
        return 1;
    } catch(const std::exception& e) { fprintf(stderr,"mimo26_rans_gpu_gate: %s\n",e.what()); return 0; }
}
