// MiMo research screen adapted from the local GLM byte-rANS screen and MZG2.
// Full 16-code packed alphabet + 256 scale bytes; NEVER canonicalizes zeros.
// Not compatible with the K3 MZG2 ABI. Sizes returned only after exact recovery.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr uint32_t M=4096, Lower=65536, Lanes=32;
struct Model {
    std::array<uint32_t,256> frequency{}, cumulative{};
    std::array<uint8_t,M> lut{};
};
bool model(const uint64_t*counts,Model&m,unsigned alphabet){
    uint64_t total=0;
    for(unsigned s=0;s<alphabet;++s){
        if(counts[s]>UINT64_MAX-total)return false;
        total+=counts[s];
    }
    if(!total||total>UINT64_MAX/M)return false;
    unsigned sum=0;
    // Give every byte a nonzero frequency, including unseen held-out symbols.
    for(unsigned s=0;s<alphabet;++s){
        m.frequency[s]=std::max(uint32_t(1),uint32_t(counts[s]*M/total));
        sum+=m.frequency[s];
    }
    while(sum!=M){
        unsigned best=alphabet;
        long double score=-1e100L;
        for(unsigned s=0;s<alphabet;++s){
            if(sum>M&&m.frequency[s]<=1)continue;
            long double residual=(long double)counts[s]*M/total-m.frequency[s];
            if(sum>M)residual=-residual;
            if(residual>score){score=residual;best=s;}
        }
        if(best==alphabet)return false;
        if(sum<M){++m.frequency[best];++sum;}
        else{--m.frequency[best];--sum;}
    }
    unsigned at=0;
    for(unsigned s=0;s<alphabet;++s){
        m.cumulative[s]=at;
        for(unsigned i=0;i<m.frequency[s];++i)m.lut[at++]=uint8_t(s);
    }
    return at==M;
}
size_t byte_index(unsigned round,unsigned lane){return (size_t(round/4)*32+lane)*4+round%4;}
unsigned value(const uint8_t*data,unsigned round,unsigned lane,bool packed){
    if(!packed)return data[byte_index(round,lane)];
    return (data[(size_t(round/8)*32+lane)*4+(round%8)/2]>>((round&1)*4))&15;
}
bool tile(const uint8_t*data,unsigned n,const Model&m,bool packed,uint64_t&stored,uint64_t&escapes){
    std::array<uint32_t,Lanes> states;
    states.fill(Lower);
    std::vector<uint16_t> words;
    words.reserve(n/2);
    for(unsigned r=n*(packed?2:1)/32;r-->0;){
        for(unsigned lane=32;lane-->0;){
            unsigned symbol=value(data,r,lane,packed);
            uint32_t f=m.frequency[symbol],state=states[lane];
            if(uint64_t(state)>=(uint64_t(f)<<20)){
                words.push_back(uint16_t(state));state>>=16;
            }
            uint64_t next=(uint64_t(state/f)<<12)+state%f+m.cumulative[symbol];
            if(next>UINT32_MAX)return false;
            states[lane]=uint32_t(next);
        }
    }
    std::reverse(words.begin(),words.end());
    // Decode even expanded candidates; raw escape is accounted separately.
    size_t cursor=0;
    for(unsigned r=0;r<n*(packed?2:1)/32;++r){
        for(unsigned lane=0;lane<32;++lane){
            uint32_t state=states[lane],slot=state&(M-1);
            if(state<Lower)return false;
            unsigned symbol=m.lut[slot];
            state=m.frequency[symbol]*(state>>12)+slot-m.cumulative[symbol];
            if(state<Lower){
                if(cursor==words.size())return false;
                state=(state<<16)|words[cursor++];
            }
            states[lane]=state;
            if(value(data,r,lane,packed)!=symbol)return false;
        }
    }
    if(cursor!=words.size())return false;
    for(auto state:states)if(state!=Lower)return false;
    uint64_t encoded=128+2*words.size();
    if(encoded>=n){encoded=n;++escapes;}
    stored+=(encoded+3)&~uint64_t(3);
    return true;
}
}

extern "C" int mimo26_rans_screen(const uint8_t*data,size_t n,
        const uint64_t*counts,unsigned tile_bytes,unsigned packed,uint64_t*payload,uint64_t*escapes){
    if(packed>1||!data||!counts||!payload||!escapes||!n||n%128||
       (tile_bytes!=16384&&tile_bytes!=32768&&tile_bytes!=65536))return 0;
    try{
        Model m;
        if(!model(counts,m,packed?16:256))return 0;
        uint64_t bytes=0,raw=0;
        for(size_t at=0;at<n;at+=tile_bytes){
            unsigned length=unsigned(std::min(size_t(tile_bytes),n-at));
            if(!tile(data+at,length,m,packed!=0,bytes,raw))return 0;
        }
        *payload=bytes;*escapes=raw;return 1;
    }catch(...){return 0;}
}
