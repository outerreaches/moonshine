#include "mimo26_rans_screen.cpp"
#include <cassert>
#include <cstdio>
int main(){
    std::array<uint64_t,256>counts{};counts.fill(1);uint64_t bytes=17,raw=23;
    std::vector<uint8_t>data(65536);for(size_t i=0;i<data.size();++i)data[i]=uint8_t(i);
    for(unsigned packed:{0u,1u})for(unsigned size:{16384u,32768u,65536u}){
        assert(mimo26_rans_screen(data.data(),data.size(),counts.data(),size,packed,&bytes,&raw));
        assert(raw==data.size()/size&&bytes==data.size());
    }
    counts.fill(0);counts[8]=1000000;
    for(unsigned packed:{0u,1u})for(uint8_t code:{uint8_t(0),uint8_t(8),uint8_t(128),uint8_t(136)}){
        std::fill(data.begin(),data.end(),code);
        assert(mimo26_rans_screen(data.data(),data.size(),counts.data(),16384,packed,&bytes,&raw));
    }
    assert(mimo26_rans_screen(data.data(),128,counts.data(),16384,1,&bytes,&raw));
    assert(!mimo26_rans_screen(data.data(),127,counts.data(),16384,1,&bytes,&raw));
    assert(!mimo26_rans_screen(data.data(),128,counts.data(),3,1,&bytes,&raw));
    assert(!mimo26_rans_screen(data.data(),128,counts.data(),16384,2,&bytes,&raw));
    assert(!mimo26_rans_screen(nullptr,128,counts.data(),16384,1,&bytes,&raw));
    counts.fill(0);assert(!mimo26_rans_screen(data.data(),128,counts.data(),16384,1,&bytes,&raw));
    counts[0]=UINT64_MAX;counts[1]=1;
    assert(!mimo26_rans_screen(data.data(),128,counts.data(),16384,1,&bytes,&raw));
    puts("PASS full alphabets, raw fallback, signed zeros, unseen symbols, partial tile, geometry and count overflow");
}
