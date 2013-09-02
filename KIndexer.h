#ifndef __KINDEXER__
#define __KINDEXER__

#include <cassert>

template<int cn>
class KIdx {
public:
    KIdx(int c0) {
        assert(cn == 1);
        dim[0] = c0;
    }

    KIdx(int c0, int c1) {
        assert(cn == 2);
        dim[0] = c0; dim[1] = c1;
    }

    KIdx(int c0, int c1, int c2) {
        assert(cn == 3);
        dim[0] = c0; dim[1] = c1; dim[2] = c2;
    }

    KIdx(int c0, int c1, int c2, int c3) {
        assert(cn == 4);
        dim[0] = c0; dim[1] = c1; dim[2] = c2; dim[3] = c3;
    }

    int operator()(int x) const {
        assert(cn == 1 && x < dim[0]);
        return x;
    }

    int operator()(int y, int x) const {
        assert(cn == 2 && y < dim[0] && x < dim[1]);
        return y * dim[1] + x;
    }

    int operator()(int z, int y, int x) const {
        assert(cn == 3 && z < dim[0] && y < dim[1] && x < dim[2]);
        return z * dim[2] * dim[1] + y * dim[2] + x;
    }

    int operator()(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return w * dim[3] * dim[2] * dim[1] + z * dim[3] * dim[2] + y * dim[3] + x;
    }

    int dim[cn];
};

template<typename _Tp, int cn>
class KIdx_ {
public:
    KIdx_(_Tp *data, int c0) {
        assert(cn == 1);
        p = data;
        sum = dim[0] = c0;
    }

    KIdx_(_Tp *data, int c0, int c1) {
        assert(cn == 2);
        p = data;
        dim[0] = c0; dim[1] = c1;
        sum = dim[0] * dim[1];
    }

    KIdx_(_Tp *data, int c0, int c1, int c2) {
        assert(cn == 3);
        p = data;
        dim[0] = c0; dim[1] = c1; dim[2] = c2;
        sum = dim[0] * dim[1] * dim[2];
    }

    KIdx_(_Tp *data, int c0, int c1, int c2, int c3) {
        assert(cn == 4);
        p = data;
        dim[0] = c0; dim[1] = c1; dim[2] = c2; dim[3] = c3;
        sum = dim[0] * dim[1] * dim[2] * dim[3];
    }

    _Tp &operator()(int x) const {
        assert(x < sum);
        return p[x];
    }

    _Tp &operator()(int y, int x) const {
        assert(cn == 2 && y < dim[0] && x < dim[1]);
        return p[y * dim[1] + x];
    }

    _Tp &operator()(int z, int y, int x) const {
        assert(cn == 3 && z < dim[0] && y < dim[1] && x < dim[2]);
        return p[z * dim[2] * dim[1] + y * dim[2] + x];
    }

    _Tp &operator()(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return p[w * dim[3] * dim[2] * dim[1] + z * dim[3] * dim[2] + y * dim[3] + x];
    }

    int index(int x) const {
        assert(cn == 1 && x < dim[0]);
        return x;
    }

    int index(int y, int x) const {
        assert(cn == 2 && y < dim[0] && x < dim[1]);
        return y * dim[1] + x;
    }

    int index(int z, int y, int x) const {
        assert(cn == 3 && z < dim[0] && y < dim[1] && x < dim[2]);
        return z * dim[2] * dim[1] + y * dim[2] + x;
    }

    int index(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return w * dim[3] * dim[2] * dim[1] + z * dim[3] * dim[2] + y * dim[3] + x;
    }

    void set_data(_Tp *data) {
        p = data;
    }

    long long sum;
    int dim[cn];
    _Tp *p;
};

#endif