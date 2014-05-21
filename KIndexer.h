#ifndef __KINDEXER__
#define __KINDEXER__

#include <cassert>

template<typename _Tp, int cn>
class KIdx_ {
public:
    KIdx_(_Tp *data, int c0) {
        assert(cn == 1);
        p = data;
        sum2 = sum3 = sum = dim[0] = c0;
    }

    KIdx_(_Tp *data, int h, int w) {
        assert(cn == 2);
        p = data;
        dim[0] = h; dim[1] = w;
        sum2 = sum3 = sum = dim[0] * dim[1];
    }

    KIdx_(_Tp *data, int h, int w, int c) {
        assert(cn == 3);
        p = data;
        dim[0] = h; dim[1] = w; dim[2] = c;
        sum3 = sum = dim[0] * dim[1] * dim[2];
        sum2 = dim[1] * dim[2];
    }

    KIdx_(_Tp *data, int c0, int c1, int c2, int c3) {
        assert(cn == 4);
        p = data;
        dim[0] = c0; dim[1] = c1; dim[2] = c2; dim[3] = c3;
        sum = dim[0] * dim[1] * dim[2] * dim[3];
        sum2 = dim[2] * dim[3];
        sum3 = dim[1] * dim[2] * dim[3];
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
        return p[z * sum2 + y * dim[2] + x];
    }

    _Tp &operator()(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return p[w * sum3 + z * sum2 + y * dim[3] + x];
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
        return z * sum2 + y * dim[2] + x;
    }

    int index(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return w * sum3 + z * sum2 + y * dim[3] + x;
    }

    void set_data(_Tp *data) {
        p = data;
    }

    long long sum, sum2, sum3;
    int dim[cn];
    _Tp *p;
};

template<typename _Tp, int cn>
class KIdx_Col {
public:
    KIdx_Col(_Tp *data, int c0) {
        assert(cn == 1);
        p = data;
        sum3 = sum2 = sum = dim[0] = c0;
    }

    KIdx_Col(_Tp *data, int h, int w) {
        assert(cn == 2);
        p = data;
        dim[0] = h; dim[1] = w;
        sum3 = sum2 = sum = dim[0] * dim[1];
    }

    KIdx_Col(_Tp *data, int h, int w, int c) {
        assert(cn == 3);
        p = data;
        dim[0] = h; dim[1] = w; dim[2] = c;
        sum3 = sum = dim[0] * dim[1] * dim[2];
        sum2 = dim[0] * dim[1];
    }

    KIdx_Col(_Tp *data, int c0, int c1, int c2, int c3) {
        assert(cn == 4);
        p = data;
        dim[0] = c0; dim[1] = c1; dim[2] = c2; dim[3] = c3;
        sum = dim[0] * dim[1] * dim[2] * dim[3];
        sum2 = dim[0] * dim[1];
        sum3 = sum2 * dim[2];
    }

    _Tp &operator()(int x) const {
        assert(x < sum);
        return p[x];
    }

    _Tp &operator()(int y, int x) const {
        assert(cn == 2 && y < dim[0] && x < dim[1]);
        return p[y + x * dim[0]];
    }

    _Tp &operator()(int z, int y, int x) const {
        assert(cn == 3 && z < dim[0] && y < dim[1] && x < dim[2]);
        return p[z + y * dim[0] + x * sum2];
    }

    _Tp &operator()(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return p[w + z * dim[0] + y * sum2 + z * sum3];
    }

    int index(int x) const {
        assert(cn == 1 && x < dim[0]);
        return x;
    }

    int index(int y, int x) const {
        assert(cn == 2 && y < dim[0] && x < dim[1]);
        return y + x * dim[0];
    }

    int index(int z, int y, int x) const {
        assert(cn == 3 && z < dim[0] && y < dim[1] && x < dim[2]);
        return z + y * dim[0] + x * sum2;
    }

    int index(int w, int z, int y, int x) const {
        assert(cn == 4 && w < dim[0] && z < dim[1] && y < dim[2] && x < dim[3]);
        return w + z * dim[0] + y * sum2 + z * sum3;
    }

    void set_data(_Tp *data) {
        p = data;
    }

    long long sum, sum2, sum3;
    int dim[cn];
    _Tp *p;
};

#endif
