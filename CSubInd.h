#include <vector>
#include <iostream>
#include <cstdio>

struct CSubInd {
    int dims;
    std::vector<int> sizes;
    std::vector<int> row_major_presum;
    std::vector<int> col_major_presum;

    void init(int dims_, int *sizes_) {
        dims = dims_;
        sizes.resize(dims_);

        for (int idx = 0; idx < dims; idx++) {
            sizes[idx] = sizes_[idx];
        }

        row_major_presum.resize(dims, 1);
        col_major_presum.resize(dims, 1);

        int base = 1;
        for (int idx = dims - 1; idx >= 0; idx--) {
            row_major_presum[idx] = base;
            base *= sizes[idx];
        }

        base = 1;
        for (int idx = 0; idx < dims; idx++) {
            col_major_presum[idx] = base;
            base *= sizes[idx];
        }
    }

    // convert index to row-major sub
    std::vector<int> ind2rowsub(int ind) {
        std::vector<int> sub(dims, 0);
        for (int jdx = dims - 1; jdx >= 0; jdx--) {
            sub[jdx] = ind / row_major_presum[jdx] % sizes[jdx];
            ind -= sub[jdx] * row_major_presum[jdx];
        }
        return sub;
    }

    // convert index to col-major sub
    std::vector<int> ind2colsub(int ind) {
        std::vector<int> sub(dims, 0);
        for (int jdx = 0; jdx < dims; jdx++) {
            sub[jdx] = ind / col_major_presum[jdx] % sizes[jdx];
            ind -= sub[jdx] * col_major_presum[jdx];
        }
        return sub;
    }

    // convert sub to row-major index
    int sub2rowind(std::vector<int> const &sub) {
        int ind = 0;
        for (int jdx = dims - 1; jdx >= 0; jdx--) {
            ind += sub[jdx] * row_major_presum[jdx];
        }
        return ind;
    }

    // convert sub to col-major index
    int sub2colind(std::vector<int> const &sub) {
        int ind = 0;
        for (int jdx = 0; jdx < dims; jdx++) {
            ind += sub[jdx] * col_major_presum [jdx];
        }
        return ind;
    }   
};

void test() {
    int dims = 4;
    int size [] = { 5, 4, 3, 2 };
    CSubInd si;
    si.init(dims, size);
    for (int idx = 0; idx < 5 * 4 * 3 * 2; idx++) {
        auto res1 = si.ind2rowsub(idx);
        std::cout << idx << "; ";
        for (auto v : res1) {
            std::cout << v << " ";
        }
        std::cout << "; ";

        auto res2 = si.ind2colsub(idx);
        
        for (auto v : res2) {
            std::cout << v << " ";
        }
        std::cout << "; ";
        

        if (idx == si.sub2rowind(res1) && idx == si.sub2colind(res2)) {
            std::cout << " PASS ";
        }
        std::cout << std::endl;
    }
}

int main() {
    test();
    return 0;
}