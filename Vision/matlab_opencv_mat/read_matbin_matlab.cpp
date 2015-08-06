#include "mex.h"
#include <string>
#include <vector>
#include <fstream>
using namespace std;

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
    std::vector<int> ind2sub_row(int ind) {
        std::vector<int> sub(dims, 0);
        for (int jdx = dims - 1; jdx >= 0; jdx--) {
            sub[jdx] = ind / row_major_presum[jdx] % sizes[jdx];
            ind -= sub[jdx] * row_major_presum[jdx];
        }
        return sub;
    }

    // convert index to col-major sub
    std::vector<int> ind2sub_col(int ind) {
        std::vector<int> sub(dims, 0);
        for (int jdx = 0; jdx < dims; jdx++) {
            sub[jdx] = ind / col_major_presum[jdx] % sizes[jdx];
            ind -= sub[jdx] * col_major_presum[jdx];
        }
        return sub;
    }

    // convert sub to row-major index
    int sub2ind_row(std::vector<int> const &sub) {
        int ind = 0;
        for (int jdx = dims - 1; jdx >= 0; jdx--) {
            ind += sub[jdx] * row_major_presum[jdx];
        }
        return ind;
    }

    // convert sub to col-major index
    int sub2ind_col(std::vector<int> const &sub) {
        int ind = 0;
        for (int jdx = 0; jdx < dims; jdx++) {
            ind += sub[jdx] * col_major_presum[jdx];
        }
        return ind;
    }
};

void read_matbin(const char *path, mxArray* &res) {
    mxClassID const matlab_class_mapper [] = { 
        mxUINT8_CLASS, mxINT8_CLASS, 
        mxUINT16_CLASS, mxINT16_CLASS,
        mxUINT32_CLASS, mxSINGLE_CLASS, 
        mxDOUBLE_CLASS
    };
    std::ifstream fin(path, std::ofstream::in | std::ofstream::binary);
    if (fin.fail()) {
        mexErrMsgIdAndTxt("MATLAB:read_matbin:invalidPath",
            "Cannot read matbin file");
    }

    int type, dims;
    fin.read((char *) &type, sizeof(int));
    fin.read((char *) &dims, sizeof(int));

    int depth = type & 7;
    int elem_size = 1 << (depth / 2);

    int *sizes = new int [dims + 1];
    size_t matsize = 1;
    for (int idx = 0; idx < dims; idx++) {
        fin.read((char *) (sizes + idx), sizeof(int));
        matsize *= sizes[idx];
    }
    
    int channels = ((type & (511 << 3)) >> 3) + 1;
    if (channels > 1) {
        sizes[dims] = channels;
        dims++;     
    }

    char *data = new char[elem_size * matsize * channels];
    fin.read(data, elem_size * matsize * channels);
    fin.close();

    CSubInd si;
    si.init(dims, &sizes[0]);

    res = mxCreateNumericArray(dims, &sizes[0], matlab_class_mapper[depth], mxREAL);

    char *dst_data = (char *)mxGetData(res);

    size_t count = matsize * channels;
    std::vector<int> sub(dims, 0);
    for (int idx = 0; idx < count; idx++) {
        std::vector<int> sub = si.ind2sub_row(idx);
        int dst_idx = si.sub2ind_col(sub) * elem_size;
        int src_idx = idx * elem_size;
        for (int jdx = 0; jdx < elem_size; jdx++) {
            dst_data[dst_idx + jdx] = data[src_idx + jdx];
        }
    }

    delete []data;
    delete []sizes;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if(1 != nrhs) {
        mexPrintf("Usage: read_matbin(\'XXXX.matbin\')");
        return;
    }
    
    if(!mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt( "MATLAB:read_matbin:invalidInputType",
                    "Input must be of type char.");
    }
    
    if(0 == nlhs) {
        return;
    }
    
    string matbin_path = mxArrayToString(prhs[0]);
    ifstream fin(matbin_path, std::fstream::in | std::fstream::binary);
    if(fin.fail()) {
        mexErrMsgIdAndTxt( "MATLAB:read_matbin:invalidPath",
                    "Cannot read matbin file");
    }
    
    read_matbin(matbin_path.c_str(), plhs[0]);
}

