#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstdio>

/*
matbin format:
TYPE DIM_NUM DIM1 DIM2 DIM3 ...
DATA
*/
bool write_matbin(const char *path, cv::Mat const &dst) {
    if (dst.empty()) return false;
    
    int elem_size = 1 << (dst.depth() / 2);

    std::ofstream fout(path, std::ofstream::out | std::ofstream::binary);
    if (!fout.fail()) {
        int type = dst.type();
        fout.write((char *) (&type), sizeof(int));

        int dims = dst.size.p[-1];
        fout.write((char *) (&dims), sizeof(int));

        size_t matsize = 1;
        for (int idx = 0; idx < dims; idx++) {
            int len = dst.size.p[idx];
            fout.write((char *) (&len), sizeof(int));
            matsize *= len;
        }

        int channels = dst.channels();
        fout.write((char *) dst.data, elem_size * matsize * channels);
        fout.close();
        return true;
    }
    else {
        return false;
    }
}

bool read_matbin(const char *path, cv::Mat &res) {
    std::ifstream fin(path, std::ofstream::in | std::ofstream::binary);
    if (fin.fail()) return false;

    int type, dims;
    fin.read((char *) &type, sizeof(int));
    fin.read((char *) &dims, sizeof(int));

    int depth = type & 7;
    int elem_size = 1 << (depth / 2);

    int *sizes = new int [dims];
    size_t matsize = 1;
    for (int idx = 0; idx < dims; idx++) {
        fin.read((char *) (sizes + idx), sizeof(int));
        matsize *= sizes[idx];
    }
    
    int channels = ((type & (511 << 3)) >> 3) + 1;

    res.create(dims, sizes, type);
    fin.read((char *) res.data, elem_size * matsize * channels);
    fin.close();
    delete []sizes; 

    return true;
}