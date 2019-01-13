// ECE 4122
// Input Image CUDA Class

#pragma once

#include "complex.cuh"

#include <fstream>
#include <sstream>
#include <iostream>

class InputImage {

public:
    __host__ InputImage(const char* filename) {
        std::ifstream ifs(filename);
        if(!ifs) {
            std::cout << "Can't open image file " << filename << std::endl;
            exit(1);
        }
    
        ifs >> w >> h;
        data = new Complex[w * h];
        for(int r = 0; r < h; ++r) {
            for(int c = 0; c < w; ++c) {
                float real;
                ifs >> real;
                data[r * w + c] = Complex(real);
            }
        }
    }
    
    __host__~InputImage(){
        delete[] data;
    }
    
    __host__ int get_width() const {
        return w;
    }
    
    __host__ int get_height() const {
        return h;
    }
    
    __host__ Complex* get_image_data() const {
        return data;
    }
    
    __host__ void save_image_data(const char *filename, Complex *d, int w, int h) {
        std::ofstream ofs(filename);
        if(!ofs) {
            std::cout << "Can't create output image " << filename << std::endl;
            return;
        }
    
        ofs << w << " " << h << std::endl;
    
        for(int r = 0; r < h; ++r) {
            for(int c = 0; c < w; ++c) {
                ofs << d[r * w + c] << " ";
            }
            ofs << std::endl;
        }
    }
    
    __host__ void save_image_data_real(const char* filename, Complex* d, int w, int h) {
        std::ofstream ofs(filename);
        if(!ofs) {
            std::cout << "Can't create output image " << filename << std::endl;
            return;
        }
    
        ofs << w << " " << h << std::endl;
    
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                ofs << d[r * w + c].real << " ";
            }
            ofs << std::endl;
        }
    }
    
    private:
        int w;
        int h;
        Complex* data;
};
