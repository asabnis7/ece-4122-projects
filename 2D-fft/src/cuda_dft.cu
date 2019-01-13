// ECE 4122
// 2D DFT using CUDA

#include "input_image.cuh"
#include "complex.cuh"
#include <cmath>
#include <chrono>
#include <thread>
#include <string>
#include <cuda.h>

#define T_P_B 1024 

///////////////////////////// Forward DFT functions ////////////////////////////
__global__ void row_dft(Complex* img, Complex * dftR, int w, int h)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < w*h)
	{
		int n = idx % w;
        for (int k = 0; k < w; k++)
        {
			float arg = -2*PI*float(k*n)/float(w);
            Complex W(cos(arg), sin(arg));
            dftR[idx] = dftR[idx] + W*img[idx-n+k];
        }
	}
}

__global__ void column_dft(Complex* dftR, Complex * dft, int w, int h)
{    
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < w*h)
	{
		int n = (idx - (idx % w))/h;	        
	    for (int k = 0; k < h; k++)
	    {
	        float arg = -2*PI*float(k*n)/float(h);
	        Complex W(cos(arg), sin(arg));
	        dft[idx] = dft[idx] + W*dftR[idx+(k-n)*w];
	    }
	}
}
////////////////////////////// End Forward DFT /////////////////////////////////


/////////////////////////// Inverse DFT functions //////////////////////////////
__global__ void row_inv(Complex* dft, Complex * invR, int w, int h)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < w*h)
	{
		int n = idx % w;
        for (int k = 0; k < w; k++)
        {
			float arg = 2*PI*float(k*n)/float(w);
            Complex W(cos(arg), sin(arg));
            invR[idx] = invR[idx] + W*dft[idx-n+k];
        }
		invR[idx] = invR[idx]*(1.0/float(w));
	}
}

__global__ void column_inv(Complex* invR, Complex * img, int w, int h)
{    
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < w*h)
	{
		int n = (idx - (idx % w))/h;	        
	    for (int k = 0; k < h; k++)
	    {
	        float arg = 2*PI*float(k*n)/float(h);
	        Complex W(cos(arg), sin(arg));
	        img[idx] = img[idx] + W*invR[idx+(k-n)*w];
	    }
		img[idx] = img[idx]*(1.0/float(h));
	}
}
///////////////////////////// End Inverse DFT //////////////////////////////////


int main(int argc, char* argv[])
{
    auto startTime = std::chrono::system_clock::now();
    
    // read in image and dimensions
    int width, height, dim;                 
    
    InputImage img(argv[2]);                            
    width = img.get_width();
    height = img.get_height();
    dim = width*height;
    Complex * imgData = img.get_image_data();    
 
	int blockSize = ceil(float(dim)/float(T_P_B));
 
    // Forward DFT
    if (argv[1] == std::string("forward"))
    {    
        Complex * dft = new Complex[width * height];
 		Complex * d_dft;
		Complex * d_dftR; 
		Complex * d_imgData;

		cudaMalloc((void**)&d_imgData, dim*sizeof(Complex));
		cudaMalloc((void**)&d_dft, dim*sizeof(Complex));	
 		cudaMalloc((void**)&d_dftR, dim*sizeof(Complex));	
		
		cudaMemcpy(d_imgData, imgData, dim*sizeof(Complex), cudaMemcpyHostToDevice);		

		row_dft<<<blockSize, T_P_B>>>(d_imgData, d_dftR, width, height);
		cudaDeviceSynchronize();
		column_dft<<<blockSize, T_P_B>>>(d_dftR, d_dft, width, height);
		cudaDeviceSynchronize();		
	
		cudaMemcpy(dft, d_dft, dim*sizeof(Complex), cudaMemcpyDeviceToHost);
 
      	img.save_image_data(argv[3], dft, width, height);
        
		cudaFree(d_dft);
		cudaFree(d_dftR);
		cudaFree(d_imgData);
		delete[] dft;

		auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Computation took " << elapsed.count() 
                  << " seconds." << std::endl;
    }

    // Inverse DFT - for cross-checking
    if (argv[1] == std::string("reverse"))
    { 
        Complex * inv = new Complex[width * height];
 		Complex * d_inv; 
		Complex * d_invR; 
		Complex * d_imgData;

		cudaMalloc((Complex**)&d_imgData, dim*sizeof(Complex));
		cudaMalloc((Complex**)&d_inv, dim*sizeof(Complex));	
 		cudaMalloc((Complex**)&d_invR, dim*sizeof(Complex));	
		
		cudaMemcpy(d_imgData, imgData, dim*sizeof(Complex), cudaMemcpyHostToDevice);		

		row_inv<<<blockSize, T_P_B>>>(d_imgData, d_invR, width, height);
		cudaDeviceSynchronize();
		column_inv<<<blockSize, T_P_B>>>(d_invR, d_inv, width, height);
		cudaDeviceSynchronize();		
	
		cudaMemcpy(inv, d_inv, dim*sizeof(Complex), cudaMemcpyDeviceToHost);
  
        img.save_image_data_real(argv[3], inv, width, height);
        
		cudaFree(d_inv);
		cudaFree(d_invR);
		cudaFree(d_imgData);
		delete[] inv;

        auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Computation took " << elapsed.count() 
                  << " seconds." << std::endl;
    }
}
