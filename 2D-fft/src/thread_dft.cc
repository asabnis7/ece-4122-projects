// ECE 4122
// 2D DFT using multithreading

#include "input_image.h"
#include "complex.h"
#include <cmath>
#include <chrono>
#include <thread>
#include <string>

#define threads 8 
#define PI 3.14159265358979323846f


///////////////////////////// Forward DFT functions ////////////////////////////
void row_dft(Complex* img, Complex * dftR, int w, int h, int row, int startR)
{
    for (int r = startR; r < startR + row; r++)
    {
        for (int n = 0; n < w; n++)
        {
            for (int k = 0; k < w; k++)
            {
                float arg = -2*M_PI*float(k*n)/float(w);
                Complex W(cos(arg), sin(arg));
                dftR[w*r+n] = dftR[w*r+n] + W*img[w*r+k];
            }
        }
    }
}

void column_dft(Complex* dftR, Complex * dft, int w, int h, int col, int startC)
{    
    for (int c = startC; c < startC + col; c++)
    {
        for (int n = 0; n < h; n++)
        {
            for (int k = 0; k < h; k++)
            {
                float arg = -2*M_PI*float(k*n)/float(h);
                Complex W(cos(arg), sin(arg));
                dft[c+n*w] = dft[c+n*w] + W*dftR[c+k*w];
            }
        }
    }
}
////////////////////////////// End Forward DFT /////////////////////////////////


/////////////////////////// Inverse DFT functions //////////////////////////////
void row_inv(Complex* dft, Complex * invR, int w, int h, int row, int startR)
{
    for (int r = startR; r < startR + row; r++)
    {
        for (int n = 0; n < w; n++)
        {
            for (int k = 0; k < w; k++)
            {
                float arg = 2*PI*float(k*n)/float(w);
                Complex W(cos(arg), sin(arg));
                invR[w*r+n] = invR[w*r+n] + W*dft[w*r+k];
            }
            invR[w*r+n] = invR[w*r+n]*(1.0/float(w));
        }
    }
}

void column_inv(Complex* invR, Complex * img, int w, int h, int col, int startC)
{    
    for (int c = startC; c < startC + col; c++)
    {
        for (int n = 0; n < h; n++)
        {
            for (int k = 0; k < h; k++)
            {
                float arg = 2*PI*float(k*n)/float(h);
                Complex W(cos(arg), sin(arg));
                img[c+n*w] = img[c+n*w] + W*invR[c+k*w];
            }
            img[c+n*w] = img[c+n*w]*(1.0/float(h));
        }
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
    
    // distribute rows/columns by thread
    int avgRow = height/threads;
    int leftoverRow = height%threads;
    int avgCol = width/threads;
    int leftoverCol = width%threads;
    int row, col;
    int startR = 0;
    int startC = 0;
    
    // thread array
    std::thread dftThread[threads];                     
   
    // Forward DFT
    if (argv[1] == std::string("forward"))
    {    
        Complex * dftR = new Complex[width * height];       
        Complex * dft = new Complex[width * height];
   
        for (int t = 0; t < threads; t++)
        {   
            (t < leftoverRow) ? row = avgRow+1 : row = avgRow;
            dftThread[t] = std::thread(row_dft, imgData, dftR, 
                                    width, height, row, startR);  
            startR += row;
        }
        for (int t = 0; t < threads; t++) dftThread[t].join(); 
        for (int t = 0; t < threads; t++)
        {      
            (t < leftoverCol) ? col = avgCol+1 : col = avgCol;
            dftThread[t] = std::thread(column_dft, dftR, dft, 
                                    width, height, col, startC);
            startC += col;
        }
        for (int t = 0; t < threads; t++) dftThread[t].join();  
        
        img.save_image_data(argv[3], dft, width, height);

    	delete[] dftR;
    	delete[] dft;

        auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Computation took " << elapsed.count() 
                  << " seconds." << std::endl;
    }

    // Inverse DFT - for cross-checking
    if (argv[1] == std::string("reverse"))
    {
        Complex * invR = new Complex[width * height];       
        Complex * invImg = new Complex[width * height];
  
        startR = 0;
        startC = 0;
        for (int t = 0; t < threads; t++)
        {   
            (t < leftoverRow) ? row = avgRow+1 : row = avgRow;
            dftThread[t] = std::thread(row_inv, imgData, invR, 
                                    width, height, row, startR);
            startR += row;
        }
        for (int t = 0; t < threads; t++) dftThread[t].join(); 
        for (int t = 0; t < threads; t++)
        {      
            (t < leftoverCol) ? col = avgCol+1 : col = avgCol;
            dftThread[t] = std::thread(column_inv, invR, invImg, 
                                    width, height, col, startC);
            startC += col;
        }
        for (int t = 0; t < threads; t++) dftThread[t].join(); 
 
        img.save_image_data_real(argv[3], invImg, width, height);
        
	    delete[] invR;
    	delete[] invImg;
        
	    auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Computation took " << elapsed.count() 
                  << " seconds." << std::endl;
    }
}
