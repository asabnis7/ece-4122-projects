// Arjun Sabnis
// ECE 4122
// 2D FFT using MPI

#include "input_image.h"
#include "complex.h"
#include "mpi.h"
#include <cmath>
#include <chrono>
#include <thread>
#include <string>

#define startTag    1
#define endTag      2

#define PI 3.14159265358979323846f

///////////////////////////// Forward FFT functions ////////////////////////////

Complex * slice(Complex * data, int start, int size, int stride)
{	
	Complex * temp = new Complex[size/2];
	for (int i = start; i < size; i += stride)
	{
		temp[i/2] = data[i];
	}
	return temp;
} 	

void row_fft(Complex* img, int w, int startR)
{
    int N = w/2;
    if (w <= 1) return;
    
    // copy row for easier indexing
    Complex * rowarr = new Complex[w];
    std::copy(img, img+w, rowarr);

    // divide and recursively call FFT
    Complex * even = slice(rowarr, 0, w, 2);
	Complex * odd = slice(rowarr, 1, w, 2);
	row_fft(even,  N, 0);
	row_fft(odd, N, 0);
	
    // calculate FFT in-place and assemble
	for (int k = 0; k < N; k++)
	{
		Complex W(cos(-2*PI*float(k)/float(w)), sin(-2*PI*float(k)/float(w)));
		img[k] = even[k] + W*odd[k];
		img[k+N] = even[k] - W*odd[k];
	}
}

void transpose(Complex* data, int w)
{
    for (int i = 0; i < w; i++)
        for (int j = i+1; j < w; j++)
            std::swap(data[j*w+i], data[i*w+j]);
}

////////////////////////////// End FFT Function ////////////////////////////////


int main(int argc, char* argv[])
{
    auto startTime = std::chrono::system_clock::now();
    
    // read in image and dimensions
    InputImage img(argv[2]);                            
    int width = img.get_width();
    int height = img.get_height();
    int dim = width*height;
    Complex * imgData = img.get_image_data();
   
    // MPI variables
    int rank, numWorkers, numTasks, conf;
 
    conf = MPI_Init(&argc, &argv);
    if (conf != MPI_SUCCESS)
    {
        std::cout << "Error starting MPI program" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, conf);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // MPI Typedef
    MPI_Datatype complextype;
    MPI_Datatype oldtype[1]; 
	oldtype[0] = MPI_FLOAT;
    int blockcount[1];
	blockcount[0] = 2;
    MPI_Aint offset[1];
	offset[0] = 0;
    MPI_Type_create_struct(1, blockcount, offset, oldtype, &complextype);
    MPI_Type_commit(&complextype);

    // distribute rows/columns by workers
    numWorkers = numTasks - 1;
    int avgRow, leftoverRow, avgCol, leftoverCol;
    int rows, cols;
    int startR = 0;
    int startC = 0; 
    if (numWorkers != 0)
    {
        avgRow = height/numWorkers;
        leftoverRow = height%numWorkers;
        avgCol = width/numWorkers;
        leftoverCol = width%numWorkers;
    }
    else
    {
        std::cout << "Only one task, running on master." << std::endl;
        avgRow = height;
        avgCol = width;
    }
    
///////////////////////////////// Master code //////////////////////////////////
    if (rank == 0)
    {
        // first send rows for calculation
        MPI_Status status;
        for (int i = 1; i <= numWorkers; i++)
        {
            (i <= leftoverRow) ? rows = avgRow+1 : rows = avgRow;

            MPI_Send(&rows, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&startR, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&width, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&imgData[startR*width], rows*width, complextype, i, 
                    startTag, MPI_COMM_WORLD);

            //std::cout << "Sent row values to task " << i << std::endl;
            startR += rows;
        }

        for (int i = 1; i <= numWorkers; i++)
        {
            MPI_Recv(&rows, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&width, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&startR, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&imgData[startR*width], rows*width, complextype, i, endTag,
                    MPI_COMM_WORLD, &status);

            //std::cout << "Received row data from worker " << i << std::endl;
        }
        
        transpose(imgData, width);

        for (int i = 1; i <= numWorkers; i++)
        {
            (i <= leftoverCol) ? cols = avgCol+1 : cols = avgCol;

            MPI_Send(&cols, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&startC, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&height, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&imgData[startC*height], cols*height, complextype, i,
                    startTag, MPI_COMM_WORLD);
            
            //std::cout << "Sent column values to task " << i << std::endl;
            startC += cols;
        }

        for (int i = 1; i <= numWorkers; i++)
        {
            MPI_Recv(&cols, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&startC, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&height, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&imgData[startC*height], cols*height, complextype, i, 
                    endTag, MPI_COMM_WORLD, &status);

            //std::cout << "Received column data from worker " << i << std::endl;
        }

        transpose(imgData, height);

        if (numWorkers == 0)
        {
            for (int r = 0; r < height; r++)
                row_fft(&imgData[width*r], width, r);
            transpose(imgData, width);
            for (int c = 0; c < width; c++)
                row_fft(&imgData[height*c], height, c);
            transpose(imgData, height);
        }

//        std::cout << "Writing image data to file..." << std::endl;
        img.save_image_data(argv[3], imgData, width, height);
        
        auto endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Computation took " << elapsed.count() << 
            " seconds." << std::endl;

    	MPI_Type_free(&complextype); 
        MPI_Finalize();
    }

///////////////////////////////// Worker code //////////////////////////////////
    else
    {
        MPI_Status status;
        MPI_Recv(&rows, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&startR, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&width, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status);
        MPI_Recv(&imgData[startR*width], rows*width, complextype, 0, startTag, 
                MPI_COMM_WORLD, &status);

//      std::cout << "Worker " << rank << " received rows" << std::endl;
            
        for (int r = startR; r < startR+rows; r++)
            row_fft(&imgData[width*r], width, r);
        
        MPI_Send(&rows, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&width, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&startR, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&imgData[startR*width], rows*width, complextype, 0, endTag, 
                MPI_COMM_WORLD);
       
//      std::cout << "Worker " << rank << " sent rows to master" << std::endl;
         
        MPI_Recv(&cols, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&startC, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&height, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status);
        MPI_Recv(&imgData[startC*height], cols*height, complextype, 0, startTag, 
                MPI_COMM_WORLD, &status);

//      std::cout << "Worker " << rank << " received columns" << std::endl;

        for (int c = startC; c < startC+cols; c++)
            row_fft(&imgData[height*c], height, c);

        MPI_Send(&cols, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&startC, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&height, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&imgData[startC*height], cols*height, complextype, 0, endTag, 
                MPI_COMM_WORLD);
        
//      std::cout << "Worker " << rank << " sent columns to master" << std::endl;
       
        MPI_Finalize();
    }
}
