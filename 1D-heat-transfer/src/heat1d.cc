// Arjun Sabnis
// 1-D heat transfer using MPI

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstdlib>

#define startTag    1
#define leftTag     2
#define rightTag    3
#define endTag      4

void updateGrid(int offset, int endLoc, double* grid, double* newGrid, double r, double t1, double t2, int numPts)
{
    for (int i = offset; i <= endLoc; i++)
    {
        if (i == 0)
            newGrid[0] = (1-2*r)*grid[0] + r*t1 + r*grid[1];
        else if (i == numPts-1)
            newGrid[i] = (1-2*r)*grid[i] + r*grid[i-1] + r*t2;
        else
            newGrid[i] = (1-2*r)*grid[i] + r*grid[i-1] + r*grid[i+1];           
    }
}       

void csvPrint(int gridPts, double * grid)
{
    std::ofstream csv("../heat1Doutput.csv", std::ios::out);            
    
    if (csv.is_open()){ 
        for (int i = 0; i < gridPts; i++)
        {
            if (i == gridPts - 1)
                csv << grid[i];
            else
                csv << grid[i] << ", ";
        }
    }
    else
        std::cout << "Unable to open file, try again." << std::endl;
            
    csv.close();
}

void changeOld(double* grid, double* newGrid, int offset, int endLoc)
{
    for (int i = offset; i <= endLoc; i++)
    {
        grid[i] = newGrid[i];
    //  std::cout << "grid " << i << "=" << grid[i] << std::endl;
    }
}

int main(int argc, char * argv[])
{
    double temp1 = std::atof(argv[1]);
    double temp2 = std::atof(argv[2]);
    int numGridPts = std::atof(argv[3]);
    int numTimesteps = std::atof(argv[4]);
    double grid[numGridPts] = {0};
    double newGrid[numGridPts] = {0};

    double k = 1;
    double h = 2;
    double r = k/(h*h);

    int rank, numWorkers, numTasks, conf;    // MPI variables
    int left, right, offset, first, last;    // give workers neighbors/update location
    int averagePts, leftoverPts, gridPts;    // distribute tasks

    conf = MPI_Init(&argc, &argv);
    if (conf != MPI_SUCCESS)
    {
        std::cout << "Error starting MPI program" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, conf);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    numWorkers = numTasks - 1;
    
    if (numWorkers != 0)
    {
        averagePts = numGridPts/numWorkers;
        leftoverPts = numGridPts%numWorkers;
    }
    else
    {
        std::cout << "Only one task, running on master" << std::endl;
        averagePts = numGridPts;
        leftoverPts = 0;
    } 
    offset = 0;
    
    if (rank == 0) 
    {
        MPI_Status status;
        for (int i = 1; i <= numWorkers; i++)
        {
            if (i <= leftoverPts) gridPts = averagePts + 1;
            else gridPts = averagePts;

            if (i == 1) left = 0;
            else left = i - 1;
            if (i == numWorkers) right = 0;
            else right = i + 1;

            MPI_Send(&offset, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&gridPts, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&right, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&left, 1, MPI_INT, i, startTag, MPI_COMM_WORLD); 
            MPI_Send(&grid[offset], gridPts, MPI_DOUBLE, i, startTag, MPI_COMM_WORLD);
            MPI_Send(&newGrid[offset], gridPts, MPI_DOUBLE, i, startTag, MPI_COMM_WORLD);
            
            std::cout << "Sent values to task " << i << std::endl;
            
            offset += gridPts;
        }
        
        for (int i = 1; i <= numWorkers; i++)
        {
            MPI_Recv(&offset, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&gridPts, 1, MPI_INT, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&grid[offset], gridPts, MPI_DOUBLE, i, endTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&newGrid[offset], gridPts, MPI_DOUBLE, i, endTag, MPI_COMM_WORLD, &status);

            std::cout << "Received data from worker " << i << std::endl;
        }
        if (numWorkers == 0)
        {
            for (int t = 0; t < numTimesteps; t++)
            {
                updateGrid(0, numGridPts-1, &grid[0], &newGrid[0], r, temp1, temp2, numGridPts);
                changeOld(&grid[0], &newGrid[0], 0, numGridPts-1);
            }
        }
        std::cout << "Writing data to heat1Doutput.csv..." << std::endl;
        csvPrint(numGridPts, newGrid);
        MPI_Finalize();
    }
        
    else
    {
        MPI_Status status;
        MPI_Recv(&offset, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&gridPts, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&right, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&left, 1, MPI_INT, 0, startTag, MPI_COMM_WORLD, &status); 
        MPI_Recv(&grid[offset], gridPts, MPI_DOUBLE, 0, startTag, MPI_COMM_WORLD, &status);
        MPI_Recv(&newGrid[offset], gridPts, MPI_DOUBLE, 0, startTag, MPI_COMM_WORLD, &status);
        
        first = offset;
        last = offset + gridPts - 1;

        std::cout << "Worker " << rank << " received data" << std::endl;
        std::cout << "first = " << first << ", last = " << last << std::endl; 
        for (int t = 0; t < numTimesteps; t++)
        {
            if (left != 0)
            {
                MPI_Sendrecv(&grid[offset], 1, MPI_DOUBLE, left, rightTag, 
                            &grid[offset-1], 1, MPI_DOUBLE, left, leftTag, MPI_COMM_WORLD, &status);
            }
            if (right != 0)
            {
                MPI_Sendrecv(&grid[offset+gridPts-1], 1, MPI_DOUBLE, right, leftTag, 
                            &grid[offset+gridPts], 1, MPI_DOUBLE, right, rightTag, MPI_COMM_WORLD, &status);
            }
            
            updateGrid(first, last, &grid[0], &newGrid[0], r, temp1, temp2, numGridPts);
            changeOld(&grid[0], &newGrid[0], first, last);
        } 

        MPI_Send(&offset, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&gridPts, 1, MPI_INT, 0, endTag, MPI_COMM_WORLD); 
        MPI_Send(&grid[offset], gridPts, MPI_DOUBLE, 0, endTag, MPI_COMM_WORLD);
        MPI_Send(&newGrid[offset], gridPts, MPI_DOUBLE, 0, endTag, MPI_COMM_WORLD);
        std::cout << "Worker " << rank << " sent values to master" << std::endl;
        MPI_Finalize();
    }
}
