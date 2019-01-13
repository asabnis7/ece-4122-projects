#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <iomanip>

#define T_P_B 1024 


///////////////////////////// Global variables /////////////////////////////////

std::string dimension;          // grid dimension
float k;                        // k-step
int timesteps;                  // num of timesteps
int width, height, depth;       // grid size
float startTemp, fixedTemp;     // node start temp
int heat_x, heat_y, heat_z,     // fixed heater vars
    heat_w, heat_h, heat_d;

float *d_old, *d_new, *d_heaters, // grids for values
      *g_old, *g_new, *heaters;


///////////////////////////// CUDA Functions ///////////////////////////////////

__global__ void heat_sim(float *oldg, float * newg, float *fixed, 
						 int width, int height, int depth, float k)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float left, right, up, down, above, below;
	float old = oldg[idx];
	
	if (idx < (width*height*depth))
	{	
		if (fixed[idx] != 0) newg[idx] = fixed[idx];
		else if (fixed[idx] == 0)
		{	
			// x-, x+
			((idx%width) == 0) ? (left = old) : (left = oldg[idx-1]);
			((idx%width) == (width-1)) ? (right = old) : (right = oldg[idx+1]);
		
			// y-, y+
			(idx%(width*height) < width) ? (up = old) : (up = oldg[idx - width]);
			(idx%(width*height) >= ((height-1)*width)) 
			? (down = old) : (down = oldg[idx + width]);
		
			// z-, z+
			if (depth <= 1)
			{
				above = 0.0;
				below = 0.0;
				newg[idx] = oldg[idx] + k*(up+down+left+right-(4.0*oldg[idx]));
				
			}
			else if (depth > 1)
			{
				if (idx < (width*height)) above = old;
				else above = oldg[idx - (width*height)];
				if (idx >= ((depth-1)*(width*height))) below = old;
				else below = oldg[idx + (width*height)];
				newg[idx] = oldg[idx] + k*(up+down+left
											+right+above+below-(6.0*oldg[idx]));
			}
		}
	}
}

__global__ void grid_cpy(float *oldg, float *newg, int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size)
		oldg[idx] = newg[idx];
}

///////////////////////////// End CUDA Functions ///////////////////////////////



int main(int argc, char * argv[])
{
///////////////////////////// Config file parser ///////////////////////////////

    std::ifstream conf(argv[1]);
    if (conf.is_open())
    {
        std::string line;
        while (getline(conf, line)){
            if ((line[0] == '#') || line.empty() || line[0] == '\r')  
		        continue;
			
			// get dimension
            while ((line[0] == '#') || line.empty() || line[0] == '\r') 
				getline(conf,line);
            dimension = line.substr(0,2);
            
			// get k value
			getline(conf, line);
            while ((line[0] == '#') || line.empty() || line[0] == '\r') 
				getline(conf,line);
            k = std::stof(line);
            
			// get timesteps
			getline(conf, line);	
            while ((line[0] == '#') || line.empty() || line[0] == '\r') 
				getline(conf,line);
            timesteps = std::stoi(line);
            
			// get grid size
			getline(conf, line);
            while ((line[0] == '#') || line.empty() || line[0] == '\r') 
				getline(conf,line);
            int comma = line.find(',');
            width = std::stoi(line.substr(0, comma));
            line = line.substr(comma+1);
            if (dimension == "2D"){
                height = std::stoi(line);
                depth = 1;
            }
            else if (dimension == "3D"){
                comma = line.find(','); 
                height = std::stoi(line.substr(0, comma));
                depth = std::stoi(line.substr(comma+1));
            }
            
			// get block start temp
    		getline(conf, line);
            while ((line[0] == '#') || line.empty() || line[0] == '\r') 
				getline(conf,line);
            startTemp = std::stof(line);
            
			// create heaters
			heaters = new float[width*height*depth];
			std::fill(heaters, heaters+(width*height*depth), 0);
                
			while(getline(conf, line)){
		    	if (line[0] == '#' || line.empty() || line[0] == '\r') 
			        continue;
                int comma = line.find(',');
                heat_x = std::stoi(line.substr(0, comma));
                line = line.substr(comma+1);
                comma = line.find(',');
                heat_y = std::stoi(line.substr(0, comma));
                line = line.substr(comma+1);
                comma = line.find(',');
                if (dimension == "2D"){
                    heat_w = std::stoi(line.substr(0, comma));
                    line = line.substr(comma+1);
                    comma = line.find(',');
                    heat_h = std::stoi(line.substr(0, comma));
                    line = line.substr(comma+1);
                    heat_d = 1;
                    heat_z = 0;
                    fixedTemp = std::stof(line);
                }
                else if (dimension == "3D"){
                    heat_z = std::stoi(line.substr(0, comma));
                    line = line.substr(comma+1);
                    comma = line.find(',');
                    heat_w = std::stoi(line.substr(0, comma));
                    line = line.substr(comma+1);
                    comma = line.find(',');
                    heat_h = std::stoi(line.substr(0, comma));
                    line = line.substr(comma+1);
                    comma = line.find(',');
                    heat_d = std::stoi(line.substr(0, comma));
                    line = line.substr(comma+1);
                    fixedTemp = std::stof(line);
                }
                for (int i = heat_x+width*heat_y; 
						 i < heat_x+heat_w+width*heat_y; i++)
                    for (int j = 0; j < heat_h; j++)
                        for (int k = heat_z; k < heat_z+heat_d; k++)
                            heaters[i+(j*width)+(k*width*height)] = fixedTemp;
            }
        }
    }
    else std::cerr << "Couldn't open config file.";

////////////////////////// End config file parser //////////////////////////////

	int dim = width*height*depth;
	
	// set up host grids
	g_old = new float[dim];
	g_new = new float[dim];
	std::fill(g_new, g_new+dim, 0);
	std::fill(g_old, g_old+dim, 0);

	for (int i = 0; i < dim; i++)
	{
		g_old[i] = startTemp;
		if (heaters[i] != 0) g_old[i] = heaters[i];
	}

	// allocate blockSize - must be at least one block
	int blockSize = ceil(float(dim)/float(T_P_B));

	// allocate device memory in 1D array
	cudaMalloc((void**)&d_new, dim*sizeof(float));
	cudaMalloc((void**)&d_old, dim*sizeof(float));
	cudaMalloc((void**)&d_heaters, dim*sizeof(float));
	
	// copy filled arrays from host to device
	cudaMemcpy(d_old, g_old, dim*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_new, g_new, dim*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_heaters, heaters, dim*sizeof(float), cudaMemcpyHostToDevice);
	
	// run kernels
	for (int t = 0; t < timesteps; t++)
	{
		heat_sim<<<blockSize, T_P_B>>> (d_old, d_new, d_heaters, 
										 width, height, depth, k);
		cudaDeviceSynchronize();
		grid_cpy<<< blockSize, T_P_B>>> (d_old, d_new, dim);		
		cudaDeviceSynchronize();
	}

	// copy data back from device to host
	cudaMemcpy(g_new, d_new, dim*sizeof(float), cudaMemcpyDeviceToHost);


    // print out to csv
    std::ofstream csv("../heatOutput.csv", std::ios::out);            
    if (csv.is_open()){ 
        for (int i = 0; i < dim; i++)
        {
            if (i%width == width-1) csv << g_new[i] << std::endl;
            else csv << g_new[i] << ", ";
            if (i%(width*height) == (width*height)-1) csv << std::endl;
        }
    }
    else
        std::cout << "Unable to open file, try again." << std::endl;           
    csv.close();

	// deallocate all memory
	delete[] g_old;
	delete[] g_new;
	delete[] heaters;
	cudaFree(d_old);
	cudaFree(d_new);
	cudaFree(d_heaters);
}
