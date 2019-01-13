#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#define threadNum 12

///////////////////////////// Global variables /////////////////////////////////

std::string dimension;          // grid dimension
float k;                        // k-step
int timesteps;                  // num of timesteps
int width, height, depth;       // grid size
float startTemp, fixedTemp;     // node start temp
int heat_x, heat_y, heat_z,     // fixed heater vars
    heat_w, heat_h, heat_d;

float *g_old, *g_new, *heaters; // hold values


///////////////////////////// Thread Functions /////////////////////////////////

void heat_sim(float *oldg, float * newg, float *fixed, int idx, int len)
{
    for (int i = idx; i < idx+len-1; i++)
    {
	    float left, right, up, down, above, below;
	    float old = oldg[i];
	
    	if (fixed[i] != 0) newg[i] = fixed[i];
    	else if (fixed[i] == 0)
    	{	
    		// x-, x+
    		((i % width) == 0) ? (left = old) : (left = oldg[i-1]);
    		((i % width) == (width-1)) ? (right = old) : (right = oldg[i+1]);
    	
    		// y-, y+
    		(i % (width*height) < width) ? (up = old) : (up = oldg[i - width]);
    		(i % (width*height) >= ((height-1)*width)) 
                                ? (down = old) : (down = oldg[i + width]);
    	
    		// z-, z+
    		if (depth <= 1)
    		{
    			above = 0.0;
    			below = 0.0;
    			newg[i] = oldg[i] + k*(up+down+left+right-(4.0*oldg[i]));
    			
    		}
    		else if (depth > 1)
    		{
    			if (i < (width*height)) above = old;
    			else above = oldg[i - (width*height)];
    			if (i >= ((depth-1)*(width*height))) below = old;
    			else below = oldg[i + (width*height)];
    			newg[i] = oldg[i] + k*(up+down+left
    					  +right+above+below-(6.0*oldg[i]));
    		}
    	}
    }
}

void grid_cpy(float *oldg, float *newg, int idx, int len)
{
	for (int i = idx; i < idx+len; i++)	
        oldg[i] = newg[i];
}

//////////////////////////// End Thread Functions //////////////////////////////



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

    auto startTime = std::chrono::system_clock::now();

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
	int avg = dim/threadNum;
    int leftover = dim%threadNum;
    int len;

    std::thread th[threadNum];

	// run threads
	for (int t = 0; t < timesteps; t++)
	{   
        int start = 0;
        for (int i = 0; i < threadNum; i++)
        {
            (i <= leftover) ? len = avg+1 : len = avg;
            th[i] = std::thread(heat_sim, g_old, g_new, heaters, start, len);
            start += len;
        }
        start = 0;
        for (int i = 0; i < threadNum; i++) th[i].join();
        for (int i = 0; i < threadNum; i++)
        {
            (i <= leftover) ? len = avg+1 : len = avg;
            th[i] = std::thread(grid_cpy, g_old, g_new, start, len);
            start += len;
        }
        for (int i = 0; i < threadNum; i++) th[i].join();
	}

    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

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

    printf("Computation for %d nodes took %f seconds.\n\r", dim, elapsed.count());

	// deallocate all memory
	delete[] g_old;
	delete[] g_new;
	delete[] heaters;
}
