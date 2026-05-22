#include <iostream>
#include <vector>
#include <limits.h>
#include <fstream>
#include <cassert>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>//cuda inclusion
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>//opencv inclusions
#include <sstream>
#include <string>

using namespace std;

#define MAX_SIZE 1024

#define cudaErrorCheck(result) { cudaAssert((result), __FILE__, __FUNCTION__, __LINE__); }
/*
void cudaAssert

outputs errors in the cuda calls
*/
inline void cudaAssert(cudaError_t err, const char *file,  const char *function, int line, bool quit=true)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr,"cudaAssert failed with error \'%s\', in File: %s, Function: %s, at Line: %d\n", cudaGetErrorString(err), file, function, line);
        if (quit) exit(err);
    }
}

/*
void UpdateEquations(current_state, next_state, gridRows, gridCols)

This cuda kernel is what is used to complete the neccesary game of life computations on the GPU.

This function takes in the current state of the image, and the dimensions of the grid that is being used.
This function will output the next state of the image.
*/
__global__ void UpdateEquations(uchar* current_state, uchar* next_state, int gridRows, int gridCols)
{
    int currentrow = blockIdx.y*blockDim.y + threadIdx.y;//determine the position of the thread in the gpu
    int currentcol = blockIdx.x*blockDim.x + threadIdx.x;

    int occupied_neighbours = 0;
    if (currentrow < gridRows && currentcol < gridCols)//if the current thread is within our block
	{
        for (int jy = currentrow - 1; jy <= currentrow + 1; jy++)//for each neigbor of the current pixel
        {
            for (int jx = currentcol - 1; jx <= currentcol + 1; jx++)
            {
                if (jx == currentcol && jy == currentrow) continue;//skips its own entry
                
                int row = jy;//if on the edge of the image go to the other side to obtain neighbor information
                if (row == gridRows) row = 0;
                if (row == -1) row = gridRows - 1;
                
                int col = jx;
                if (col == gridCols) col = 0;
                if (col == -1) col = gridCols - 1;
                
                if (current_state[gridCols*row + col] == 0) occupied_neighbours++;//add one to the neighbor variable if the negihbor is occupied
            }
        }

        if (current_state[gridCols*currentrow  + currentcol] == 0)//if pixel on current state is alive
        {
            if (occupied_neighbours <= 1 || occupied_neighbours >= 4)
            { 
                next_state[gridCols*currentrow + currentcol] = 255;//next state is dead
            }
            if (occupied_neighbours == 2 || occupied_neighbours == 3)
            { 
                next_state[gridCols*currentrow + currentcol] = 0;//No change on next state
            }
        }
        else if (current_state[gridCols*currentrow  + currentcol] == 255)//if pixel on current state is dead
        {
            if (occupied_neighbours == 3)
            {
                next_state[gridCols*currentrow + currentcol] = 0;//reproduction on next state
            }
        }
    }

}


int main(int argc, char** argv)
{
	cudaError_t err;//variable to store cuda error information
	
	//-----------------------
	// Convert Command Line
	//-----------------------
			
	int ny = atoi(argv[1]);//obtains the image size from command line
	int nx = atoi(argv[2]);
    int maxiter = atoi(argv[3]);//obtains the number of iterations from command line
    int showtime = atoi(argv[4]);//obtain the iterval that we will demonstrate the current image
    int block_size_x = atoi(argv[5]);//defines the block size
    int block_size_y= atoi(argv[6]);
    

    //-----------------------
	// Calculats block size and grid size based on inputted dimensions
	//-----------------------
    dim3 block_size(block_size_x,block_size_y);

    int gridy = (int)ceil((double)ny/(double)block_size_y);
    int gridx = (int)ceil((double)nx/(double)block_size_x);

    dim3 grid_size(gridx , gridy);
	
    assert(ny <= MAX_SIZE);//make sure that sizes are withing proper requirments
    assert(nx <= MAX_SIZE);

    //---------------------------------
    // Generate the initial image randomly
    //---------------------------------
    srand(clock());//used to obtain random values
    cv::Mat population(ny, nx, CV_8UC1);
    cv::Mat SerialPopulation(ny, nx, CV_8UC1);//the Mat objects that will contain the images at different points of time
    cv::Mat proxyPopulation(ny, nx, CV_8UC1);
    cv::Mat NewSerialPopulation(ny, nx, CV_8UC1);
    cv::Mat comparedData(ny, nx, CV_8UC1);
    for (unsigned int iy = 0; iy < ny; iy++)
    {
        for (unsigned int ix = 0; ix < nx; ix++)
        {
            //seed a 1/2 density of alive (just arbitrary really)
            int state = rand()%2;
            if (state == 0){
                population.at<uchar>(iy,ix) = 255; //dead
                SerialPopulation.at<uchar>(iy,ix) = 255;
            }
            else {
                population.at<uchar>(iy,ix) = 0;   //alive
                SerialPopulation.at<uchar>(iy,ix) = 0;
            }
        }
    }
    if (showtime != 0){
        cv::namedWindow("Initial Game Status",cv::WINDOW_AUTOSIZE);//creates the initial image on our monitors
        cv::imshow("Initial Game Status",population);      
        cv::waitKey(1000);        
    }

    //---------------------------------------
    // Setup Profiling
    //---------------------------------------
    cudaEvent_t start, stop;//create all required elements for cuda timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //---------------------------------------
    // Create GPU (Device) Buffers
    //---------------------------------------
    uchar* dev_population;
    uchar* dev_proxyPopulation;

    err = cudaMalloc((void**)&dev_population, (ny*nx)*sizeof(uchar)); cudaErrorCheck(err);//allocate memory on the GPU for our images
    err = cudaMalloc((void**)&dev_proxyPopulation, (ny*nx)*sizeof(uchar)); cudaErrorCheck(err);

    //---------------------------------------
    // Copy Memory To Device
    //---------------------------------------
    err = cudaMemcpy(dev_population, population.data, (ny*nx)*sizeof(uchar), cudaMemcpyHostToDevice); cudaErrorCheck(err);//copy the origonal image to the GPU
    
    for (int iter = 0; iter < maxiter; iter++)
    {
        UpdateEquations<<<grid_size, block_size>>>(dev_population, dev_proxyPopulation, ny, nx);//our cuda kernel
        if (showtime != 0 && iter%showtime == 0){//if we are meant to show the progress of our image
            err = cudaMemcpy(proxyPopulation.data, dev_proxyPopulation, (ny*nx)*sizeof(uchar), cudaMemcpyDeviceToHost); assert(err == cudaSuccess);//obtain the images current orientation
            cv::namedWindow("Current Parallel CUDA Game Status",cv::WINDOW_AUTOSIZE);//print the current orientation
            cv::imshow("Current Parallel CUDA Game Status",proxyPopulation);      
        }
        err = cudaMemcpy(dev_population, dev_proxyPopulation, (ny*nx)*sizeof(uchar), cudaMemcpyDeviceToDevice); cudaErrorCheck(err);//switch the images in the image for the next iteration
        if (iter == maxiter - 1){//if we are on the last iteration
            err = cudaMemcpy(proxyPopulation.data, dev_proxyPopulation, (ny*nx)*sizeof(uchar), cudaMemcpyDeviceToHost); assert(err == cudaSuccess);
            comparedData = proxyPopulation.clone();
        }
    }    
    cudaEventRecord(stop,0);//stop recording the time and determine the ellapsed time for the cuda implementation
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);//time in milliseconds    

    gpu_time /= 1000.0;
    std::cout << "Done GPU Computations in " << gpu_time << " seconds" << std::endl; std::cout.flush();//output gpu time

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //----------------------------------------------------
	// CPU COMPUTATION
    //----------------------------------------------------

    double cpu_t_start = (double)clock()/(double)CLOCKS_PER_SEC;//obtain the start time for the serial code
    for (int iter = 0; iter < maxiter; iter++)//serial code is identical to provided code
    {   
        for (int iy = 0; iy < ny; iy++)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                int occupied_neighbours = 0;

                for (int jy = iy - 1; jy <= iy + 1; jy++)
                {
                    for (int jx = ix - 1; jx <= ix + 1; jx++)
                    {
                        if (jx == ix && jy == iy) continue;
                        
                        int row = jy;
                        if (row == ny) row = 0;
                        if (row == -1) row = ny-1;
                        
                        int col = jx;
                        if (col == nx) col = 0;
                        if (col == -1) col = nx - 1;
                        
                        if (SerialPopulation.at<uchar>(row,col) == 0) occupied_neighbours++;
                    }
                }
            
                if (SerialPopulation.at<uchar>(iy,ix) == 0)   //alive
                {
                    if (occupied_neighbours <= 1 || occupied_neighbours >= 4) NewSerialPopulation.at<uchar>(iy,ix) = 255; //dies
                    if (occupied_neighbours == 2 || occupied_neighbours == 3) NewSerialPopulation.at<uchar>(iy,ix) = 0; //same as population
                }
                else if (SerialPopulation.at<uchar>(iy,ix) == 255) //dead
                {
                    if (occupied_neighbours == 3)
                    {
                        NewSerialPopulation.at<uchar>(iy,ix) = 0; //reproduction
                    }
                }
            }
        }
        SerialPopulation = NewSerialPopulation.clone();
    }  
    double cpu_time = (double)clock()/(double)CLOCKS_PER_SEC - cpu_t_start;//obtain the cpu time
    std::cout << "CPU Time = " << cpu_time << std::endl;

    cv::namedWindow("Final CUDA Game Status",cv::WINDOW_AUTOSIZE);//output final game statuses
    cv::imshow("Final CUDA Game Status",comparedData);        
    cv::namedWindow("Final Serial Game Status",cv::WINDOW_AUTOSIZE);
    cv::imshow("Final Serial Game Status",SerialPopulation);  
    
    //----------------------------------------------------
	// Validation
    //----------------------------------------------------
    int success = 0;
    for (int i = 0; i < SerialPopulation.rows; i++){//validate that the cuda and serial implementations are identical
        for (int j = 0; j < SerialPopulation.cols; j++){
            if (SerialPopulation.at<uchar>(i,j) != comparedData.at<uchar>(i,j)){
                success = 1;
            }
        }
    }
    if (success){//output a message stating if our implementation worked or not
        cout << "The Serial and CUDA Final Image are NOT Identical therefore our CUDA Implementation does NOT Work" << endl;
    }
    else{
        cout << "The Serial and CUDA Final Image are Identical therefore our CUDA Implementation Works" << endl;
    }

    //----------------------------------------------
    // Display Timing Results and Images
    //----------------------------------------------
    std::cout << "GPU Time = " << gpu_time << std::endl;
    std::cout << "CPU Time = " << cpu_time << std::endl;
    std::cout << "Speedup = " << cpu_time/gpu_time << std::endl;
}