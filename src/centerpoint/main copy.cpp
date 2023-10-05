/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h> 
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <fstream>
#include <memory>
#include <chrono>
#include <dirent.h>
#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <unistd.h>
#include <string>
#include "common.h"
#include "centerpoint.h"
#include <chrono>
std::string Model_File = "/home/nvidia/Desktop/way/Lidar_AI_Solution/CUDA-CenterPoint/model/rpn_centerhead_sim.plan";
std::string Save_Dir   = "../data/prediction/";
ros::Publisher pub;
bool verbose = true;
Params params;
cudaStream_t stream = NULL;
float *d_points = nullptr;
CenterPoint centerpoint(Model_File, verbose);
int count=0;
void GetDeviceInfo()
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}




void PrintBoxPred(std::vector<Bndbox>boxes){
    for (const auto box : boxes) {
          std::cout << box.x << " ";
          std::cout << box.y << " ";
          std::cout << box.z << " ";
          std::cout << box.w << " ";
          std::cout << box.l << " ";
          std::cout << box.h << " ";
          std::cout << box.vx << " ";
          std::cout << box.vy << " ";
          std::cout << box.rt << " ";
          std::cout << box.id << " ";
          std::cout << box.score << " ";
          std::cout << "\n";}

}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}


void SaveToBin(const pcl::PointCloud<pcl::PointXYZ>& cloud, const std::string& path) {
    std::cout << "bin_path: " << path << std::endl;
    //Create & write .bin file
    std::ofstream out(path.c_str(), std::ios::out|std::ios::binary|std::ios::app);
    if(!out.good()) {
        std::cout<<"Couldn't open "<<path<<std::endl;
        return;
    }
    for (size_t i = 0; i < cloud.points.size (); ++i) {
        out.write((char*)&cloud.points[i].x, 3*sizeof(float));
       
    }
    out.close();
}
int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}
void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    auto start1 = std::chrono::high_resolution_clock::now();
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaStream_t stream = NULL;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreate(&stream));

    pcl::PointCloud<pcl::PointXYZ> cloud;
    
    pcl::fromROSMsg(*msg, cloud);
    std::string dataFile = "/home/nvidia/nvme_disk/nvidia/way/bins/"+std::to_string(count)+".bin";
    
    SaveToBin(cloud, dataFile);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start1);
    //std::cout << "Save Time: " << duration1.count() << " milliseconds" << std::endl;
    
    count++;
}    


int main(int argc,  char **argv)
{
    

    const char *value = nullptr;
    
    GetDeviceInfo();
    Params params;
    
    checkCudaErrors(cudaStreamCreate(&stream));

    
    centerpoint.prepare();

    float *d_points = nullptr;    
    checkCudaErrors(cudaMalloc((void **)&d_points, MAX_POINTS_NUM * params.feature_num * sizeof(float)));
    
    ros::init(argc, argv, "pointcloud_detector");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/kitti/velo/pointcloud", 10, pointCloudCallback);
    
    ros::spin();
    centerpoint.perf_report();
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaStreamDestroy(stream));
    return 0;
}