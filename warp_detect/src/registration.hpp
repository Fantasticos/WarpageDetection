 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include "testicp.h"
#include <iostream>
//#include <pcl/features/gasd.h>
#include <pcl/common/pca.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include "visualize.h"

using namespace std;
using namespace Eigen;



namespace reg{
//=============================  voxel map class =============================================
//[IMPORTANT] Make sure input cloud lies in XY plane and centered at XY plane, facing -Z direction

//this class divide the cloud into small voxels and return a bool map, the boolmap start from the upper left corner of the bounding box
class VoxelMap {
  private: 
    int nX; int nY;
    bool** boolMap; int** numMap;
    double voxelSize;
    int bool_thr=10;//number of point for judging if a voxel is occupied or not
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointXYZ MinPoint;//[min_x,min_y,min_z]
    pcl::PointXYZ MaxPoint;//[max_x,max_y,max_z]
  public:
    void getWHMinMax(int& width, int& height, pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt){
      width=nX;height=nY;min_pt=MinPoint;max_pt=MaxPoint;}
    VoxelMap(){voxelSize = 0.1;}
    VoxelMap(double size){voxelSize=size;}
    void setVoxelSize(double size){voxelSize=size;}
    //make sure input cloud is in XY plane
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud){cloud=incloud;}
    void setBoolThreshold(int num){bool_thr=num;}
    bool** getBoolMap();
    std::vector<int> findMapLoc(bool** Map1, int h1, int w1, bool** Map2, int h2, int w2);
};


//=============================  registration class =============================================
class registration {
    public:
    void setScan(pcl::PointCloud<pcl::PointXYZ>::Ptr scan);//set scanned cloud
    void setModel(pcl::PointCloud<pcl::PointXYZ>::Ptr model);//set model cloud
    Eigen::Matrix4f getTransform();//get the transformation from model cloud to scanned cloud
    void visualize();//visualize the result of registration
    
    private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene;
    pcl::PointCloud<pcl::PointXYZ>::Ptr object;
    Eigen::Matrix4f T;
    VoxelMap Vmap;  
};
}