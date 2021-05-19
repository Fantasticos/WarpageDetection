 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <iostream>
#include <math.h>
#include <pcl/filters/passthrough.h>
#include <string>
#include <vector>
#include <pcl/filters/crop_box.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/pca.h>
#include "lineseg2d.h"
#include "visualize.h"

typedef pcl::PointXYZ PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;

namespace warp{
class warpage{
    private:
    PointCloudT::Ptr surf_cluster;
    void calcLine(pcl::ModelCoefficients::Ptr coefsOfPlane1, pcl::ModelCoefficients::Ptr coefsOfPlane2, pcl::ModelCoefficients::Ptr coefsOfLine);
    void voxel_edge_detect(Eigen::Vector4f inmin_pt,Eigen::Vector4f inmax_pt,int num_mosaic,bool visual, int edge_expect);
    int num_mosaic=20;
    public:
    void setInputCloud(PointCloudT::Ptr cloud);
    void setNumMosaic(int n){num_mosaic = n;}
    void getWarpage(bool visual);
};
}