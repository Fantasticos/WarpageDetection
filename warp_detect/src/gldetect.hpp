 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>

using namespace cv;

namespace gld{
    class gldetect{
        private:
            static bool isEqual(const Vec4i& _l1, const Vec4i& _l2);
            std::vector<Eigen::Vector4f> lsd(char* img_pth);//cloud be replaced with Tomotake's algorithm
            std::vector<Eigen::Vector4f> lsd3d(pcl::PointCloud<pcl::PointXYZ>::Ptr scene, std::vector<Eigen::Vector4f> glnorms);
        public:
            std::vector<Eigen::Vector4f> get3DLines(char* img_pth,pcl::PointCloud<pcl::PointXYZ>::Ptr scene);
    };
}