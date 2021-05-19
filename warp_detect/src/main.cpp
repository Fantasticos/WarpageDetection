 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department

#include <iostream>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <vector>
#include "registration.hpp"
#include "warpage.hpp"
#include "visualize.h"
#include "gldetect.hpp"


using namespace std;
//pcl::visualization::PCLVisualizer v1r("scene");

//[IMPORTANT] all point clouds should be in meters.
int main(int argc, char **argv){
    //detect green line and get green line coordinate in helios reference
    pcl::PointCloud<pcl::PointXYZ>::Ptr helios (new pcl::PointCloud<pcl::PointXYZ>);//helios scan
    pcl::io::loadPLYFile<pcl::PointXYZ> ("../data/helios-room.ply", *helios);//load helios cloud
    
    Eigen::Matrix4f scale1;
    scale1<<0.001,0,0,0,0,0.001,0,0,0,0,0.001,0,0,0,0,1;//scale if the unit is different, also pbm is at origin, should be moved a little on y direcition
    pcl::transformPointCloud(*helios, *helios ,scale1);

    gld::gldetect greenline;
    std::vector<Eigen::Vector4f> glines = greenline.get3DLines("../data/triton.png",helios);
    for (auto i=0;i<glines.size();++i) std::cout<<glines[i]<<std::endl;
    
    /*
    pcl::visualization::PCLVisualizer visu("helios");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr heliosrgb (new pcl::PointCloud<pcl::PointXYZRGB>);//helios scan
    pcl::io::loadPLYFile<pcl::PointXYZRGB> ("../data/helios-room.ply", *heliosrgb);//load helios cloud
    pcl::transformPointCloud(*heliosrgb, *heliosrgb ,scale1);
    visu.addPointCloud (heliosrgb, "helios");
    pcl::PointXYZ pp1,pp2,pp3,pp4;
    pp1.getVector4fMap()=glines[0];
    pp2.getVector4fMap()=glines[1];
    pp3.getVector4fMap()=glines[2];
    pp4.getVector4fMap()=glines[3];
    visu.addLine<pcl::PointXYZ> (pp1,pp2,0,255.0,0, "green_lineV");
    visu.addLine<pcl::PointXYZ> (pp3,pp4,0,255.0,0, "green_lineH");
    visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3,"green_lineV");
    visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3,"green_lineH");
    visu.spin ();*/

    //load scan and model
    pcl::PointCloud<pcl::PointXYZ>::Ptr object1 (new pcl::PointCloud<pcl::PointXYZ>);//contain the model of window frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr object2 (new pcl::PointCloud<pcl::PointXYZ>);//contain the model of window frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr room (new pcl::PointCloud<pcl::PointXYZ>);//contain the whole room
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene (new pcl::PointCloud<pcl::PointXYZ>);//contain the window
    std::cout<<"loading"<<std::endl;
    //read room and model from given path
    if(pcl::io::loadPCDFile<pcl::PointXYZ> ("../data/AW-2Z_without_noise_front.pcd", *object2) < 0||
    pcl::io::loadPCDFile<pcl::PointXYZ> ("../data/AW-1Z_without_noise_front.pcd", *object1) < 0 ||
    pcl::io::loadPLYFile<pcl::PointXYZ> ("../data/dspCloud.ply", *room) < 0)
    std::cout<<"Error loading object/room file!\n"<<std::endl;
    std::cout<<"load succeed"<<std::endl;
    

    pcl::transformPointCloud(*room, *room,scale1);

    //set bbox vector
    std::vector<Eigen::Vector4f> min_pts(6);//minimum points
    min_pts[0]<<1.2,0.5,-0.6,1;
    min_pts[1]<<1.2,-1.5,-0.6,1;
    min_pts[2]<<-1.05,-1.95,-0.6,1;
    min_pts[3]<<-4,-1.95,-0.6,1;
    min_pts[4]<<-4.6,0.5,-0.6,1;
    min_pts[5]<<-4.6,-1.5,-0.6,1;
    std::vector<Eigen::Vector4f> max_pts(6);//maximum points
    max_pts[0]<<1.5,1.5,0.9,1;
    max_pts[1]<<1.5,-0.5,0.9,1;
    max_pts[2]<<1.05,-1.75,0.9,1;
    max_pts[3]<<-2,-1.75,0.9,1;
    max_pts[4]<<-4.3,1.5,0.9,1;
    max_pts[5]<<-4.3,-0.5,0.9,1;
    //set window type 
    std::vector<int> window_type={2,2,1,1,2,2};

    for(auto i = 0; i < min_pts.size();++i){
    //for each bbox, detect the position orientation and warpage
    //crop the cloud with bbox
    pcl::CropBox<PointNT> cropBoxFilter (true);
    cropBoxFilter.setInputCloud (room);
    cropBoxFilter.setMin (min_pts[i]);
    cropBoxFilter.setMax (max_pts[i]);   
    cropBoxFilter.filter (*scene);
    
    //v1r.addPointCloud (scene, ColorHandlerT (scene, 255.0 ,0,0), "room"+std::to_string(i));
    //v1r.addCoordinateSystem (1.0);
    
    //sor filter

    //bool map and icp positioning with input model and scan
    reg::registration widreg;
    widreg.setScan(scene);//set scanned cloud
    if (window_type[i]==1) widreg.setModel(object1);//set model cloud
    else widreg.setModel(object2);
    Eigen::Matrix4f regtrans=widreg.getTransform();//get the transformation from model cloud to scanned cloud
    //widreg.visualize();//visualize the alignment

    //warpage detect with input control edge position
    pcl::transformPointCloud(*scene, *scene,regtrans.inverse());
    //pcl::visualization::PCLVisualizer v1("icp");
    //v1.addPointCloud (scene, ColorHandlerT (scene, 255.0 ,0,0), "scene");
    //pcl::transformPointCloud(*object, *object,rotMatrix1.inverse());
    //v1.addPointCloud (object, ColorHandlerT (object, 0, 255.0 ,0), "object");
    //v1.spin();
    
    warp::warpage widwarp;
    widwarp.setInputCloud(scene);
    widwarp.setNumMosaic(20);
    widwarp.getWarpage(true);

    }
    //v1r.spin();
    return 1;
}