#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include "testicp.h"
#include <iostream>
#include <pcl/features/gasd.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>

using namespace std;
using namespace Eigen;
using namespace cv;
//typedef pcl::PointNormal PointNT;
//typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerT;
//#define PI 3.1415926536

//=============================  2D lsd function  =============================================
bool isEqual(const Vec4i& _l1, const Vec4i& _l2)
{
    Vec4i l1(_l1), l2(_l2);

    float length1 = sqrtf((l1[2] - l1[0])*(l1[2] - l1[0]) + (l1[3] - l1[1])*(l1[3] - l1[1]));
    float length2 = sqrtf((l2[2] - l2[0])*(l2[2] - l2[0]) + (l2[3] - l2[1])*(l2[3] - l2[1]));

    float product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1]);

    if (fabs(product / (length1 * length2)) < cos(CV_PI / 30))
        return false;

    float mx1 = (l1[0] + l1[2]) * 0.5f;
    float mx2 = (l2[0] + l2[2]) * 0.5f;

    float my1 = (l1[1] + l1[3]) * 0.5f;
    float my2 = (l2[1] + l2[3]) * 0.5f;
    float dist = sqrtf((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2));

    if (dist > std::max(length1, length2))
        return false;

    return true;
}

std::vector<Eigen::Vector4f> lsd(char* img_pth)
{
  //read image
  cv::Mat image, lab_img;
  image = cv::imread( img_pth, 1 );
  //use lab color space to extract grean lines
  cv::cvtColor(image, lab_img,cv::COLOR_BGR2Lab);
  cv::Mat glimg;
  cv::inRange(lab_img,cv::Scalar(180,0,0),cv::Scalar(255,80,255),glimg);
  cv::imwrite("gl.png",glimg);
  cv::dilate(glimg,glimg,cv::Mat());
  cv::erode(glimg,glimg,cv::Mat());
  cv::imwrite("gl_1.png",glimg);
  //detect line using houghline detection
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(glimg,lines,3,CV_PI/180,50,50,5);
    std::cout<<"line detected"<<lines.size()<<std::endl;
    for(size_t i=0;i<lines.size();i++)
    {
        cv::Vec4i l = lines[i];
        std::cout<<l<<std::endl;
        cv::line(image, cv::Point(l[0],l[1]),cv::Point(l[2],l[3]),cv::Scalar(0,0,255),3,8);
    }
    //group lines
    std::vector<int> labels;
    int numberOfLines = cv::partition(lines, labels, isEqual);
    //std::cout<<numberOfLines<<labels[0]<<labels[1]<<labels[2]<<std::endl;
    //gind the longest line in each group
    std::vector<float> max_length(numberOfLines,0.0);
    std::vector<int> max_index(numberOfLines);
    for(size_t i=0;i<lines.size();i++)
    {
        for(size_t j=0;j<numberOfLines;j++)
        {
            if (labels[i]==j)
            {
                float length = sqrtf((lines[i][2] - lines[i][0])*(lines[i][2] - lines[i][0]) + (lines[i][3] - lines[i][1])*(lines[i][3] - lines[i][1]));
                if(length>max_length[j]){
                    max_length[j]=length;
                    max_index[j]=i;
                }
                break;
            }
        }
    }
    
    //find the normal of green planes according to these longest lines
    float angleh=CV_PI*62.0/180.0;
    float anglev=CV_PI*38.75/180.0;
    float y=960/tan(angleh/2);
    std::vector<Eigen::Vector4f> glnorms;
    for(size_t i=0;i<numberOfLines;i++){
        Eigen::Vector3f p1(lines[max_index[i]][0]-960,y,600-lines[max_index[i]][1]);
        Eigen::Vector3f p2(lines[max_index[i]][2]-960,y,600-lines[max_index[i]][3]);
        Eigen::Vector4f n;
        n<<(p1/100.0).cross(p2/100.0),1;
        //std::cout<<n<<std::endl;
        glnorms.push_back(n);
    }
  return glnorms;
}
//=============================  3D green line detection function  =============================================
std::vector<Eigen::Vector4f> gldetect(pcl::PointCloud<pcl::PointXYZ>::Ptr scene, std::vector<Eigen::Vector4f> glnorms)
{
    Eigen::Vector4f camera_center(0,0,-0.05,1);//camera position
    Eigen::Vector4f gline_norm;

    std::vector<Eigen::Vector4f> gl3ds;//store the detected 3d lines
    for(int i=0;i<glnorms.size();i++)
    {
    gline_norm=glnorms[i];
    //translate and rotate the cloud alone y and then z to make the green line in yz plane
    double norm_x=gline_norm(0);
    double norm_y=gline_norm(1);
    double norm_z=gline_norm(2);
    double rotAngy=atan2(norm_z,norm_x);//y axis rotation angle
    double rotAngz=-atan2(norm_y,sqrt(norm_z*norm_z+norm_x*norm_x));//z axis rotation angle
    
    //apply transformation matrix
    Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
    transMat.col(3) = camera_center;
    //std::cout<<transMat<<std::endl;

    Eigen::Matrix4f rotMaty = Eigen::Matrix4f::Identity();
    rotMaty(0,0) = cos(rotAngy);
    rotMaty(0,2) = sin(rotAngy);
    rotMaty(2,0) = -sin(rotAngy);
    rotMaty(2,2) = cos(rotAngy);
    Eigen::Matrix4f rotMatz = Eigen::Matrix4f::Identity();
    rotMatz(0,0) = cos(rotAngz);
    rotMatz(0,1) = -sin(rotAngz);
    rotMatz(1,0) = sin(rotAngz);
    rotMatz(1,1) = cos(rotAngz);

    //apply transformation to Helios2 pointcloud,first translate to camera frame, then rotate along y, finally rotate along z
    pcl::transformPointCloud (*scene, *scene, rotMatz*rotMaty*transMat);
    //std::cout<<rotMatz*rotMaty*gline_norm<<std::endl;

    //use a bounding box to crop the local feature, assume the height is less than 4m and width 1m,depth no more than 10m
    double bbox_x=0.5;
    double bbox_z=2;
    double bbox_y=10;
    Eigen::Vector4f min_pt(-bbox_x,0.0,-bbox_z,1.0);
    Eigen::Vector4f max_pt(bbox_x,bbox_y,bbox_z,1.0);
    pcl::CropBox<pcl::PointXYZ> cropBoxFilter (true);
    cropBoxFilter.setInputCloud (scene);
    cropBoxFilter.setMin (min_pt);
    cropBoxFilter.setMax (max_pt);
    pcl::PointCloud<pcl::PointXYZ>::Ptr crop_result (new pcl::PointCloud<pcl::PointXYZ>);
    cropBoxFilter.filter (*crop_result);


    //fit a plane on the cropped cloud
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud (crop_result);
    seg.segment (*inliers, *coefficients);

    
    //calculate the green line and rotate and translate it back to helios2 space
    //green line is calculated by intersecting the ransac plan (wall) with yz plane
    double a=coefficients -> values[0];
    double b=coefficients -> values[1];
    double c=coefficients -> values[2];
    double d=coefficients -> values[3];//here the ransac plane is ax+by+cz+d=0

    //find two points on the intersection to represent the green line, choose x=0,z= 2 or -2, calculate y
    Eigen::Vector4f glpt1(0.0,(-c*bbox_z-d)/b,bbox_z,1.0f);
    Eigen::Vector4f glpt2(0.0,(c*bbox_z-d)/b,-bbox_z,1.0f);

    //transform them to point cloud space
    glpt1=(rotMatz*rotMaty*transMat).inverse()*glpt1;
    glpt2=(rotMatz*rotMaty*transMat).inverse()*glpt2;
    pcl::transformPointCloud (*scene, *scene, (rotMatz*rotMaty*transMat).inverse());

    gl3ds.push_back(glpt1);
    gl3ds.push_back(glpt2);
    }
    //=====================================================
    //fit the plane of the wall
    /*
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud (scene);
    seg.segment (*inliers, *coefficients);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(scene);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*window);
    */
  return gl3ds;
}
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
    //make sure input cloud is in XY plane
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud){cloud=incloud;}
    void setBoolThreshold(int num){bool_thr=num;}
    bool** getBoolMap();
};

bool** VoxelMap::getBoolMap(){
  //get xy size
  pcl::getMinMax3D(*cloud,MinPoint, MaxPoint);
  //boolmap size
  nX=(MaxPoint.x-MinPoint.x)/voxelSize+1;
  nY=(MaxPoint.y-MinPoint.y)/voxelSize+1;
  //initialize map
  numMap=new int*[nY];
  boolMap=new bool*[nY];
  for (int i=0;i<nY;++i){
    numMap[i]=new int[nX];
    boolMap[i]=new bool[nX];
    for(int j=0;j<nX;++j){
      numMap[i][j]=0;
      boolMap[i][j]=true;
    }
  }
  //project all points to int** numMap
  for(int i=0;i<cloud->points.size();++i){
    int xx=(cloud->points[i].x-MinPoint.x)/voxelSize;//find x map coordinate for points[i], xx=0 is left most
    int yy=(cloud->points[i].y-MinPoint.y)/voxelSize;//find z map coordinate for points[i], yy=0 is up most
    numMap[yy][xx]++;
  }
  //get bool** boolMap by threshold
  for (int i=0;i<nY;++i){
    for (int j=0;j<nX;++j){
      if(numMap[i][j]<bool_thr){boolMap[i][j]=0;}
    }
  }
  return boolMap;
}
//compare 2 boolMap and find the maximum correlation, return a vector represent the position of map 2 left upper corner in map 1
//the algorithm will end once a qualified result is found
std::vector<int> findMapLoc(bool** Map1, int h1, int w1, bool** Map2, int h2, int w2)
{
  vector<int> loc;
  int max_score=0;
  int max_i=0, max_j=0;
  for (int i1=0;i1<=h1-h2;++i1){
    for (int j1=0;j1<=w1-w2;++j1){
      int score=0;
      for (int i2=0;i2<h2;++i2){
        for (int j2=0;j2<w2;++j2){
          if(Map1[i1+i2][j1+j2]==Map2[i2][j2]) score++;
        }
      }
      if(score==w2*h2){
        loc.push_back(j1);
        loc.push_back(i1);
        return loc;
      }
      if(score>max_score){
        max_score=score;
        max_i=i1;
        max_j=j1;
      }
    }
  }
  loc.push_back(max_j);
  loc.push_back(max_i);
  return loc;
}
//=============================  main function (pose detection) =============================================

//
int main(int argc, char **argv)
{
//============================ detect 2d green lines==================================
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::vector<Eigen::Vector4f> glnorms=lsd("../triton.png");//get green line plane norms from 2D image
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference for 2d lsd = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

//============================ load files =============================================
  pcl::PointCloud<pcl::PointXYZ>::Ptr object (new pcl::PointCloud<pcl::PointXYZ>);//contain the model of window frame
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene (new pcl::PointCloud<pcl::PointXYZ>);//contain the window part of helios scan

  //read scene and model from given path
  if(pcl::io::loadPCDFile<pcl::PointXYZ> ("../data/AW-2Z_without_noise_front.pcd", *object) < 0||
  pcl::io::loadPLYFile<pcl::PointXYZ> ("../../data/helios.ply", *scene) < 0)
  std::cout<<"Error loading object/scene file!\n"<<std::endl;

//=========================== detect 3d green lines===================================
  begin = std::chrono::steady_clock::now();
  //get 3d green lines from cloud, return gl3ds in form of {line1_start, line1_end, line2_start, line2_end , ...}
  std::vector<Eigen::Vector4f> gl3ds=gldetect(scene,glnorms);
  end = std::chrono::steady_clock::now();
  std::cout << "Time difference for 3d gl detect = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  //Eigen::Matrix4f rotMatrix1;rotMatrix1<<1,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,1;//x axis rotation
  //pcl::transformPointCloud(*object, *object,rotMatrix1);
  Eigen::Matrix4f rotMatrix1;rotMatrix1<<-1,0,0,0,0,-1,0,0,0,0,1,0,0,0,0,1;//z axis rotation
  pcl::transformPointCloud(*object, *object,rotMatrix1);

// prepare map according to window
//========================= use voxel map to find the window===========================
  //Eigen::Matrix4f voxel_trans=transformXZ(scene);
  begin = std::chrono::steady_clock::now();
  pcl::GASDEstimation<pcl::PointXYZ, pcl::GASDSignature512> gasd;
  gasd.setInputCloud (scene);
  // Output datasets
  pcl::PointCloud<pcl::GASDSignature512> descriptor;
  // Compute the descriptor
  gasd.compute (descriptor);
  // Get the alignment transform
  Eigen::Matrix4f gasdtrans = gasd.getTransform ();
  pcl::transformPointCloud(*scene, *scene, gasdtrans);
  //find bool voxel map for scene
  double voxel_size=0.1;
  VoxelMap Vmap(voxel_size);
  Vmap.setInputCloud(scene);
  Vmap.setBoolThreshold(10);
  bool** boolMapS=Vmap.getBoolMap();
  int mapWidthS, mapHeightS;
  pcl::PointXYZ min_ptS;
  pcl::PointXYZ max_ptS;
  Vmap.getWHMinMax(mapWidthS,mapHeightS,min_ptS,max_ptS);
  
  //find bool Map for object (cad model), it can be done outside the program
  Vmap.setInputCloud(object);
  Vmap.setBoolThreshold(10);
  int mapWidthO, mapHeightO;
  pcl::PointXYZ min_ptO;
  pcl::PointXYZ max_ptO;
  bool** boolMapO=Vmap.getBoolMap();
  Vmap.getWHMinMax(mapWidthO,mapHeightO,min_ptO,max_ptO);
  //find match by boolMap, return the position
  std::vector<int> winMapLoc=findMapLoc(boolMapS, mapHeightS, mapWidthS, boolMapO, mapHeightO, mapWidthO);
  //std::cout<<"Window detected at ["<<winMapLoc[0]<<", "<<winMapLoc[1]<<"]\n";
  //find the transform matrix from object to map location
  Eigen::Matrix4f booltrans=Eigen::Matrix4f::Identity();
  booltrans(0,3)=min_ptS.x+voxel_size*winMapLoc[0]-min_ptO.x;
  booltrans(1,3)=min_ptS.y+voxel_size*winMapLoc[1]-min_ptO.y;

  pcl::transformPointCloud (*object,*object,booltrans);
  end = std::chrono::steady_clock::now();
  std::cout << "Time difference for Voxel = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  
  //output for test icp gpu
  //pcl::PCDWriter writer;
  //writer.write ("object.pcd", *object, false);
  //writer.write ("scene.pcd", *scene, false);
  //crop the scene according to the voxel map result, leave the crop box larger
  double scal_ratio=0.3;
  min_ptS.x=min_ptS.x+voxel_size*(winMapLoc[0]-mapWidthO*scal_ratio/2.0);
  min_ptS.y=min_ptS.y+voxel_size*(winMapLoc[1]-mapHeightO*scal_ratio/2.0);
  max_ptS.x=min_ptS.x+voxel_size*(mapWidthO+mapWidthO*scal_ratio);
  max_ptS.y=min_ptS.y+voxel_size*(mapHeightO+mapHeightO*scal_ratio);
  pcl::CropBox<pcl::PointXYZ> cropBoxFilter (true);
    cropBoxFilter.setInputCloud (scene);
    cropBoxFilter.setMin (min_ptS.getVector4fMap());
    cropBoxFilter.setMax (max_ptS.getVector4fMap());
    cropBoxFilter.filter (*scene);

//========================= use ICP to find the window location =======================
  //downsample and icp
  pcl::PointCloud<pcl::PointXYZ>::Ptr object_aligned_dsp (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_aligned_dsp (new pcl::PointCloud<pcl::PointXYZ>);
  //try different leafsize
  //float leafvector[10]={0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.01,0.02};
  //for (int i=0;i<10;i++)
  //

      begin = std::chrono::steady_clock::now();
      //std::cout<<"leaf size is : "<<leafvector[i]<<std::endl;
  //Downsample for ICP registration as 0.5cm
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  float leaficp = 0.005f;
  grid.setLeafSize (leaficp, leaficp, leaficp);
  grid.setInputCloud (object);
  grid.filter (*object_aligned_dsp);
  grid.setInputCloud (scene);
  grid.filter (*scene_aligned_dsp);

 //output for test icp gpu
  pcl::PCDWriter writer;
  writer.write ("object_dsp.pcd", *object_aligned_dsp, false);
  writer.write ("scene_dsp.pcd", *scene_aligned_dsp, false);

  ICP_OUT icp_result;
  icp_result=icp(object_aligned_dsp,scene_aligned_dsp,100,0.00000001,0.9);
  Eigen::Matrix4f icptrans=icp_result.trans;
  //visualize the GPU result
  /*
  icptrans<<0.999578, -0.00162628, 0.0289634, 0.0277311,
0.00166881, 0.999997, -0.00144424, -0.0296341,
-0.0289607, 0.00149207, 0.999577, 0.0513276,
0, 0, 0, 1;*/





  std::cout<<"ICP transformation: "<<std::endl<<icptrans<<std::endl;
  Eigen::Matrix4f obj_scn_trans=gasdtrans.inverse()*icptrans*booltrans;
  //std::cout<<trans<<std::endl;
  
  end = std::chrono::steady_clock::now();
  std::cout << "Time difference for ICP = " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin ).count() << "[ms]" << std::endl;
  
  //std::cout<<"Final transformation matrix from model to scene is:\n"<<obj_scn_trans<<std::endl;
  float rotAngy=-3.1415926536/6.0;
  Eigen::Matrix4f rotMaty= Eigen::Matrix4f::Identity();
rotMaty(0,0) = cos(rotAngy);
rotMaty(0,1) = -sin(rotAngy);
rotMaty(1,0) = sin(rotAngy);
rotMaty(1,1) = cos(rotAngy);
Eigen::Vector4f origin(0.025,0.065,0,1);
Eigen::Vector4f o_trans=rotMaty*obj_scn_trans*rotMatrix1*origin;
std::cout<<o_trans<<std::endl;
float error=sqrt((1.5-o_trans(0))*(1.5-o_trans(0))+(2.0-o_trans(1))*(2.0-o_trans(1))+(-0.5-o_trans(2))*(-0.5-o_trans(2)));
std::cout<<"Origin error is: "<<error<<"m"<< std::endl;
  //}

pcl::transformPointCloud (*object,*object,icptrans);

  pcl::visualization::PCLVisualizer visu("gasd");
    //visu.addPointCloud (object, ColorHandlerT (object, 255, 255.0, 0.0), "object");
    visu.addPointCloud (object_aligned_dsp, ColorHandlerT (object_aligned_dsp, 0.0, 0.0, 255.0), "object_aligned");
    //visu.addPointCloud (scene, ColorHandlerT (scene, 0, 255.0, 0.0), "scene");
    visu.addPointCloud (scene_aligned_dsp, ColorHandlerT (scene_aligned_dsp, 255.0, 0.0,0), "scene_aligned");
    //visu.addPointCloud (scene, ColorHandlerT (scene, 255.0, 0.0,0), "scene");
    visu.addCoordinateSystem (1.0);
    visu.spin ();
    
    
    

    return 0;
}

//========================back up code===========================
/*
//calculate transformation to XZ plane and do transformation
Eigen::Matrix4f transformXZ(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud)
{
  pcl::PCA<pcl::PointXYZ> pca;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pca_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pca.setInputCloud(incloud);
  pca.project(*incloud,*pca_cloud);
  Eigen::Vector3f PCA_Norm=pca.getEigenVectors().col(2);
  Eigen::Vector3f PCA_Mean=pca.getMean().head<3>();
  //check if the normal direction is pointing far away from origin,if dot<0, invert the normal vector
  if (PCA_Mean.dot(PCA_Norm)< 0) PCA_Norm=-PCA_Norm;
  //suppose the cloud is kind of aligned to z axis (no rotation in y direction, the camera is not inclined left or right)
  //translate to centroid, rotate along z axis, rotate along x axis
  Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
  transMat.block<3,1>(0,3) = -PCA_Mean;//model centroid to origin
  //rotate z axis
  float x=PCA_Norm(0);
  float y=PCA_Norm(1);
  float x2y2=std::sqrt(x*x+y*y);
  float sinz=x/x2y2;
  float cosz=y/x2y2;
  Eigen::Matrix4f rotMat1;rotMat1<<cosz,-sinz,0,0,sinz,cosz,0,0,0,0,1,0,0,0,0,1;// z axis rotation
  //rotate x axis
  float z=PCA_Norm(2);
  float sinx=-z;
  float cosx=x2y2;
  Eigen::Matrix4f rotMat2;rotMat2<<1,0,0,0,0,cosx,-sinx,0,0,sinx,cosx,0,0,0,0,1;// x axis rotation
  pcl::transformPointCloud(*incloud, *incloud,rotMat2*rotMat1*transMat);
  return rotMat2*rotMat1*transMat;
}*/