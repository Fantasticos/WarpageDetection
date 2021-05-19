 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
 
#ifndef TESTICP_H
#define TESTICP_H

#include "Eigen/Eigen"
#include <vector>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
// try pytorch to accelerate matrix calculation
//#include <torch/torch.h>
using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;

namespace testicp{
typedef struct{
    Eigen::Matrix4f trans;
    std::vector<float> distances;
    int iter;
}  ICP_OUT;
typedef struct{
    std::vector<float> distances;
    std::vector<int> indices;
} NEIGHBOR;

//Eigen::MatrixXf pcl_to_matrix(PointCloudPtr cloud);
inline Eigen::Matrix4f best_fit_transform(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B);
inline ICP_OUT icp(PointCloudPtr cloudA, PointCloudPtr cloudB, int max_iterations=20, double tolerance = 0.001,double overlap=0.8);
// throughout method
inline NEIGHBOR nearest_neighbor(const Eigen::MatrixXf &src, pcl::KdTreeFLANN<PointT> kdtree);




inline Eigen::Matrix4f best_fit_transform(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B){
    /*
    Notice:
    1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
    2/ matrix type 'MatrixXf' or 'MatrixXf' matters.
    */
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    Eigen::Matrix4f T = Eigen::MatrixXf::Identity(4,4);
    Eigen::Vector3f centroid_A=A.colwise().mean();
    Eigen::Vector3f centroid_B=B.colwise().mean();
    Eigen::MatrixXf AA = A.rowwise()-centroid_A.transpose();
    Eigen::MatrixXf BB = B.rowwise()-centroid_B.transpose();
    int row = A.rows();
/*
    for(int i=0; i<row; i++){
        centroid_A += A.block<1,3>(i,0).transpose();
        centroid_B += B.block<1,3>(i,0).transpose();
    }
    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Time difference for icp centroid calculation = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    //begin = std::chrono::steady_clock::now();
    centroid_A /= row;
    centroid_B /= row;
    
    for(int i=0; i<row; i++){
        AA.block<1,3>(i,0) = A.block<1,3>(i,0) - centroid_A.transpose();
        BB.block<1,3>(i,0) = B.block<1,3>(i,0) - centroid_B.transpose();
    }
    */
    //end = std::chrono::steady_clock::now();
    //std::cout << "Time difference for icp centroid adding= " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    //begin = std::chrono::steady_clock::now();
    Eigen::MatrixXf H = AA.transpose()*BB;
    Eigen::MatrixXf U;
    Eigen::VectorXf S;
    Eigen::MatrixXf V;
    Eigen::MatrixXf Vt;
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    //end = std::chrono::steady_clock::now();
    //std::cout << "Time difference for icp matrix preparation= " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    // std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    JacobiSVD<Eigen::MatrixXf> svd(H, ComputeFullU | ComputeFullV);
    U = svd.matrixU();
    //S = svd.singularValues();
    V = svd.matrixV();
    Vt = V.transpose();
    //std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    //std::cout << "Time difference for icp SVD= " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count() << "[us]" << std::endl;

    R = Vt.transpose()*U.transpose();

    if (R.determinant() < 0 ){
        Vt.block<1,3>(2,0) *= -1;
        R = Vt.transpose()*U.transpose();
    }

    t = centroid_B - R*centroid_A;

    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    return T;

}
inline ICP_OUT icp(PointCloudPtr cloudA, PointCloudPtr cloudB, int max_iterations, double tolerance, double overlap){
    //load Eigen::Matrix A and B from pcl::pointcloud
    Eigen::MatrixXf A=(cloudA->getMatrixXfMap(3,4,0)).transpose();//3,4,0 is dimension,stride,and offset for pointxyz type
    Eigen::MatrixXf B=(cloudB->getMatrixXfMap(3,4,0)).transpose();
    PointCloudPtr srccloud(new PointCloud);
    //pcl::copyPointCloud(*cloudA, *srccloud);//copy cloudA, all transformation will perform on A
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud (cloudB);

    
    int row = A.rows();
    int rowb = B.rows();
    int rowoverlap=row*overlap;
    std::cout<<row<<" "<<rowoverlap<<std::endl;
    Eigen::MatrixXf src = Eigen::MatrixXf::Ones(3+1,row);
    Eigen::MatrixXf src3f = Eigen::MatrixXf::Ones(3,row);
    Eigen::MatrixXf dst = Eigen::MatrixXf::Ones(3+1,rowb);
    

    NEIGHBOR neighbor;
    Eigen::Matrix4f T;
    Eigen::MatrixXf dst_chorder = Eigen::MatrixXf::Ones(3,rowoverlap);
    Eigen::MatrixXf src3fover = Eigen::MatrixXf::Ones(3,rowoverlap);

    std::vector<float> distancestmp;
    std::vector<int> indicestmp;

    ICP_OUT result;
    int iter = 0;
    //std::cout<<"rows: "<<row<<std::endl;
    //std::cout<<"loading matrix to loop"<<std::endl;
    for (int i = 0; i<row; i++){
        src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
        src3f.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
    }
    for (int i = 0; i<rowb; i++){
    dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
    }
    //std::cout<<"loading finished"<<std::endl;
    double prev_error = 0;
    double mean_error = 0;
    //Prepare file to write distance 
    //FILE * fp;
    //if((fp=fopen("dist10.txt","wb"))==NULL)
     //           {
     //               printf("cannot open the file");
      //          }

    for (int i=0; i<max_iterations; i++)
    {
        //printf("Iteration [%d/%d] start\n",i+1,max_iterations);
        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        neighbor = nearest_neighbor(src3f.transpose(),kdtree);
        //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time difference for icp neighbour matching = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    //begin = std::chrono::steady_clock::now();
        //sort the neighbors by distance and take those within the overlap ratio
        //sort by distance
        std::vector<std::size_t> indextmp (neighbor.indices.size());
        std::iota(indextmp.begin(), indextmp.end(), 0);
        std::sort(indextmp.begin(), indextmp.end(), [&](size_t a, size_t b) { return neighbor.distances[a] < neighbor.distances[b]; });
        distancestmp.clear();indicestmp.clear();//clean up the two vectors
        //end = std::chrono::steady_clock::now();
        //std::cout << "Time difference for icp neighbour sorting = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    //begin = std::chrono::steady_clock::now();
        
                
           
        for (int j=0;j<row;j++) { 
            distancestmp.push_back(neighbor.distances[indextmp[j]]);
            indicestmp.push_back(neighbor.indices[indextmp[j]]);
            //if(i==10)//save data once for histogram
            //{
            //fprintf(fp,"%f\n",neighbor.distances[indextmp[j]]);
            //}
        } 
        neighbor.distances=distancestmp;
        neighbor.indices=indicestmp;
        
        //remove distance points from src3f to src3fover
        for (int j=0;j<rowoverlap;j++) { 
            src3fover.block<3,1>(0,j)=src3f.block<3,1>(0,indextmp[j]);
        }
        //remove distance points from dst to 
        for(int j=0; j<rowoverlap; j++){
            dst_chorder.block<3,1>(0,j) = dst.block<3,1>(0,neighbor.indices[j]);
        }
        
        
        //end = std::chrono::steady_clock::now();
        //std::cout << "Time difference for icp neighbour processing = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    
        //find the best transform according to these two paired cloud
        T = best_fit_transform(src3fover.transpose(),dst_chorder.transpose());
        //printf("Best Transform found\n");
        
        //end = std::chrono::steady_clock::now();
        //std::cout << "Time difference for icp transform calculation = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[us]" << std::endl;
    
        src = T*src;
        for(int j=0; j<row; j++){
            src3f.block<3,1>(0,j) = src.block<3,1>(0,j);
        }

        mean_error = std::accumulate(neighbor.distances.begin(),neighbor.distances.end(),0.0)/neighbor.distances.size();
        //printf("Mean error get\n");
        if (abs(prev_error - mean_error) < tolerance){
            printf("tolerance achieved at error %f\n",mean_error);
            break;
        }
        prev_error = mean_error;
        iter = i+2;
    }

    T = best_fit_transform(A,src3f.transpose());
    result.trans = T;
    result.distances = neighbor.distances;
    result.iter = iter;

    return result;
}


inline NEIGHBOR nearest_neighbor(const Eigen::MatrixXf &src, pcl::KdTreeFLANN<PointT> kdtree)
{
  NEIGHBOR neigh;
  int totalcount = src.rows();
  int K = 1; 
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (int i = 0; i < totalcount; ++i)
    {
        PointT p;
        p.x=src(i,0);p.y=src(i,1);p.z=src(i,2);
        kdtree.nearestKSearch (p, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        neigh.distances.push_back(pointNKNSquaredDistance[0]);
        //std::cout<<pointNKNSquaredDistance[0]<<std::endl;
        neigh.indices.push_back(pointIdxNKNSearch[0]);
    }
  return neigh;
}
}







#endif

