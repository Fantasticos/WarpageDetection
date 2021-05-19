 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
#include "registration.hpp"

bool** reg::VoxelMap::getBoolMap(){
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
std::vector<int> reg::VoxelMap::findMapLoc(bool** Map1, int h1, int w1, bool** Map2, int h2, int w2)
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

void reg::registration::setScan(pcl::PointCloud<pcl::PointXYZ>::Ptr scan){//set scanned cloud
    scene = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    scene->width = scan->width;
    scene->height = scan->height;
    scene->points.resize(scene->width * scene->height);
    pcl::copyPointCloud(*scan,*scene);
}

void reg::registration::setModel(pcl::PointCloud<pcl::PointXYZ>::Ptr model){//set model cloud
    object = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    object->width = model->width;
    object->height = model->height;
    object->points.resize(object->width * object->height);
    
    pcl::copyPointCloud(*model,*object);
}

Eigen::Matrix4f reg::registration::getTransform(){
    //========================= use voxel map to find the window===========================
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_PCA (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(scene);
    pca.project(*scene,*cloud_PCA);
    Eigen::Vector3f PCA_Norm=pca.getEigenVectors().col(2);
    Eigen::Vector3f PCA_Mean=pca.getMean().head<3>();
    //check if the normal direction is pointing at origin,if dot>0, invert the normal vector
    //[IMPORTANT] this is assuming the window is facing the scanner
    if (PCA_Mean.dot(PCA_Norm)>0) PCA_Norm=-PCA_Norm;

    //rotate the scan to let it face downward at -Z direction
    //[IMPORTANT] here the scan is supposed to be Z-upright initially
    float x=PCA_Norm(0);
    float y=PCA_Norm(1);
    float sin=x/std::sqrt(x*x+y*y);
    float cos=y/std::sqrt(x*x+y*y);
    Eigen::Matrix4f transMatrix=Eigen::Matrix4f::Identity();
    transMatrix.block<3,1>(0,3) = -PCA_Mean;
    Eigen::Matrix4f rotMatrix1;rotMatrix1<<cos,-sin,0,0,sin,cos,0,0,0,0,1,0,0,0,0,1;// z axis rotation
    Eigen::Matrix4f rotMatrix2;rotMatrix2<<1,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,1;// x axis rotation -90 degree
    Eigen::Matrix4f initrans=rotMatrix2*rotMatrix1*transMatrix;
    pcl::transformPointCloud(*scene, *scene, initrans);
    //find bool voxel map for scene
    double voxel_size=0.1;
    Vmap.setVoxelSize(voxel_size);//set the voxel size
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
    std::vector<int> winMapLoc=Vmap.findMapLoc(boolMapS, mapHeightS, mapWidthS, boolMapO, mapHeightO, mapWidthO);

    //find the transform matrix from object to map location
    Eigen::Matrix4f booltrans=Eigen::Matrix4f::Identity();
    booltrans(0,3)=min_ptS.x+voxel_size*winMapLoc[0]-min_ptO.x;
    booltrans(1,3)=min_ptS.y+voxel_size*winMapLoc[1]-min_ptO.y;

    pcl::transformPointCloud (*object,*object,booltrans);
    
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
    

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //Downsample for ICP registration as 0.5cm
    pcl::VoxelGrid<pcl::PointXYZ> grid;
    float leaficp = 0.005f;
    grid.setLeafSize (leaficp, leaficp, leaficp);
    grid.setInputCloud (object);
    grid.filter (*object_aligned_dsp);
    grid.setInputCloud (scene);
    grid.filter (*scene_aligned_dsp);

    //output for test icp gpu
    //pcl::PCDWriter writer;
    //writer.write ("object_dsp.pcd", *object_aligned_dsp, false);
    //writer.write ("scene_dsp.pcd", *scene_aligned_dsp, false);

    testicp::ICP_OUT icp_result;
    icp_result=testicp::icp(object_aligned_dsp,scene_aligned_dsp,2000,0.00000001,0.9);
    Eigen::Matrix4f icptrans=icp_result.trans;

    std::cout<<"ICP transformation: "<<std::endl<<icptrans<<std::endl;
    Eigen::Matrix4f obj_scn_trans=initrans.inverse()*icptrans*booltrans;
    //std::cout<<trans<<std::endl;
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference for ICP = " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin ).count() << "[ms]" << std::endl;
    
    pcl::transformPointCloud (*object,*object,icptrans);
    return obj_scn_trans;
}

void reg::registration::visualize(){
    pcl::visualization::PCLVisualizer visu("registration");
    visu.addPointCloud (object, ColorHandlerT (object, 0.0, 0.0, 255.0), "object_aligned");
    visu.addPointCloud (scene, ColorHandlerT (scene, 255.0, 0.0,0), "scene_aligned");
    //visu.addCoordinateSystem (1.0);
    visu.spin ();
}