 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
#include "warpage.hpp"

void warp::warpage::calcLine(pcl::ModelCoefficients::Ptr coefsOfPlane1, pcl::ModelCoefficients::Ptr coefsOfPlane2, pcl::ModelCoefficients::Ptr coefsOfLine)
{
    pcl::ModelCoefficients temcoefs;
    double a1, b1, c1, d1, a2,b2, c2, d2;
    double tempy, tempz;
    a1= coefsOfPlane1->values[0];
    b1= coefsOfPlane1->values[1];
    c1= coefsOfPlane1->values[2];
    d1= coefsOfPlane1->values[3];
    a2= coefsOfPlane2->values[0];
    b2= coefsOfPlane2->values[1];
    c2= coefsOfPlane2->values[2];
    d2= coefsOfPlane2->values[3];
    tempz= -(d1 / b1 - d2 / b2) / (c1 / b1 - c2 / b2);
    tempy= (-c1 / b1)*tempz - d1 / b1;
    coefsOfLine->values.push_back(0.0);
    coefsOfLine->values.push_back(tempy);
    coefsOfLine->values.push_back(tempz);
    coefsOfLine->values.push_back(b1*c2 - c1*b2);
    coefsOfLine->values.push_back(c1*a2 - a1*c2);
    coefsOfLine->values.push_back(a1*b2 - b1*a2);
}
pcl::visualization::PCLVisualizer visu1("wholemosaic");
void warp::warpage::voxel_edge_detect(Eigen::Vector4f inmin_pt,Eigen::Vector4f inmax_pt,int num_mosaic,bool visual, int edge_expect)
{
    
    pcl::CropBox<PointNT> cropBoxFilter (true);
    cropBoxFilter.setInputCloud (surf_cluster);

    // Cropbox
    cropBoxFilter.setMin (inmin_pt);
    cropBoxFilter.setMax (inmax_pt);

    // Cloud
    PointCloudT::Ptr crop_result (new PointCloudT);
    cropBoxFilter.filter (*crop_result); 
    //visu1.addPointCloud (crop_result, ColorHandlerT (crop_result, 255.0, 255.0, 255.0), "cloud");

    //mosaic result
    PointCloudT::Ptr mosaic_result (new PointCloudT);

    double bottom=0;
    double middle=0;
    double top=0;
    double x0,y0,z0,vx,vy,vz;

    std::vector<double> x_values, y_values, coeff;//for cubic edge fitting

    for(int i=0;i<num_mosaic;i++)
    {
        //corp mosaic blocks according to given min and max point and the orientation of edge
        cropBoxFilter.setInputCloud (surf_cluster);
        Eigen::Vector4f min_pt;
        Eigen::Vector4f max_pt;
        if(edge_expect==1||edge_expect==2)//vertical
        {
            min_pt << inmin_pt(0), inmin_pt(1)+(inmax_pt(1)-inmin_pt(1))/num_mosaic*i, inmin_pt(2), 1.0f;
            max_pt << inmax_pt(0), inmin_pt(1)+(inmax_pt(1)-inmin_pt(1))/num_mosaic*(i+1), inmax_pt(2), 1.0f;
        }
        else if(edge_expect==3||edge_expect==4)//horizontal
        {
            min_pt << inmin_pt(0)+(inmax_pt(0)-inmin_pt(0))/num_mosaic*i, inmin_pt(1), inmin_pt(2), 1.0f;
            max_pt << inmin_pt(0)+(inmax_pt(0)-inmin_pt(0))/num_mosaic*(i+1),inmax_pt(1),  inmax_pt(2), 1.0f;
        }
        else {std::cout<<"wrong input for edge position edpectation, please enter a number between 1 to 4"<<std::endl;break;}
    
        cropBoxFilter.setMin (min_pt);
        cropBoxFilter.setMax (max_pt);   
        cropBoxFilter.filter (*mosaic_result);
        int npoints=mosaic_result->points.size();
        if(i%2==0) visu1.addPointCloud (mosaic_result, ColorHandlerT (mosaic_result, 0.0, 255.0, 0.0), "surf"+std::to_string(min_pt(0))+std::to_string(min_pt(1))+std::to_string(min_pt(2)));
        else visu1.addPointCloud (mosaic_result, ColorHandlerT (mosaic_result, 255.0, 0.0, 0.0), "surf"+std::to_string(min_pt(0))+std::to_string(min_pt(1))+std::to_string(min_pt(2)));
        //visu1.spin();


        //set up ransac plane detector
        pcl::ModelCoefficients::Ptr coefficients1(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients);
        pcl::ModelCoefficients::Ptr coefsOfLine(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<PointNT> seg;
        pcl::ExtractIndices<PointNT> extract;
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.003);
        //std::cout<<mosaic_result->points.size()<<std::endl;

        //ransac for first plane
        seg.setInputCloud (mosaic_result);
        seg.segment (*inliers, *coefficients1);
        //std::cout<<inliers->indices.size()<<std::endl;

        //define the end points of the edge segment
        PointNT p1,p2;

        //if the second plane exist,ransac for second plane
        if (npoints-inliers->indices.size() > npoints/5)
        {
            //delete first plane
            extract.setInputCloud(mosaic_result);
            extract.setIndices(inliers);
            extract.setNegative(true);
            PointCloudT cloudF;
            extract.filter(cloudF);
            mosaic_result->swap(cloudF);

            seg.setInputCloud (mosaic_result);
            seg.segment (*inliers, *coefficients2);
            //std::cout<<mosaic_result->points.size()<<std::endl;

            //calculate the intersection line parameters
            calcLine(coefficients1,coefficients2,coefsOfLine);
            x0=coefsOfLine->values[0];
            y0=coefsOfLine->values[1];
            z0=coefsOfLine->values[2];
            vx=coefsOfLine->values[3];
            vy=coefsOfLine->values[4];
            vz=coefsOfLine->values[5];
            }

        //else the second plane doesn't exist
        else
        {
            
            //extract the first and only plane
            extract.setInputCloud(mosaic_result);
            extract.setIndices(inliers);
            extract.setNegative(false);
            PointCloudT::Ptr plane_cloud(new PointCloudT);
            extract.filter(*plane_cloud);

            //concave hull method to find the boundary points
            PointCloudT::Ptr cloud_hull (new PointCloudT);
            pcl::ConcaveHull<PointNT> chull;
            chull.setInputCloud (plane_cloud);
            chull.setAlpha (0.05);
            chull.setKeepInformation(true);
            chull.reconstruct (*cloud_hull);

            //std::cout<<"hull size:"<<cloud_hull->points.size()<<std::endl;
            //visu1.addPointCloud<PointNT> (cloud_hull, "hull cloud"+std::to_string(i));
            //visu1.spin();

            //use kd tree to find the outer edge points
            pcl::KdTreeFLANN<PointNT> kdtree;
            kdtree.setInputCloud (cloud_hull);
            int K = cloud_hull->points.size()/4;//set the size to be 1/4 of the total size to detect one edge (usually points on the lateral side is more than 1/4)
            
            //set search point at the middle of the voxel boundary, now only consider vertical edges
            PointNT searchPoint;
            //set search points according to expectation of edge position, 1 left, 2 right, 3 up, 4 down
            if (edge_expect==1)
            {
                searchPoint.x= max_pt(0);
                searchPoint.y= (min_pt(1)+max_pt(1))/2.0;
                searchPoint.z= (min_pt(2)+max_pt(2))/2.0;
            }
            else if (edge_expect==2)
            {
                searchPoint.x= min_pt(0);
                searchPoint.y= (min_pt(1)+max_pt(1))/2.0;
                searchPoint.z= (min_pt(2)+max_pt(2))/2.0;
            }
            else if (edge_expect==3)
            {
                searchPoint.x= (min_pt(0)+max_pt(0))/2.0;
                searchPoint.y= max_pt(1);
                searchPoint.z= (min_pt(2)+max_pt(2))/2.0;
            }
            else 
            {
                searchPoint.x= (min_pt(0)+max_pt(0))/2.0;
                searchPoint.y= min_pt(1);
                searchPoint.z= (min_pt(2)+max_pt(2))/2.0;
            }
        
        
            PointCloudT::Ptr tmp_pca_cloud (new PointCloudT);
            PointCloudT::Ptr tmp_pca_cloud_after (new PointCloudT);
            tmp_pca_cloud->width  = K;
            tmp_pca_cloud->height = 1;
            tmp_pca_cloud->points.resize (tmp_pca_cloud->width * tmp_pca_cloud->height);

            //search for K nearest points
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                for (std::size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
                {
                tmp_pca_cloud->points[i]=cloud_hull->points[pointIdxNKNSearch[i]];
                
                //for cubic fit
                x_values.push_back(cloud_hull->points[pointIdxNKNSearch[i]].x);
                y_values.push_back(cloud_hull->points[pointIdxNKNSearch[i]].y);
                }
            }

            //use pca to find the edge line
            pcl::PCA<PointNT> pca;
            pca.setInputCloud(tmp_pca_cloud);
            pca.project(*tmp_pca_cloud,*tmp_pca_cloud_after);
            Eigen::Vector3f tmp_PCA_Mean=pca.getMean().head<3>();//mean value of edge
            Eigen::Vector3f tmp_v=pca.getEigenVectors().col(0);//direction vector of edge
            x0=tmp_PCA_Mean(0);
            y0=tmp_PCA_Mean(1);
            z0=tmp_PCA_Mean(2);
            vx=tmp_v(0);
            vy=tmp_v(1);
            vz=tmp_v(2);
        }

        //calculate the intersection segment
        if (edge_expect==1||edge_expect==2){
            p1.y=min_pt(1);
            p1.x=(p1.y-y0)/vy*vx+x0;
            p1.z=(p1.y-y0)/vy*vz+z0;
            p2.y=max_pt(1);
            p2.x=(p2.y-y0)/vy*vx+x0;
            p2.z=(p2.y-y0)/vy*vz+z0;
            //record the horizontal error
            if (i==0)bottom=p1.x;
            if (i==num_mosaic/2)middle=p1.x;
            if (i==num_mosaic-1)top=p2.x;
        }
        else {
            p1.x=min_pt(0);
            p1.y=(p1.x-x0)/vx*vy+y0;
            p1.z=(p1.x-x0)/vx*vz+z0;
            p2.x=max_pt(0);
            p2.y=(p2.x-x0)/vx*vy+y0;
            p2.z=(p2.x-x0)/vx*vz+z0;
            //record the vertical error
            if (i==0)bottom=p1.y;
            if (i==num_mosaic/2)middle=p1.y;
            if (i==num_mosaic-1)top=p2.y;
        }

    
        //std::cout<<"p1"<<p1.x<<" "<<p1.y<<" "<<p1.z<<std::endl;
        //std::cout<<"p2"<<p2.x<<" "<<p2.y<<" "<<p2.z<<std::endl;

        
        if(visual){
        visu1.addLine<PointNT> (p1,p2,255,0,0, "line"+std::to_string(min_pt(0))+std::to_string(min_pt(1))+std::to_string(min_pt(2)));
        visu1.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, "line"+std::to_string(min_pt(0))+std::to_string(min_pt(1))+std::to_string(min_pt(2)));
        }
    }
    double error1=top-middle;
    double error2=bottom-middle;
    std::cout<<"top(left)-middle warpage result is "<<error1<<" m"<<std::endl;
     std::cout<<"bottom(right)-middle warpage result is "<<error2<<" m"<<std::endl;


    //for cubic fit
    if(x_values.size()>=4)//fitting exist
        {
        polyfit(y_values, x_values, coeff, 3);
        printf("%f + %f*y + %f*y^2 + %f*y^3\n", coeff[0], coeff[1], coeff[2], coeff[3]);}

    if(visual){visu1.spin (); }//visu1.addCoordinateSystem(1.0);
}
void warp::warpage::setInputCloud(PointCloudT::Ptr cloud){
    surf_cluster = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ>);
    surf_cluster->width = cloud->width;
    surf_cluster->height = cloud->height;
    surf_cluster->points.resize(surf_cluster->width * surf_cluster->height);
    pcl::copyPointCloud(*cloud,*surf_cluster);
}
void warp::warpage::getWarpage(bool visual){

    Eigen::Vector4f min_pt;
    Eigen::Vector4f max_pt;
    min_pt << 0.075f, 0.12f, 0.02f, 1.0f;
    max_pt << 0.125f, 1.18f, 0.04f, 1.0f;
    std::cout<<"edge 1"<<std::endl;
    voxel_edge_detect( min_pt,max_pt,num_mosaic,visual,1);//last parameter is where you expect the edge lies in the cluster,1 left, 2 right, 3 up, 4 down

    min_pt << -0.03f, 0.04f, -0.02f, 1.0f;
    max_pt << 0.03f, 1.28f, 0.02f, 1.0f;
    std::cout<<"edge 2"<<std::endl;
    voxel_edge_detect(min_pt,max_pt,num_mosaic,visual,2);

    min_pt << 0.525f, 0.12f, 0.02f, 1.0f;
    max_pt << 0.575f, 1.18f, 0.04f, 1.0f;
    voxel_edge_detect(min_pt,max_pt,num_mosaic,visual,2);

    min_pt << 0.63f, 0.04f, -0.02f, 1.0f;
    max_pt << 0.67f, 1.28f, 0.02f, 1.0f;
    voxel_edge_detect(min_pt,max_pt,num_mosaic,visual,1);

    min_pt << 0.04f, 0.01f, -0.02f, 1.0f;
    max_pt << 0.61f, 0.05f, 0.02f, 1.0f;
    voxel_edge_detect(min_pt,max_pt,num_mosaic/2,visual,4);

    min_pt << 0.125f, 0.08f, 0.02f, 1.0f;
    max_pt << 0.525f, 0.12f, 0.04f, 1.0f;
    voxel_edge_detect(min_pt,max_pt,num_mosaic/2,visual,3);

    min_pt << 0.04f, 1.275f, -0.02f, 1.0f;
    max_pt << 0.61f, 1.325f, 0.02f, 1.0f;
    voxel_edge_detect(min_pt,max_pt,num_mosaic/2,visual,3);

    min_pt << 0.125f, 1.075f, 0.02f, 1.0f;
    max_pt << 0.525f, 1.225f, 0.04f, 1.0f;
    voxel_edge_detect(min_pt,max_pt,num_mosaic/2,visual,4);
}