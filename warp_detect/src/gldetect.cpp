 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
#include "gldetect.hpp"

bool gld::gldetect::isEqual(const Vec4i& _l1, const Vec4i& _l2)
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

std::vector<Eigen::Vector4f> gld::gldetect::lsd(char* img_pth)
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
    /*
    //print all detected small line segments
    std::cout<<"line detected"<<lines.size()<<std::endl;
    for(size_t i=0;i<lines.size();i++)
    {
        cv::Vec4i l = lines[i];
        std::cout<<l<<std::endl;
        cv::line(image, cv::Point(l[0],l[1]),cv::Point(l[2],l[3]),cv::Scalar(0,0,255),3,8);
    }*/
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

std::vector<Eigen::Vector4f> gld::gldetect::lsd3d(pcl::PointCloud<pcl::PointXYZ>::Ptr scene, std::vector<Eigen::Vector4f> glnorms)
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
  return gl3ds;
}

std::vector<Eigen::Vector4f> gld::gldetect::get3DLines(char* img_pth,pcl::PointCloud<pcl::PointXYZ>::Ptr scene)
{
    std::vector<Eigen::Vector4f> glnorms=lsd(img_pth);//get green line plane norms from 2D image
    return lsd3d(scene,glnorms);
}
    