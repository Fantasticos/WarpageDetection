 //By Zhongyuan Liu 
 //CERLAB CMU
 //Mechanical Engineering Department
 
 #ifndef LINESEG2D_H_
 #define LINESEG2D_H_
 #include <pcl/point_types.h>
 #include <pcl/common/geometry.h>
 #include <Eigen/QR>
 #include <stdio.h>
 #include <vector>
 #include <iostream>
 #include <cmath>
 //#include <pcl/segmentation/segmentation.h>
 inline void polyfit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order)
{
	Eigen::MatrixXd A(xv.size(), order+1);
	Eigen::VectorXd yv_mapped = Eigen::VectorXd::Map(&yv.front(), yv.size());
	Eigen::VectorXd result;

	assert(xv.size() == yv.size());
	assert(xv.size() >= order+1);

	// create matrix
	for (size_t i = 0; i < xv.size(); i++)
	for (size_t j = 0; j < order+1; j++)
		A(i, j) = pow(xv.at(i), j);

	// solve for linear least squares fit
	result = A.householderQr().solve(yv_mapped);

	coeff.resize(order+1);
	for (size_t i = 0; i < order+1; i++)
		coeff[i] = result[i];
}

    class LineSeg2D
    {
        private:
        double mis_align_thres=0.002;                                   //threshold of misalignment
        int p_min=20;                                                   //minimum points in a line segment
        double l_min=0.2;                                               //minimum length of a line segment
        int seedsize=10;                                                //points contained in a seed
        std::vector<int> seed;                                          //store the indices of seeds
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;                      //input cloud
        std::vector<pcl::PointIndices::Ptr> indicesVector;              //output segment indices
        struct line;
        std::vector<line> segments;                                     //line segments with line struct
        bool point_inline(int p_ind, line l);                           //calculate distance between point and line according to l.a, l.b and l.c
        double point_line_dist(int p_ind, line l);
        line fit_line(int start_ind, int end_ind);                      //fit a line containing points from start_ind to end_ind
        void seed_detection();                                          //detect all seeds 
        void region_growing();
        void overlap_process();
		void seg_cluster();
        void regrow_segments();
		int num_edges;
		double line_line_inner_product(line l1, line l2);
		double line_line_dist(line l1, line l2);
        Eigen::Vector2d line_line_intersect(line l1, line l2);
        std::vector<Eigen::Vector2d> intersects;
        void cubic_fit();
        public:
        std::vector<pcl::PointIndices::Ptr> extractLineIndices();
        std::vector<Eigen::Vector3d> getSegParams();
        std::vector<double> angle_diagnose();
        std::vector<double> length_diagnose();
        std::vector<Eigen::Vector2d> getIntersectCoordinate();
		void setNumEdges(int x)
		{
			num_edges = x;
			if (x != 8 && x != 12)
			{
                std::cerr<<"Input edge number is neither 8 nor 12"<<std::endl;
				throw std::exception();
			}
		}
        LineSeg2D(){}
        LineSeg2D(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud)
        {
            cloud=inputcloud;
        }
        void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud)
        {
            cloud=inputcloud;
            std::cout<<"Clould loaded Successfully"<<std::endl;
        }
        void setParam(double m, int pmin, double lmin,int size)
        {
            mis_align_thres=m;
            p_min=pmin;
            l_min=lmin;
            seedsize=size;
            std::cout<<"Parameters loaded Successfully"<<std::endl;
        }
    };



struct LineSeg2D::line
{
    int start;//start index
    int end;//end index
    double length;//line length
    int size;//num of line points
    //line fitting model ax+by+c=0
    double a;
    double b;
    double c;
};

//Fit a line according to start index and end index
inline LineSeg2D::line LineSeg2D::fit_line(int start_ind, int end_ind)
{
    //fitting a line to y=ax+c, here b=-1, using simple linear fitting
    //start_ind<end_ind
    line l;
    l.start=start_ind;
    l.end=end_ind;
    // incase that the line is vertical
    double slope=std::abs((cloud->points[start_ind].y-cloud->points[end_ind].y)/(cloud->points[start_ind].x-cloud->points[end_ind].x));
    int n=end_ind-start_ind+1;
    if (slope<1)//fit as y=ax+c, where b=-1
    {
    l.b=-1;
    double xsum=0,x2sum=0,ysum=0,xysum=0;                //variables for sums/sigma of xi,yi,xi^2,xiyi etc
    for (int i=start_ind;i<=end_ind;i++)
    {
        xsum=xsum+cloud->points[i].x;                        //calculate sigma(xi)
        ysum=ysum+cloud->points[i].y;                        //calculate sigma(yi)
        x2sum=x2sum+(cloud->points[i].x)*(cloud->points[i].x);                //calculate sigma(x^2i)
        xysum=xysum+(cloud->points[i].x)*(cloud->points[i].y);                    //calculate sigma(xi*yi)
    }
    l.a=(n*xysum-xsum*ysum)/(n*x2sum-xsum*xsum);            //calculate slope
    l.c=(x2sum*ysum-xsum*xysum)/(x2sum*n-xsum*xsum);            //calculate intercept
    //l.length=std::sqrt(std::pow((cloud->points[start_ind].x-cloud->points[end_ind].x),2)+std::pow((cloud->points[start_ind].z-cloud->points[end_ind].z),2))
    }
    else//fit as a=-1, which is x=by+c
    {
    l.a=-1;
    double ysum=0,y2sum=0,xsum=0,xysum=0;                //variables for sums/sigma of xi,yi,xi^2,xiyi etc
    for (int i=start_ind;i<=end_ind;i++)
    {
        xsum=xsum+cloud->points[i].x;                        //calculate sigma(xi)
        ysum=ysum+cloud->points[i].y;                        //calculate sigma(yi)
        y2sum=y2sum+(cloud->points[i].y)*(cloud->points[i].y);                //calculate sigma(x^2i)
        xysum=xysum+(cloud->points[i].y)*(cloud->points[i].x);                    //calculate sigma(xi*yi)
    }
    l.b=(n*xysum-xsum*ysum)/(n*y2sum-ysum*ysum);            //calculate slope
    l.c=(y2sum*xsum-ysum*xysum)/(y2sum*n-ysum*ysum);            //calculate intercept
    }
    l.length=pcl::geometry::distance(cloud->points[start_ind],cloud->points[end_ind]);
    l.size=n;
    return l;
}

//Judge whether a point is in line according to mis_align_thres
inline bool LineSeg2D::point_inline(int p_ind, line l)
{
    return point_line_dist(p_ind,l) <mis_align_thres;
}

//calculate the distance between given point p and the line
inline double LineSeg2D::point_line_dist(int p_ind, line l)
{
    return std::abs(l.a*cloud->points[p_ind].x+l.b*cloud->points[p_ind].y+l.c)/std::sqrt(l.a*l.a+l.b*l.b);
}

//calculate the distance between two lines
inline double LineSeg2D::line_line_dist(line l1, line l2)
{
	double dist = 0;
	for (int i = 0; i < l1.length; i++)
	{
		dist += point_line_dist(l1.start + i, l2);
	}
	for (int i = 0; i < l2.length; i++)
	{
		dist += point_line_dist(l2.start + i, l1);
	}
	return dist / (l1.length + l2.length);
}
inline double LineSeg2D::line_line_inner_product(line l1, line l2)
{
	return std::abs(l1.a * l2.a + l1.b * l2.b)/std::sqrt(l1.a*l1.a+l1.b*l1.b)/ std::sqrt(l2.a * l2.a + l2.b * l2.b);
}

//Detect all the seeds in the plane, store the indices in vector<int> seed
inline void LineSeg2D::seed_detection()
{
    int N=cloud->points.size();
    for(int i=0;i<N-p_min;i++)
    {
        bool flag=1;
        line l=fit_line(i,i+seedsize-1);
        for (int j=i;j<i+seedsize;j++)
        {
            if(!point_inline(j,l)){flag=0;break;}
        }
        if (flag){seed.push_back(i);}
    }
    std::cout<< seed.size()<<"seeds detected from "<<cloud->points.size()<<"points"<<std::endl;
}

//do region growing and find all segments in the cloud
inline void LineSeg2D::region_growing()
{
    std::vector<line> tmpsegments;
    int N=cloud->points.size();
    for(int i=0; i<seed.size();i++)
    {
        int pf=seed[i]+seedsize-1; int pb=seed[i];
        line l=fit_line(pb,pf);
        pf++;
        do {
            if(pf>N-1)break;// when region growing beyond boundary,break loop
            else l=fit_line(pb,pf);
            pf++;
        }
        while (point_inline(pf,l));
        pf--;
        pb--;
        do {
            if(pb<0)break;// when region growing beyond boundary,break loop
            else l=fit_line(pb,pf);
            pb--;
        }
        while (point_inline(pb,l));
        pb++;
        if (l.length>=l_min&&l.size>=p_min)
        {
            
            if(tmpsegments.empty())tmpsegments.push_back(l);
            else //if the segment is not completely included by the former ones, push it back
            {
                //std::cout<<l.start<<"  "<<segments.back().start<<"  "<<l.end<<"  "<<segments.back().end<<std::endl;
                if(l.start<tmpsegments.back().start||l.end>tmpsegments.back().end){tmpsegments.push_back(l);std::cout<<l.start<<"  "<<l.end<<std::endl;}
            }
        }
        while(seed[i]<l.end)i++;
    }
    //remove the segments that are completely included by others
    
    for (int i=0; i<tmpsegments.size();i++)
    {
        bool is_included=0;
        for (int j=0; j<tmpsegments.size();j++)
        {
            if (tmpsegments[i].start>=tmpsegments[j].start&&tmpsegments[i].end<=tmpsegments[j].end&&i!=j)
            {
                is_included=1;
                std::cout<<i<<" included by "<<j<<std::endl;
                break;
            }
        }
        if(!is_included){segments.push_back(tmpsegments[i]);std::cout<<tmpsegments[i].start<<"  "<<tmpsegments[i].end<<std::endl;}
    }
    std::cout<< segments.size()<<"segments grown from "<<seed.size()<<"seeds"<<std::endl;
    
}


//std::vector<pcl::PointIndices::Ptr> LineSeg2D::extractLineIndices(){}
inline void LineSeg2D::overlap_process()
{
    for (int i=0;i<segments.size()-1;i++)
    {
        int m1=segments[i].start;
        int n1=segments[i].end;
        int m2=segments[i+1].start;
        int n2=segments[i+1].end;
        if (m2<m1)m2=m1;
        if (m2<=n1)
        {
            int k=m2;
            for(k=m2;k<=n1;k++)
            {
               double di= point_line_dist(k,segments[i]);
               double dj= point_line_dist(k,segments[i+1]);
               if(di<dj) continue;
               else break; 
            }
            n1=k-1;
            m2=k;
        }
        else break;
        segments[i]=fit_line(m1,n1);
        segments[i+1]=fit_line(m2,n2);
    }
    
    //for (int i=0;i<segments.size()-1;i++)
    //{
    //    std::cout<<segments[i].start<<"  "<<segments[i].end<<std::endl;
    //}
}
//cluster the segments
inline void LineSeg2D::seg_cluster()
{std::cout<<"segments num"<<segments.size()<<std::endl;
	for (int i = 0; i < segments.size(); i++)
	{
        int nextpos = (i + 1) % segments.size();//to avoid out bound
        std::cout<<"segment "<<i<<" and next "<<nextpos<<std::endl;
		if (line_line_inner_product(segments[i], segments[nextpos]) > 0.3 && line_line_dist(segments[i], segments[nextpos]) < 0.1)
		{
			segments[i]= fit_line(segments[i].start, segments[nextpos].end);
			segments.erase(segments.begin()+nextpos);
            std::cout<<"segment "<<nextpos<<" erased "<<std::endl;
            i--;
		}
	}
	
}
//regrow the segments so that every boundary point is used
inline void LineSeg2D::regrow_segments()
{
    if(num_edges==12||num_edges==8)
    {
        int nexti;
        for(int i=0;i<num_edges;i++)
        {
            if(i%4==3)nexti=i-3;
            else nexti=i+1;
            int m1=segments[i].start;
            int n1=segments[i].end;
            int m2=segments[nexti].start;
            int n2=segments[nexti].end;
            if(m2-n1>0)
            {
            int k=n1;
            for(k=n1;k<m2;k++)
            {
               double di= point_line_dist(k,segments[i]);
               double dj= point_line_dist(k,segments[nexti]);
               std::cout<<di<<"  "<<dj<<std::endl;
               if(di<dj) {
                std::cout<<"inline point"<<k<<"with edge "<<i<<std::endl;
                segments[i]=fit_line(m1,k);
                continue;}
               else break; 
            }
            n1=k-1;
            m2=k;
            segments[nexti]=fit_line(m2,n2);
            std::cout<<"new seperation "<<k-1<<"with edge "<<i<<std::endl;
            }
            else break;
            
        }
    }
}
inline std::vector<pcl::PointIndices::Ptr> LineSeg2D::extractLineIndices()
{
    seed_detection();
    region_growing();
	std::cout << "growing finished" << std::endl;
    overlap_process();
	seg_cluster();
    cubic_fit();
    //regrow_segments();
    for (int i=0;i<segments.size();i++)
    {
        pcl::PointIndices::Ptr tmp(new pcl::PointIndices);
        if(segments[i].size>p_min/2)
        {
            for (int j=segments[i].start;j<=segments[i].end;j++)
        {
            tmp->indices.push_back(j);
        }
        indicesVector.push_back(tmp);
        }
        //std::cout<<segments[i].start<<" "<<segments[i].end<<std::endl;
    }
    //cout<<indicesVector.size()<<std::endl;
    return indicesVector; 
}
inline std::vector<Eigen::Vector3d> LineSeg2D::getSegParams()
{
    std::vector<Eigen::Vector3d> lineparams;
    for(int i=0;i<indicesVector.size();i++)
    {
        Eigen::Vector3d tmp;
        tmp(0)=segments[i].a;
        tmp(1)=segments[i].b;
        tmp(2)=segments[i].c;
        lineparams.push_back(tmp);
        std::cout<<"segment "<<i<<":"<<std::endl<<tmp<<std::endl;
    }
    return lineparams;
}

inline Eigen::Vector2d LineSeg2D::line_line_intersect(line l1, line l2)
{
    Eigen::Vector2d intersect;
    intersect(0)=(l1.b*l2.c-l2.b*l1.c)/(l1.a*l2.b-l2.a*l1.b);
    intersect(1)=(l2.a*l1.c-l1.a*l2.c)/(l1.a*l2.b-l2.a*l1.b);
    return intersect;
}

inline std::vector<double> LineSeg2D::angle_diagnose()
{
    std::vector<double> angles;
    if(num_edges==12||num_edges==8)
    {
        int nexti;
        for(int i=0;i<num_edges;i++)
        {
            if(i%4==3)nexti=i-3;
            else nexti=i+1;
            angles.push_back(180/3.14159265358979*std::acos(line_line_inner_product(segments[i],segments[nexti])));
            std::cout<<"Angle "<<i<<"is "<<angles[i]<<" deg"<<std::endl;
        }
    }
    return angles;
}

inline std::vector<double> LineSeg2D::length_diagnose()
{
    std::vector<double> lengths;
    if(num_edges==12||num_edges==8)
    {
        int nexti,previ;
        for(int i=0;i<num_edges;i++)
        {
            if(i%4==3){nexti=i-3;previ=i-1;}
            else if(i%4==0) {nexti=i+1;previ=i+3;}
            else {nexti=i+1;previ=i-1;}
            lengths.push_back((line_line_intersect(segments[i],segments[nexti])-line_line_intersect(segments[i],segments[previ])).norm());
            intersects.push_back(line_line_intersect(segments[i],segments[nexti]));
            std::cout<<"Edge Length"<<i<<"is "<<lengths[i]<<" m"<<std::endl;
        }
    }
    return lengths;
}
inline std::vector<Eigen::Vector2d> LineSeg2D::getIntersectCoordinate()
{
    return intersects;
}
//fit a+bx+cx^2+dx^3=0 or a+by+cy^2+dy^3=0
inline void LineSeg2D::cubic_fit()
{
    for (int i=0;i<segments.size();i++)
    {
        std::cout<<"Fit cubic curve ["<<i<<"]: ";
    double slope=std::abs(segments[i].a/segments[i].b);
    std::vector<double> x_values, y_values, coeff;
	double x, y;
    if(slope<1)
    {
        //std::cout<<"segment "<<i<<"coordinate"<<std::endl;
	for (int j=segments[i].start;j<=segments[i].end;j++) {
		x_values.push_back(cloud->points[j].x);
		y_values.push_back(cloud->points[j].y);
        //std::cout<<cloud->points[j].x<<","<<cloud->points[j].y<<std::endl;
	}
    if(x_values.size()>=4)//make sure there are enough points for fitting
    {
	polyfit(x_values, y_values, coeff, 3);
	printf("%f + %f*x + %f*x^2 + %f*x^3\n", coeff[0], coeff[1], coeff[2], coeff[3]);}
    }
    else{
        for (int j=segments[i].start;j<=segments[i].end;j++) {
		x_values.push_back(cloud->points[j].x);
		y_values.push_back(cloud->points[j].y);
	    }
    if(x_values.size()>=4)//make sure there are enough points for fitting
    {
	polyfit(y_values, x_values, coeff, 3);
	printf("%f + %f*y + %f*y^2 + %f*y^3\n", coeff[0], coeff[1], coeff[2], coeff[3]);}
    }
    }
}
#endif 
