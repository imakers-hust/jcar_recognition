#include "ros/ros.h"
#include "std_msgs/String.h"
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/core.hpp>

#include <ros/spinner.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/subscriber_filter.h>

//PCL头文件
//#include<pcl/io/pcd_io.h>
//#include<pcl/point_cloud.h>
//#include<pcl/visualization/cloud_viewer.h>
//#include <pcl/features/integral_image_normal.h>
//#include<pcl/point_types.h>
//#include<pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include<pcl/common/eigen.h>

#include<pcl/search/search.h>
#include<std_msgs/Int8.h>
#include<common_msgs/targetState.h>
#include<common_msgs/targetsVector.h>

#include"ssd_detection.h"
using namespace cv::xfeatures2d;
using namespace cv;
using namespace std;
//机器手话题
std::string detect_target_str = "detect_target";
std::string detect_result_str = "detect_result";
ros::Publisher detect_result_pub;
ros::Time timer;

//kinect相机话题
std::string image_rgb_str =  "/kinect2/qhd/image_color_rect";
std::string image_depth_str = "/kinect2/qhd/image_depth_rect";
//std::string cam_info_str = "/kinect2/qhd/camera_info";

//点云定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//模型路径
std::string model_path;
float recog_threshold;

int imagenum;
bool show_image=false;
bool show_cloud =false;
bool simulation_on=false;
//机器手采集指令宏
bool recognition_on=true;
common_msgs::targetsVector coordinate_vec;
vector<PointCloud::Ptr> clouds;
std::vector<int32_t*>px_py;
//int32_t* px_py = new int32_t[2];
//PointCloud::Ptr obj_cloud(new PointCloud);
vector<cv::Rect>rects;
//1.0688009754013181e+03, 0., 9.6848745409735511e+02, 0.,
       //1.0679639125492047e+03, 5.3777681353661535e+02, 0., 0., 1.
//camera  305 kinect2
/*
double camera_factor = 1000;
double camera_cx = 968/2;
double camera_cy = 537/2;
double camera_fx = 1068.8/2;
double camera_fy = 1068/2;
*/
//my kinect2
double camera_factor = 1000;
double camera_cx = 965/2;
double camera_cy = 552/2;
double camera_fx = 1066.6/2;
double camera_fy = 1066.5/2;
//sort point by y
//bool compy(Point2f &a,Point2f &b)
//{
//  if (a.y<b.y)
//    return true;
//  else if (a.y==b.y && a.x<b.x)
//    return true;
//  else
//    return false;
//}
//bool compx(Point2f &a,Point2f &b)
//{
//  if (a.x<b.x)
//    return true;
//  else if (a.x==b.x && a.y<b.y)
//    return true;
//  else
//    return false;
//}

bool comp(const ObjInfo &a, const ObjInfo &b){
    if (a.label < b.label)
        return true;
    else if (a.label == b.label  && a.conf < b.conf)
        return true;
    else                ///这里的else return false非常重要！！！！！
        return false;
}

bool compxy(const cv::Point &a, const cv::Point &b){
    if (a.x < b.x)
        return true;
    else if (a.x == b.x  && a.y < b.y)
        return true;
    else
        return false;
}
cv::Rect toRect(cv::Point* point)
{
  std::vector<cv::Point> points;
  for(int i = 0;i<4;i++)
    points.push_back(point[i]);
  sort(points.begin(), points.end(), compxy);
  cv::Rect rect = cv::Rect(points[0],points[3]);
  return rect;
}


void InitRecognition(int image_wid, int image_hei, int image_chan)
{
  // step 1
  if(!InitSession(model_path, 1, image_wid, image_hei, image_chan))
  {
    ROS_ERROR_STREAM("init session failed!!!the path is :"<<model_path);
    return ;
  }
  ROS_INFO_STREAM("init success");
  timer = ros::Time::now();
}


void Recognition(std::vector<ObjInfo>& obj_boxes, cv::Mat src)
{
  if(src.empty())
  {
    ROS_ERROR_STREAM(" image is empty!!!");
    return ;
  }

  ROS_INFO_STREAM("Object Detection!");

  // step 2
  cv::Mat res;
  cv::cvtColor(src, res, cv::COLOR_BGR2RGB);
  if(!FeedData(res))
  {
    ROS_ERROR_STREAM("feed data");
    return ;
  }
  obj_boxes.clear();

  // step 3
  if(!Detection(obj_boxes, res,recog_threshold))
  {
    ROS_ERROR_STREAM("detection");
    return ;
  }
  //    ROS_INFO_STREAM("detection success");
  //    ROS_INFO_STREAM("rectangle success");
  for(size_t i = 0; i < obj_boxes.size(); i++)
  {
    ROS_INFO_STREAM("label:"<< obj_boxes[i].label<<" score:"<<obj_boxes[i].conf);
  }
  //cv::imwrite("../result/result.jpg", src);
}

void GetCloud(std::vector<ObjInfo>& rects, cv::Mat image_rgb, cv::Mat image_depth)
{
  clouds.clear();
  px_py.clear();
  //color recognition
  cv::Mat image_HSV;
  std::vector<cv::Mat> HSV_split;
  cv::cvtColor(image_rgb, image_HSV, cv::COLOR_BGR2HSV);
 // cv::split(image_HSV, HSV_split);
  //cv::equalizeHist(HSV_split[2],HSV_split[2]);
 // cv::merge(HSV_split,image_HSV);
  cv::Mat img_thresholded;
  //int minh = 22, maxh = 95, mins = 0, maxs = 255, minv = 31, maxv = 229;
  int minh = 0, maxh = 180, mins = 0, maxs = 255, minv = 0, maxv = 46;
//  int minh = 0, maxh = 10, mins = 43, maxs = 255, minv = 46, maxv = 255;
  cv::inRange(image_HSV, cv::Scalar(minh, mins, minv), cv::Scalar(maxh, maxs, maxv), img_thresholded);

  //开操作 (去除一些噪点)
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(img_thresholded, img_thresholded, cv::MORPH_OPEN, element);
  //闭操作 (连接一些连通域)
  cv::morphologyEx(img_thresholded, img_thresholded, cv::MORPH_CLOSE, element);
 for(size_t rect_num = 0;rect_num<rects.size();rect_num++)
 {
     cv::Point* rect_points = rects[rect_num].bbox;
     cv::Rect rect = toRect(rect_points);

     PointCloud::Ptr temp_cloud(new PointCloud);
     temp_cloud->is_dense = false;
     for(int i = rect.x;i<rect.x+rect.width;i++)
     {
       for(int j = rect.y;j<rect.y+rect.height;j++)
       {
         //remove black area of the image
         if(img_thresholded.ptr<uchar>(j)[i]>0)
           continue;
         // 获取深度图中(i,j)处的值
         ushort d = image_depth.ptr<ushort>(j)[i];

         // 计算这个点的空间坐标
         PointT p;
         p.z = double(d) / camera_factor;
         p.x = (i- camera_cx) * p.z / camera_fx;
         p.y = (j - camera_cy) * p.z / camera_fy;

         p.b = image_rgb.ptr<uchar>(j)[i*3];
         p.g = image_rgb.ptr<uchar>(j)[i*3+1];
         p.r = image_rgb.ptr<uchar>(j)[i*3+2];

         temp_cloud->points.push_back( p );
       }
     }
     temp_cloud->height = 1;
     temp_cloud->width = temp_cloud->points.size();
     clouds.push_back(temp_cloud);
     int32_t* temp_xy = new int32_t[2];
     temp_xy[0] = int32_t(rect.x+rect.width/2);
     temp_xy[1] = int32_t(rect.y+rect.height/2);
     px_py.push_back(temp_xy);
     //delete []temp_xy;

  }

}

   //  cout<<"number of clouds is :"<<obj_cloud->points.size();
   // cout<<"center is: "<<px_py[0]<<" , "<<px_py[1]<<endl;
     //pcl::io::savePCDFileBinary("obj.pcd", *obj_cloud );

void calculate_clouds_coordinate(std::vector<ObjInfo>&Obj_Frames)
{
    coordinate_vec.targets.clear();
    for(size_t i = 0;i<Obj_Frames.size();i++)
    {
        PointCloud::Ptr cloud = clouds[i];
        int* temp_xy = px_py[i];

        common_msgs::targetState coordinate;
         coordinate.tag = Obj_Frames[i].label;
        coordinate.px = temp_xy[0];
        coordinate.py = temp_xy[1];


        ////利用PCA主元分析法获得点云的三个主方向，获取质心，计算协方差，获得协方差矩阵，求取协方差矩阵的特征值和特长向量，特征向量即为主方向。
        Eigen::Vector4f pcaCentroid;
        pcl::compute3DCentroid(*cloud, pcaCentroid);
        coordinate.x = pcaCentroid(0);
        coordinate.y = pcaCentroid(1);
        coordinate.z = pcaCentroid(2);
        float coordinate_len = sqrt(coordinate.x*coordinate.x+coordinate.y*coordinate.y+coordinate.z*coordinate.z);
        //if(coordinate_len<1e-5)
        //    continue;
    //    ROS_INFO_STREAM("calculate xyz:"<<coordinate.x<<" "<<coordinate.y<<" "<<coordinate.z);

    //    ros::Time axis_begin = ros::Time::now();
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    //    ROS_INFO_STREAM("computeCovariance!");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    //    ros::Time axis_end = ros::Time::now();
    //    ros::Duration axis_interval = axis_end-axis_begin;
    //    ROS_INFO_STREAM("computing size "<<cloud->points.size()<<" for "<<axis_interval.toSec()<<"s!!!");

        eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //校正主方向间垂直
        eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
        eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
        Eigen::Vector3f orient0 = eigenVectorsPCA.col(0);
        Eigen::Vector3f orient1 = eigenVectorsPCA.col(1);
        Eigen::Vector3f orient2 = eigenVectorsPCA.col(2);
        float max_orient[3] = {0, 0, 0};
        for(size_t i = 0;i<cloud->points.size();i++)
        {
          Eigen::Vector3f temp_vec;
          temp_vec(0) = cloud->points[i].x-coordinate.x;
          temp_vec(1) = cloud->points[i].y-coordinate.y;
          temp_vec(2) = cloud->points[i].z-coordinate.z;
          float temp_orient0 = abs(temp_vec.dot(orient0));
          float temp_orient1 = abs(temp_vec.dot(orient1));
          float temp_orient2 = abs(temp_vec.dot(orient2));
          max_orient[0] = temp_orient0>max_orient[0]?temp_orient0:max_orient[0];
          max_orient[1] = temp_orient1>max_orient[1]?temp_orient1:max_orient[1];
          max_orient[2] = temp_orient2>max_orient[2]?temp_orient2:max_orient[2];
        }

        std::vector<size_t> idx(3);
        for(size_t i = 0;i!=idx.size();i++)idx[i] = i;
         // 通过比较v的值对索引idx进行排序
        std::sort(idx.begin(), idx.end(), [& max_orient](size_t i1, size_t i2) {return max_orient[i1] < max_orient[i2];});
        Eigen::Matrix3f rotation;
    //    for(size_t i = 0;i<3;i++)
        rotation.col(0) = eigenVectorsPCA.col(idx[2]);
        rotation(0,2) = 0;
        rotation(1,2) = 0;
        rotation(2,2) = 1;
        rotation.col(1) = rotation.col(2).cross(rotation.col(0));
        float len = sqrt(rotation(0,1)*rotation(0,1)+rotation(1,1)*rotation(1,1)+rotation(2,1)*rotation(2,1));
        rotation(0,1)/=len;
        rotation(1,1)/=len;
        rotation(2,1)/=len;
        rotation.col(0) = rotation.col(1).cross(rotation.col(2));
        Eigen::Quaternionf quaternion(rotation);
        coordinate.qx = quaternion.x();
        coordinate.qy = quaternion.y();
        coordinate.qz = quaternion.z();
        coordinate.qw = quaternion.w();
        cout<<"center"<<endl;
        //cout<<coordinate.tag<<endl;
        cout<<coordinate.tag<<endl;
        cout<<coordinate.px<<endl;
        cout<<coordinate.py<<endl;
        cout<< coordinate.x<<endl;
        cout<< coordinate.y<<endl;
        cout<< coordinate.z<<endl<<endl;
        rects.clear();
        //put the calculated coordinate into the vector
        coordinate_vec.targets.push_back(coordinate);

    }
}


void image_Callback( const sensor_msgs::ImageConstPtr &image_rgb,
                     const sensor_msgs::ImageConstPtr  &image_depth )
{
  if(recognition_on==true)
  {
    ROS_INFO_STREAM("Recognition is on!!!");

  //转换ROS图像消息到opencv图像
    cv::Mat mat_image_rgb = cv_bridge::toCvShare(image_rgb)->image;
    cv::Mat mat_image_depth = cv_bridge::toCvShare(image_depth)->image;

    //识别物体
    std::vector<ObjInfo> Obj_Frames;
    ros::Time start_esti = ros::Time::now();
    Recognition(Obj_Frames, mat_image_rgb);
    ros::Duration recog_interval = ros::Time::now()-start_esti;
    ROS_INFO_STREAM("Recognizing image for "<<recog_interval.toSec()<<" s!!!");

    //识别结果筛选
    sort(Obj_Frames.begin(), Obj_Frames.end(), comp);
    std::vector<ObjInfo>::iterator obj_it = Obj_Frames.begin();
    while(obj_it!=Obj_Frames.end())
    {
      //保留相同标签下置信度最大的物体
      if(obj_it != Obj_Frames.end()-1 && obj_it->label == (obj_it+1)->label)
      {
        obj_it = Obj_Frames.erase(obj_it);
        continue;
      }
      cv::Rect rect = toRect(obj_it->bbox);
      if(rect.area()>100000)
      {
        obj_it = Obj_Frames.erase(obj_it);
        continue;
      }
      obj_it++;
    }

    GetCloud(Obj_Frames, mat_image_rgb, mat_image_depth );

    calculate_clouds_coordinate(Obj_Frames);

    //发送点云对应坐标
    detect_result_pub.publish(coordinate_vec);
    cout<<"information publish success!"<<endl;
    cout<<endl;

    //绘制识别框、对应的置信度、帧率
    for(size_t i = 0;i<Obj_Frames.size();i++)
    {
      cv::Point* vertexes = Obj_Frames[i].bbox;
      for(int j = 0; j < 4; j++)
        cv::line(mat_image_rgb, vertexes[j], vertexes[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
      char buffer[100];
      std::sprintf(buffer, "label:%d conf:%.2f", Obj_Frames[i].label, Obj_Frames[i].conf);
      cv::putText(mat_image_rgb, buffer, vertexes[0], cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    if (show_image)
    {
      try
      {
        //cv::imshow(window_rgb_top, cv_bridge::toCvShare(image_rgb)->image);
        cv::imshow("rgb_image", mat_image_rgb);
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("Could not convert colored images from '%s' to 'bgr8'.", image_rgb->encoding.c_str());
      }
      try
      {
        cv::imshow("depth_image", mat_image_depth);
      }
      catch(cv_bridge::Exception& e)
      {
        ROS_ERROR("Could not convert depth images from '%s' to 'bgr8'.", image_depth->encoding.c_str());
      }
    }

  }

}

void RobotSignalCallback(const std_msgs::Int8::ConstPtr& msg)
{
  if(msg->data == 1)
    recognition_on = true;
  else
    recognition_on = false;
}

void ReleaseRecognition()
{
//  if(show_cloud)
//  {
//    clouds_show->clear();
//  }
  for(size_t i = 0;i<clouds.size();i++)
  {
    clouds[i]->clear();
  }
  // step 4
  ReleaseSession();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "kinect2_image");
  ros::NodeHandle nh;

  image_transport::ImageTransport it(nh);

  //获取识别参数
  if(argc<2)
    model_path = "/home/zhsyi/kinova/tensorflow/project/result1/frozen_inference_graph.pb";
  else model_path = argv[1];
  if(!nh.getParam("threshold", recog_threshold))
    recog_threshold = 0.2;
//  if(!nh.getParam("show_cloud", show_cloud))
//    show_cloud = false;
  if(!nh.getParam("simulation", simulation_on))
    simulation_on = false;


  message_filters::Subscriber<sensor_msgs::Image> image_rgb_sub(nh, image_rgb_str, 1);
  message_filters::Subscriber<sensor_msgs::Image>image_depth_sub(nh, image_depth_str, 1);
 // message_filters::Subscriber<sensor_msgs::CameraInfo>cam_info_sub(nh,  cam_info_str, 1);


  //同步深度图和彩色图
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_rgb_sub, image_depth_sub, 10);
  sync.registerCallback(boost::bind(&image_Callback, _1, _2));

  ros::Subscriber detect_sub = nh.subscribe(detect_target_str, 1000, RobotSignalCallback);
  detect_result_pub = nh.advertise<common_msgs::targetsVector>(detect_result_str.c_str(), 1000);

  //初始化识别
 // if(simulation_on==0)
        InitRecognition(800, 800, 3);
  //  else
  //      InitRecognition(960, 540, 3);

  ros::spin();
  while(ros::ok());
  //释放资源
  ReleaseRecognition();
  return 0;
}
