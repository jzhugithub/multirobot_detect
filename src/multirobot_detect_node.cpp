#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>  
#include <fstream>  
#include <strstream>
#include <opencv2/core/core.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <opencv2/gpu/gpu.hpp>  
#include "someMethod.h"
#include "parameter.h"


using namespace cv;
using namespace std;

class MultiRobotDetecter
{
  //node
  ros::NodeHandle nh_;
  ros::NodeHandle nh_image_param;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  //svm
  MySVM svm_detect;
  MySVM svm_classify;
  //dimension of HOG descriptor: 
    //[(window_width-block_width)/block_stride_width+1]*[(window_height-block_height)/block_stride_height+1]*bin_number*(block_width/cell_width)*(block_height/cell_height)
  int descriptor_dim_detect;
  int descriptor_dim_classify;
  int support_vector_num_detect;//number of vector, not the dimention of vector
  int support_vector_num_classify;
  Mat alpha_mat_detect;
  Mat support_vector_mat_detect;
  Mat result_mat_detect;
  //HOG descriptor
  gpu::HOGDescriptor HOG_descriptor_detect;
  HOGDescriptor HOG_descriptor_classify;
  //video
  string INPUT_VIDEO_WINDOW_NAME;
  string RESULT_VIDEO_WINDOW_NAME;
  bool show_video_flag;
  bool save_result_video_flag;
  double video_rate;
  double image_hight;
  double image_width;
  double video_delay;
  VideoWriter output_video;
  //frame
  int frame_num;
  Mat src_3,src_4,dst_3;
  gpu::GpuMat src_GPU;
  vector<Rect> location;
  
public:
  MultiRobotDetecter():
  it_(nh_),//intial it_
  nh_image_param("~")
  {
    //node
    // Subscrive to input video feed from "/dji_sdk/image_raw" topic, imageCb is the callback function
    image_sub_ = it_.subscribe("/dji_sdk/image_raw", 1, &MultiRobotDetecter::imageCb, this);
    //svm
    svm_detect.load(DetectSvmName);
    svm_classify.load(ClassifySvmName);
    descriptor_dim_detect = svm_detect.get_var_count();
    descriptor_dim_classify = svm_classify.get_var_count();
    support_vector_num_detect = svm_detect.get_support_vector_count();
    support_vector_num_classify = svm_classify.get_support_vector_count();
    alpha_mat_detect = Mat::zeros(1, support_vector_num_detect, CV_32FC1);
    support_vector_mat_detect = Mat::zeros(support_vector_num_detect, descriptor_dim_detect, CV_32FC1);
    result_mat_detect = Mat::zeros(1, descriptor_dim_detect, CV_32FC1);
    //HOG descriptor
    HOG_descriptor_detect = gpu::HOGDescriptor(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect,1,0.2,false,5);
    //HOG_descriptor_detect = HOGDescriptor(WinSizeDetect,BlockSizeDetect,BlockStrideDetect,CellSizeDetect,NbinsDetect,1,0.2,false,5);
    HOG_descriptor_classify = HOGDescriptor(WinSizeClassify,BlockSizeClassify,BlockStrideClassify,CellSizeClassify,NbinsClassify);
    for(int i=0; i<support_vector_num_detect; i++) 
    {
      const float * support_vector_detect = svm_detect.get_support_vector(i);
      for(int j=0; j<descriptor_dim_detect; j++)  
        support_vector_mat_detect.at<float>(i,j) = support_vector_detect[j];  
    }
    double * alpha_detect = svm_detect.get_alpha_vector();
    for(int i=0; i<support_vector_num_detect; i++)
      alpha_mat_detect.at<float>(0,i) = alpha_detect[i];  
    result_mat_detect = -1 * alpha_mat_detect * support_vector_mat_detect;
    vector<float> detector_detect;
    for(int i=0; i<descriptor_dim_detect; i++)
      detector_detect.push_back(result_mat_detect.at<float>(0,i)); 
    detector_detect.push_back(svm_detect.get_rho());//add rho
    cout<<"dimension of svm detector for HOG detect(w+b):"<<detector_detect.size()<<endl;
    HOG_descriptor_detect.setSVMDetector(detector_detect);
    //video
    INPUT_VIDEO_WINDOW_NAME="input video";
    RESULT_VIDEO_WINDOW_NAME="result video";
    namedWindow(INPUT_VIDEO_WINDOW_NAME);
    namedWindow(RESULT_VIDEO_WINDOW_NAME);
    if(!nh_image_param.getParam("show_video_flag", show_video_flag))show_video_flag = false;
    if(!nh_image_param.getParam("save_result_video_flag", save_result_video_flag))save_result_video_flag = false;
    if(!nh_image_param.getParam("rate", video_rate))video_rate = 5;
    if(!nh_image_param.getParam("image_hight", image_hight))image_hight = 360.0;
    if(!nh_image_param.getParam("image_width", image_width))image_width = 640.0;
    video_delay = 1000/video_rate;
    output_video = VideoWriter(ResultVideo, CV_FOURCC('M', 'J', 'P', 'G'), video_rate, Size(image_width, image_hight));
    //frame
    frame_num = 1;
  }
  
  ~MultiRobotDetecter()
  {
    destroyWindow(INPUT_VIDEO_WINDOW_NAME);
    destroyWindow(RESULT_VIDEO_WINDOW_NAME);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv_ptr->image.copyTo(src_3);
    
    //frame
    cout<<frame_num<<endl;
    frame_num++;
    
    src_3.copyTo(dst_3);
    cvtColor(src_3,src_4,CV_BGR2BGRA);
    src_GPU.upload(src_4);
    
    //detect
    HOG_descriptor_detect.detectMultiScale(src_GPU, location, HitThreshold, WinStride, Size(), DetScale, 2);
    //HOG_descriptor_detect.detectMultiScale(src0, found, HitThreshold, WinStride, Size(), DetScale, 2);
    
    //classfy
    for(int i=0; i<location.size(); i++)  
    {
      cout<<"width:"<<location[i].width<<"  height:"<<location[i].height<<endl;
      vector<float> descriptor_classify;
      Mat descriptor_mat_classify(1, descriptor_dim_classify, CV_32FC1);
      Mat src_classify;
      
      resize(src_3(location[i]),src_classify,WinSizeClassify);
      HOG_descriptor_classify.compute(src_classify,descriptor_classify);
      for(int j=0; j<descriptor_dim_classify; j++)  
	descriptor_mat_classify.at<float>(0,j) = descriptor_classify[j];
      float classifyResult = svm_classify.predict(descriptor_mat_classify);
      
      if (classifyResult == 1)//irobot
      {
	rectangle(dst_3, location[i], CV_RGB(0,0,255), 3);
	if (SAVESET)
	{
	  strstream ss;
	  string s;
	  ss<<ResultVideoFile_1<<1000*frame_num+i<<".jpg";
	  ss>>s;
	  imwrite(s,src_3(location[i]));
	}
      } 
      else if (classifyResult == 2)//obstacle
      {
	rectangle(dst_3, location[i], CV_RGB(0,255,0), 3);
	if (SAVESET)
	{
	  strstream ss;
	  string s;
	  ss<<ResultVideoFile_2<<1000*frame_num+i<<".jpg";
	  ss>>s;
	  imwrite(s,src_3(location[i]));
	}
      }
      else if (classifyResult ==3)//background
      {
	rectangle(dst_3, location[i], Scalar(0,0,255), 3);
	if (SAVESET)
	{
	  strstream ss;
	  string s;
	  ss<<ResultVideoFile_3<<1000*frame_num+i<<".jpg";
	  ss>>s;
	  imwrite(s,src_3(location[i]));
	}
      }
      else//other
      {
	rectangle(dst_3, location[i], Scalar(255,255,255), 3);
      }
    }
    location.clear();
    
    //save and show video
    if(save_result_video_flag)
    {
      output_video<<dst_3;
    }
    if(show_video_flag)
    {
      imshow(INPUT_VIDEO_WINDOW_NAME, src_3);
      imshow(RESULT_VIDEO_WINDOW_NAME, dst_3);
      waitKey(1);
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "multirobot_detect_node");//node name
  double loop_rate;
  MultiRobotDetecter mrd;//class initializing
  ros::NodeHandle nh_loop_param;
  if(!nh_loop_param.getParam("rate", loop_rate))loop_rate = 5;//video
  ros::Rate loop_rate_class(loop_rate);//frequency: n Hz
  
  while(ros::ok())
    {
      ros::spinOnce();
      loop_rate_class.sleep();
    }
    ros::spin();
    return 0;
}

