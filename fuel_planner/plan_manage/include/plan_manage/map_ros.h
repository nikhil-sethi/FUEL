#ifndef _MAP_ROS_H
#define _MAP_ROS_H

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/CompressedImage.h>
#include <memory>
#include <random>

#include <plan_env/attention_map.h>
using std::shared_ptr;
using std::normal_distribution;
using std::default_random_engine;

class AttentionMap;
class Diffuser;

namespace fast_planner {
class SDFMap;

class MapROS {
public:
  MapROS();
  ~MapROS();
  void setMap(SDFMap* map);
  void setAttentionMap(std::shared_ptr<AttentionMap> att_map);
  void setDiffusionMap(std::shared_ptr<Diffuser> diff_map);

  void init(ros::NodeHandle& nh);

private:
  void depthPoseAttCallback(const sensor_msgs::ImageConstPtr& img,
                         const geometry_msgs::PoseStampedConstPtr& pose,
                         const sensor_msgs::CompressedImageConstPtr& att);
  void cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                         const geometry_msgs::PoseStampedConstPtr& pose);

  void gmmCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

  void metricsTimer(const ros::TimerEvent& /*event*/);

  void updateESDFCallback(const ros::TimerEvent& /*event*/);
  void visCallback(const ros::TimerEvent& /*event*/);

  void publishMapAll();
  void publishMapLocal();
  void publishESDF();
  void publishUpdateRange();
  void publishUnknown();
  void publishDepth();

  void proessDepthImage();

  // custom
  void attCallback(const sensor_msgs::ImageConstPtr& img);
  unique_ptr<cv::Mat> att_image_; // holds 2d attention map
  // ros::Subscriber att_sub_; // subscribes to 2d attention map
  ros::Publisher att_3d_pub_;  // publish 3d attention map
  bool attention_needs_update_ = false;
  ros::Publisher occ_pub_;  // publish occupancy buffer
  ros::Publisher occ_inflate_pub_;  // publish inflated occupancy buffer
  ros::Timer occ_timer_, metrics_timer;
  vector<uint8_t> occupancy_buffer_light;
  void occupancyTimer(const ros::TimerEvent& e); // publishing occupancy buffer
  std::fstream entropy_file;
  ros::Subscriber gmm_sub;

  SDFMap* map_;
  std::shared_ptr<AttentionMap> _att_map;
  std::shared_ptr<Diffuser> _diff_map;
  // may use ExactTime?
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImagePose;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped, sensor_msgs::CompressedImage>
      SyncPolicyImagePoseCompressedImage;
      
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePoseCompressedImage>> SynchronizerImagePoseCompressedImage;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                          geometry_msgs::PoseStamped>
      SyncPolicyCloudPose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyCloudPose>> SynchronizerCloudPose;

  // ros::NodeHandle node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub_;
  shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::CompressedImage>> att_sub_;
  SynchronizerImagePose sync_image_pose_;
  SynchronizerImagePoseCompressedImage sync_image_pose_compimage;  
  SynchronizerCloudPose sync_cloud_pose_;

  ros::Publisher map_local_pub_, map_local_inflate_pub_, esdf_pub_, map_all_pub_, unknown_pub_,
      update_range_pub_, depth_pub_;
  ros::Timer esdf_timer_, vis_timer_;

  // params, depth projection
  double cx_, cy_, fx_, fy_;
  double depth_filter_maxdist_, depth_filter_mindist_;
  int depth_filter_margin_;
  double k_depth_scaling_factor_;
  int skip_pixel_;
  string frame_id_;
  // msg publication
  double esdf_slice_height_;
  double visualization_truncate_height_, visualization_truncate_low_;
  bool show_esdf_time_, show_occ_time_;
  bool show_all_map_;

  // data
  // flags of map state
  bool local_updated_, esdf_need_update_;
  // input
  Eigen::Vector3d camera_pos_;
  Eigen::Quaterniond camera_q_;
  unique_ptr<cv::Mat> depth_image_;
  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt;
  double fuse_time_, esdf_time_, max_fuse_time_, max_esdf_time_;
  int fuse_num_, esdf_num_;
  pcl::PointCloud<pcl::PointXYZI> point_cloud_;

  normal_distribution<double> rand_noise_;
  default_random_engine eng_;

  ros::Time map_start_time_;

  friend SDFMap;
};
}

#endif