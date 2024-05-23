#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <mutex>
#include <common_msgs/Viewpoints.h>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class ObjectFinder;
class TargetPlanner;

namespace fast_planner {
class EDTEnvironment;
class SDFMap;
class FastPlannerManager;
class FrontierFinder;
struct ExplorationParam;
struct ExplorationData;

enum EXPL_RESULT { NO_FRONTIER, FAIL, SUCCEED };
enum TARGET_SEARCH {GREEDY, TSP, TSP_REFINED};

class FastExplorationManager {
public:
  FastExplorationManager();
  ~FastExplorationManager();

  void initialize(ros::NodeHandle& nh);

  int planExploreMotion(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc,
                        const Vector3d& yaw);

  // Benchmark method, classic frontier and rapid frontier
  int classicFrontier(const Vector3d& pos, const double& yaw);
  int rapidFrontier(const Vector3d& pos, const Vector3d& vel, const double& yaw, bool& classic);

  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FrontierFinder> frontier_finder_;
  shared_ptr<ObjectFinder> object_finder;
  shared_ptr<TargetPlanner> target_planner_;
  // unique_ptr<ViewFinder> view_finder_;1

  shared_ptr<SDFMap> getSDFMapPtr(){return sdf_map_;}
  std::vector<geometry_msgs::Pose> target_vpts;
  std::vector<uint16_t> priorities;
  std::vector<uint16_t> expl_priorities;

private:
  shared_ptr<EDTEnvironment> edt_environment_;
  shared_ptr<SDFMap> sdf_map_;
  
  
  std::vector<std::vector<std::vector<Eigen::Vector3d>>> target_paths; // for each point in tour: line segments (start(vec3d) --> end (vec3d)) to each other point in tour
  ros::Subscriber vpts_sub, custom_goal_pose_sub;
  geometry_msgs::Pose custom_goal_pose;
  ros::ServiceClient tsp_client;
  bool CUSTOM_GOAL = false;
  bool use_active_perception_ = false;
  bool use_semantic_search_ = false;
  bool use_lkh_ = false;

  // Find optimal tour for coarse viewpoints of all frontiers
  void findGlobalTour(const Eigen::MatrixXd& cost_mat, vector<uint8_t>& indices);

  // Refine local tour for next few frontiers, using more diverse viewpoints
  void refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
                       const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws,
                       vector<Vector3d>& refined_pts, vector<double>& refined_yaws);

  void shortenPath(vector<Vector3d>& path);
  void targetViewpointsCallback(const common_msgs::Viewpoints& msg);
  void customPoseCallback(const geometry_msgs::PoseWithCovarianceStamped& msg);
  // void findTargetTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw, vector<int>& indices);
  void getPathForTour(const Vector3d& pos, const vector<uint8_t>& ids, vector<Vector3d>& path);
  int getTrajToView(const Eigen::Vector3d& pos,  const Eigen::Vector3d& vel, const Eigen::Vector3d& acc, const Eigen::Vector3d& yaw, Eigen::Vector3d& next_pos, double next_yaw);
  void getTargetCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw, Eigen::MatrixXd& cost_mat);
  void readTourFromFile(vector<int>& indices, const std::string& file_dir);
  void solveMOTSP(const Eigen::MatrixXd& cost_mat, const std::vector<uint16_t>& priorities, std::vector<uint8_t>& tour);

public:
  typedef shared_ptr<FastExplorationManager> Ptr;
  bool init = true;
};

}  // namespace fast_planner

#endif