// #include <fstream>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <lkh_tsp_solver/lkh_interface.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <active_perception/frontier_finder.h>
#include <active_perception/object_finder.h>
#include <active_perception/target_planner.h>
#include <plan_manage/planner_manager.h>

#include <exploration_manager/expl_data.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>
#include <target_search/TSP.h>

using namespace Eigen;



namespace fast_planner {
// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager() {
}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));

  // view_finder_.reset(new ViewFinder(edt_environment_, nh));
  planner_manager_->diffuser_->setFrontierFinder(frontier_finder_);
  frontier_finder_->setDiffuser(planner_manager_->diffuser_);


  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);

  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);

  nh.param("/use_object_vpts", use_object_vpts_, false);
  // nh.param("/use_semantic_search", use_semantic_search_, false);
  nh.param("/use_motsp", use_motsp_, false);
  nh.param("use_diffusion", use_diffusion_, false);
  nh.param("use_greedy_search", use_greedy_search, false);

  if (use_object_vpts_){
    object_finder.reset(new ObjectFinder(nh));
    object_finder->setPriorityMap(planner_manager_->att_map);
    object_finder->setSDFMap(sdf_map_);

    target_planner_.reset(new TargetPlanner);
    target_planner_->setObjectFinder(object_finder);
    target_planner_->setFrontierFinder(frontier_finder_);
    target_planner_->setDiffusionMap(planner_manager_->diffuser_);
    target_planner_->setSDFMap(sdf_map_);
    target_planner_->init(nh);
  }


  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);

  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  // planner_manager_->path_finder_->max_search_time_ = 0.05;
  planner_manager_->path_finder_->max_search_time_ = 1.0;

  // Initialize TSP par file
  ofstream par_file(ep_->tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";

  // Initialize TSP par file
  std::string dir = "/root/thesis_ws/src/thesis/sw/bringup/resource";
  ofstream par_file2(dir+"/single.par");
  par_file2 << "PROBLEM_FILE = " << dir <<"/single.tsp\n";
  par_file2 << "GAIN23 = NO\n";
  par_file2 << "OUTPUT_TOUR_FILE =" << dir << "/single.txt\n";
  par_file2 << "RUNS = 1\n";

  // Analysis
  // ofstream fout;
  // fout.open("/home/boboyu/Desktop/RAL_Time/frontier.txt");
  // fout.close();
  vpts_sub = nh.subscribe("/objects/target_vpts", 10, &FastExplorationManager::targetViewpointsCallback, this);
  custom_goal_pose_sub = nh.subscribe("/initialpose", 10, &FastExplorationManager::customPoseCallback, this);

  // custom goal. just something to start with
  custom_goal_pose.position.x = -1;
  custom_goal_pose.position.y = -1;
  custom_goal_pose.position.z = 1;
  
  tsp_client = nh.serviceClient<target_search::TSP>("/planning/motsp_service");

}

int FastExplorationManager::planExploreMotion(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d& yaw) {
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;


  ed_->views_.clear();
  ed_->global_tour_.clear();
  int ts_type = TARGET_SEARCH::TSP;

  // std::cout << "start pos: " << pos.transpose() << ", vel: " << vel.transpose()
            // << ", acc: " << acc.transpose() << std::endl;
  

  // ===== FIND FRONTIERS =======
  // Search frontiers and group them into clusters
  frontier_finder_->removeOldFrontiers();
  frontier_finder_->searchNewFrontiers();

  // diffusion
  if (use_diffusion_)
    planner_manager_->diffuser_->diffusionTimer(ros::TimerEvent());

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Find viewpoints (x,y,z,yaw) for all frontier clusters and get visible ones' info
  frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);
  
  
  if (!ed_->frontiers_.empty())
    frontier_finder_->updateFrontierCostMatrix();
    
  // ===== FIND NEXT VIEWPOINT and GLOBAL PATH ========

  // Do global and local tour planning and retrieve the next viewpoint
  Vector3d next_pos;
  double next_yaw;
  // insert target viewpoints:
  if (CUSTOM_GOAL){
      next_pos(0) = custom_goal_pose.position.x;
      next_pos(1) = custom_goal_pose.position.y;
      next_pos(2) = 0.25;
      geometry_msgs::Quaternion q = custom_goal_pose.orientation;
      double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
      double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
      next_yaw = std::atan2(siny_cosp, cosy_cosp);    
  }
  else if (use_object_vpts_ && !target_vpts.empty()){
    int num_targets_vpts = target_vpts.size();
    // greedy TODO change this to viewpoint cost to account for yaw as well
    // find the closest viewpoint to current position
    if (num_targets_vpts == 1 || use_greedy_search){

      // METRIC COST
      // double min_cost = 10000.0;
      // for (auto& vpt: target_vpts){
      //   // double dist = std::sqrt(pow(pos(0) - vpt.position.x, 2) + pow(pos(1) - vpt.position.y, 2) + pow(pos(2) - vpt.position.z, 2));
        
      //   Eigen::Vector3d tmp_pos(vpt.position.x, vpt.position.y, vpt.position.z);
      //   geometry_msgs::Quaternion q = vpt.orientation;
      //   double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
      //   double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
      //   double tmp_yaw = std::atan2(siny_cosp, cosy_cosp);

      //   vector<Vector3d> path;
      //   double cost = ViewNode::computeCost(pos, tmp_pos, yaw[0], tmp_yaw, vel, yaw[1], path);
      //   if (cost < min_cost){
      //     next_pos = tmp_pos;
      //     next_yaw = tmp_yaw;
      //     min_cost = cost;
      //   }
      // }

      // SEMANTIC COST
      int argmax = std::max_element(priorities.begin(), priorities.end())-priorities.begin();
      geometry_msgs::Pose vpt = target_vpts[argmax];
      next_pos = Eigen::Vector3d(vpt.position.x, vpt.position.y, vpt.position.z); 

      geometry_msgs::Quaternion q = vpt.orientation;
      double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
      double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
      next_yaw = std::atan2(siny_cosp, cosy_cosp);
      ed_->global_tour_ = {pos, next_pos};
    }

    // TSP
    else if (num_targets_vpts>1){
      t1 = ros::Time::now();

      Eigen::MatrixXd cost_mat;
      getTargetCostMatrix(pos, vel, yaw, cost_mat);

      vector<uint8_t> indices;
      // vector<uint16_t> priorities = vector<uint16_t>(num_targets_vpts+1,1);
      if (!use_motsp_){
        findGlobalTour(cost_mat, indices);
      }
      else{
        solveMOTSP(cost_mat, priorities, indices);     
      }
      
      // solve the TSP and read the tour as indices
      // solveTSPAndGetTour(cost_mat, "/root/thesis_ws/src/thesis/sw/bringup/resource");
      // readTourFromFile(indices, ep_->tsp_dir_);
      
      // Get the path of optimal tour from path matrix
      getPathForTour(pos, indices, ed_->global_tour_);

      double tsp_time = (ros::Time::now()-t1).toSec();
      ROS_WARN("Target TSP time: %lf", tsp_time);


      // Choose the next viewpoint from global tour
      auto vpt = target_vpts[indices[0]];
      next_pos(0) = vpt.position.x;
      next_pos(1) = vpt.position.y;
      next_pos(2) = vpt.position.z;
      geometry_msgs::Quaternion q = vpt.orientation;
      double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
      double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
      next_yaw = std::atan2(siny_cosp, cosy_cosp);
    
    }

  }
  else{ // pure exploration
    if (ed_->frontiers_.empty()) {
      ROS_WARN("No coverable frontier.");
      return NO_FRONTIER;
    }

    frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_->yaws_, ed_->averages_);
    for (uint i = 0; i < ed_->points_.size(); ++i)
      ed_->views_.push_back(ed_->points_[i] + 2.0 * Vector3d(cos(ed_->yaws_[i]), sin(ed_->yaws_[i]), 0));

    double view_time = (ros::Time::now() - t1).toSec();
    ROS_WARN("Frontier: %d, t: %lf, viewpoint: %d, t: %lf", ed_->frontiers_.size(), frontier_time, ed_->points_.size(), view_time);

    // Global plan using TSP
    if (use_greedy_search){
      double gain = -1;
      expl_priorities.clear();
        for (Frontier ftr: frontier_finder_->frontiers_){
          if (ftr.viewpoints_.front().visib_num_>gain){
            gain = ftr.viewpoints_.front().visib_num_;
            next_pos = ftr.viewpoints_.front().pos_;
            next_yaw = ftr.viewpoints_.front().yaw_;
          }
        }
    }
    else if (ed_->points_.size() > 1) {
      // Find the global tour passing through all viewpoints
      // Create TSP and solve by LKH
      // Optimal tour is returned as indices of frontier

      t1 = ros::Time::now();
      vector<uint8_t> indices;     

      // Get cost matrix for current state and clusters
      Eigen::MatrixXd cost_mat;
      frontier_finder_->getFullCostMatrix(pos, vel, yaw, cost_mat);
      
      // Solve TSP and get tour as indices of frontiers
      if (!use_motsp_){ // Metric LKH
        findGlobalTour(cost_mat, indices);
      }
      else{  // 2-opt-LNS with priorities
        expl_priorities.clear();
        for (Frontier ftr: frontier_finder_->frontiers_){
          expl_priorities.push_back((uint16_t)ftr.viewpoints_.front().visib_num_);
        }
        solveMOTSP(cost_mat, expl_priorities, indices);
      }
      

      // Get the Full Astar path of optimal tour
      frontier_finder_->getPathForTour(pos, indices, ed_->global_tour_);

      double tsp_time = (ros::Time::now()-t1).toSec();
      ROS_WARN("Exploration TSP time: %lf", tsp_time);


      if (ep_->refine_local_) {
        // Do refinement for the next few viewpoints in the global tour
        // Idx of the first K frontier in optimal tour
        t1 = ros::Time::now();

        ed_->refined_ids_.clear();
        ed_->unrefined_points_.clear();
        int knum = min(int(indices.size()), ep_->refined_num_);
        for (int i = 0; i < knum; ++i) {
          auto tmp = ed_->points_[indices[i]];
          ed_->unrefined_points_.push_back(tmp);
          ed_->refined_ids_.push_back(indices[i]);
          if ((tmp - pos).norm() > ep_->refined_radius_ && ed_->refined_ids_.size() >= 2) break;
        }

        // Get top N viewpoints for the next K frontiers
        ed_->n_points_.clear();
        vector<vector<double>> n_yaws;
        frontier_finder_->getViewpointsInfo(
            pos, ed_->refined_ids_, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

        ed_->refined_points_.clear();
        ed_->refined_views_.clear();
        vector<double> refined_yaws;
        refineLocalTour(pos, vel, yaw, ed_->n_points_, n_yaws, ed_->refined_points_, refined_yaws);
        next_pos = ed_->refined_points_[0];
        next_yaw = refined_yaws[0];

        // Get marker for view visualization
        for (int i = 0; i < ed_->refined_points_.size(); ++i) {
          Vector3d view =
              ed_->refined_points_[i] + 2.0 * Vector3d(cos(refined_yaws[i]), sin(refined_yaws[i]), 0);
          ed_->refined_views_.push_back(view);
        }
        ed_->refined_views1_.clear();
        ed_->refined_views2_.clear();
        for (int i = 0; i < ed_->refined_points_.size(); ++i) {
          vector<Vector3d> v1, v2;
          frontier_finder_->percep_utils_->setPose(ed_->refined_points_[i], refined_yaws[i]);
          frontier_finder_->percep_utils_->getFOV(v1, v2);
          ed_->refined_views1_.insert(ed_->refined_views1_.end(), v1.begin(), v1.end());
          ed_->refined_views2_.insert(ed_->refined_views2_.end(), v2.begin(), v2.end());
        }
        double local_time = (ros::Time::now() - t1).toSec();
        ROS_WARN("Local refine time: %lf", local_time);

      } else {
        // Choose the next viewpoint from global tour
        next_pos = ed_->points_[indices[0]];
        next_yaw = ed_->yaws_[indices[0]];
      }
    } else if (ed_->points_.size() == 1) {
      // Only 1 destination, no need to find global tour through TSP
      ed_->global_tour_ = { pos, ed_->points_[0] };
      ed_->refined_tour_.clear();
      ed_->refined_views1_.clear();
      ed_->refined_views2_.clear();

      if (ep_->refine_local_) {
        // Find the min cost viewpoint for next frontier
        ed_->refined_ids_ = { 0 };
        ed_->unrefined_points_ = { ed_->points_[0] };
        ed_->n_points_.clear();
        vector<vector<double>> n_yaws;
        frontier_finder_->getViewpointsInfo(
            pos, { 0 }, ep_->top_view_num_, ep_->max_decay_, ed_->n_points_, n_yaws);

        double min_cost = 100000;
        int min_cost_id = -1;
        vector<Vector3d> tmp_path;
        for (int i = 0; i < ed_->n_points_[0].size(); ++i) {
          auto tmp_cost = ViewNode::computeCost(
              pos, ed_->n_points_[0][i], yaw[0], n_yaws[0][i], vel, yaw[1], tmp_path);
          if (tmp_cost < min_cost) {
            min_cost = tmp_cost;
            min_cost_id = i;
          }
        }
        next_pos = ed_->n_points_[0][min_cost_id];
        next_yaw = n_yaws[0][min_cost_id];
        ed_->refined_points_ = { next_pos };
        ed_->refined_views_ = { next_pos + 2.0 * Vector3d(cos(next_yaw), sin(next_yaw), 0) };
      } else {
        next_pos = ed_->points_[0];
        next_yaw = ed_->yaws_[0];
      }
    } else
    ROS_ERROR("Empty destination.");

  }

  ed_->next_pos_ = next_pos;
  ed_->next_yaw_ = next_yaw;

  std::cout << "Next view: " << next_pos.transpose() << ", " << next_yaw << std::endl;


  // ===== FIND LOCAL PATH TO NEXT VIEWPOINT

  // Plan trajectory (position and yaw) to the next viewpoint
  t1 = ros::Time::now();

  // search coarse Astar path 
  // Generate trajectory of x,y,z
  planner_manager_->path_finder_->reset();
  if (planner_manager_->path_finder_->search(pos, next_pos) != Astar::REACH_END) {
    ROS_ERROR("No path to next viewpoint");
    return FAIL;
  }
  ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();
  shortenPath(ed_->path_next_goal_);

  const double len = Astar::pathLength(ed_->path_next_goal_);

  if (len<0.1) return TRAJ_FAIL;

  // plan a bspline through points
  if (getTrajToView(pos, vel, acc, yaw, next_pos, next_yaw, len) == FAIL)
    return FAIL;

  double traj_plan_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  double yaw_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Traj: %lf, yaw: %lf", traj_plan_time, yaw_time);
  double total = (ros::Time::now() - t2).toSec();
  ROS_WARN("Total time: %lf", total);
  ROS_ERROR_COND(total > 0.1, "Total time too long!!!");

  return SUCCEED;
}

void FastExplorationManager::solveMOTSP(const Eigen::MatrixXd& cost_mat, const std::vector<uint16_t>& priorities, std::vector<uint8_t>& tour){
  const uint8_t dimension = cost_mat.rows();

  // get the upper triangular matrix. reconstructed in python later on
  std::vector<uint16_t> cost_mat_flat;
  for (int i=0; i<dimension; i++)
    for (int j=i+1; j<dimension; j++){
    cost_mat_flat.push_back((uint16_t)(cost_mat(i,j)*1000));
  }

  target_search::TSP srv;
  srv.request.cost_mat_flat = cost_mat_flat;
  srv.request.priorities = priorities;
  srv.request.dim = dimension;

  // std::vector<int> tour;
  if (tsp_client.call(srv)){
    tour = srv.response.tour;
    assert(tour.size() == dimension-1);
  }
  else{
    ROS_ERROR("Failed to call service add_two_ints");
  }
}

/* Gets AStar path to the next position*/
double FastExplorationManager::getPathToView(const Eigen::Vector3d& pos, Eigen::Vector3d& next_pos){
  // Generate trajectory of x,y,z
  planner_manager_->path_finder_->reset();
  if (planner_manager_->path_finder_->search(pos, next_pos) != Astar::REACH_END) {
    ROS_ERROR("No path to next viewpoint");
    return FAIL;
  }
  ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();
  shortenPath(ed_->path_next_goal_);

  const double len = Astar::pathLength(ed_->path_next_goal_);

  return len;
}


int FastExplorationManager::getTrajToView(const Eigen::Vector3d& pos,  const Eigen::Vector3d& vel, const Eigen::Vector3d& acc, const Eigen::Vector3d& yaw, Eigen::Vector3d& next_pos, double next_yaw, const double path_length){

  // Compute time lower bound of yaw and use in trajectory generation
  double diff = fabs(next_yaw - yaw[0]);
  double time_lb_yaw = min(diff, 2 * M_PI - diff) / ViewNode::yd_;
  double time_lb_pos = (pos-next_pos).norm()/ViewNode::vm_;

  double time_lb = time_lb_yaw;

  // // Generate trajectory of x,y,z
  // planner_manager_->path_finder_->reset();
  // if (planner_manager_->path_finder_->search(pos, next_pos) != Astar::REACH_END) {
  //   ROS_ERROR("No path to next viewpoint");
  //   return FAIL;
  // }
  // ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();
  // shortenPath(ed_->path_next_goal_);

  const double radius_far = 3.0;
  const double radius_close = 1.5;  // don't change this below 1.5 or you'll observe a alot of kinodynamic search failures
  const double radius_very_close = 0.2;
  // const double len = Astar::pathLength(ed_->path_next_goal_);
  // if (path_length<radius_very_close){
  //   ROS_WARN("veru close");
  //   time_lb = time_lb_pos;
  //   if (ed_->path_next_goal_.size()>1)
  //     planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb, true); // update trajectory in place
    
  //   ed_->next_goal_ = next_pos;
  // // return SUCCEED;
  // }
  if (path_length < radius_close) {
    // Next viewpoint is very close, no need to search kinodynamic path, just use waypoints-based
    // optimization
    if (ed_->path_next_goal_.size()>1){
      planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb);
      // if (planner_manager_->planExploreTraj(ed_->path_next_goal_, vel, acc, time_lb)<0) // update trajectory in place
      //   return FAIL;
    }
      
    
    ed_->next_goal_ = next_pos;

  } 
  
  else if (path_length > radius_far) { 
    // Next viewpoint is far away, select intermediate goal on geometric path (this also deal with
    // dead end)
    std::cout << "Far goal." << std::endl;
    double len2 = 0.0;

    // cut the path until the cumulative path length stays within the max radius
    vector<Eigen::Vector3d> truncated_path = { ed_->path_next_goal_.front() };
    for (uint i = 1; i < ed_->path_next_goal_.size() && len2 < radius_far; ++i) {
      auto cur_pt = ed_->path_next_goal_[i];
      len2 += (cur_pt - truncated_path.back()).norm();
      truncated_path.push_back(cur_pt);
    }
    ed_->next_goal_ = truncated_path.back();
    planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb);
    // if (planner_manager_->planExploreTraj(truncated_path, vel, acc, time_lb)<0)
    //   return FAIL;
      
  } 
  else {
    // Search kino path to exactly next viewpoint and optimize
    std::cout << "Mid goal" << std::endl;
    ed_->next_goal_ = next_pos;

    if (!planner_manager_->kinodynamicReplan(pos, vel, acc, ed_->next_goal_, Vector3d(0, 0, 0), time_lb))
      return FAIL;
  }
  double planned_time = planner_manager_->local_data_.position_traj_.getTimeSum();
  if (planned_time < (time_lb - 0.5)){
    ROS_ERROR("Lower bound not satified! planned: %f , estimated: %f", planned_time, time_lb-0.1);
    // return FAIL;
  }
  planner_manager_->planYawExplore(yaw, next_yaw, true, ep_->relax_time_);
  // if (planner_manager_->planYawExplore(yaw, next_yaw, true, ep_->relax_time_)<0)
  //   return FAIL;

  return SUCCEED;
}

void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // Shorten the tour, only critical intermediate points are reserved.
  const double dist_thresh = 2.0;
  vector<Vector3d> short_tour = { path.front() };
  for (uint i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints to shorten path only to avoid collision
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());

  // Ensure at least three points in the path
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  path = short_tour;
}

void FastExplorationManager::findGlobalTour(const Eigen::MatrixXd& cost_mat, vector<uint8_t>& indices) {
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  // Eigen::MatrixXd cost_mat;
  // frontier_finder_->getFullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat);
  // const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();
  
  // Write matrix to file
  lkh_interface::writeCostMatToFile(cost_mat, ep_->tsp_dir_ + "/single.tsp");

  // Call LKH TSP solver
  lkh_interface::solveTSPLKH((ep_->tsp_dir_ + "/single.par").c_str());
  
  // Read tour  
  lkh_interface::readTourFromFile(indices, ep_->tsp_dir_+ "/single.txt");
  
  // // Get the path of optimal tour from path matrix
  // frontier_finder_->getPathForTour(cur_pos, indices, ed_->global_tour_);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}


// void addTargetstoCostMatrix(Eigen::MatrixXd cost_mat){
//   const int dim = cost_mat.rows()
//   Eigen::MatrixXd new_mat;
  
//   new_mat.resize(dim + target_vpts.size(), dim + target_vpts.size());
//   new_mat.block(0,0,dim, dim).array() = cost_mat;

//   for (int i=dim; i<dim + target_vpts.size(); )

// }

void FastExplorationManager::getTargetCostMatrix(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw, Eigen::MatrixXd& cost_mat){
      // Eigen::MatrixXd cost_mat;
      int dim = target_vpts.size()+1;
      cost_mat.resize(dim, dim);
      cost_mat.setZero();
      
      Eigen::Vector3d pos_i, vel_i, pos_j, vel_j;
      double yaw_i, yaw_j;
      geometry_msgs::Pose pose_i, pose_j;
      std::cout <<dim<<std::endl;
      target_paths.resize(target_vpts.size());
      for (auto& vec: target_paths){
        vec.resize(0);
      }
      
      // Costs between target viewpoints
      for (int i=0; i<dim; i++){
        for (int j=i+1; j<dim; j++){
          pose_j = target_vpts[j-1];
          pos_j = Eigen::Vector3d(pose_j.position.x, pose_j.position.y, pose_j.position.z);

          geometry_msgs::Quaternion q = pose_j.orientation;
          double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
          double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
          yaw_j = std::atan2(siny_cosp, cosy_cosp);
          vector<Vector3d> path_ij;
          // std::cout << i<<", "<<j<<" | ";
          if (i==0){ // jth target to current pose
              // Assymetric TSP
              cost_mat(i, j) = ViewNode::computeCost(cur_pos, pos_j, cur_yaw[0], yaw_j, cur_vel, cur_yaw[1], path_ij);
          }
          else{ // jth target to ith target
            
            pose_i = target_vpts[i-1];            
            
            geometry_msgs::Quaternion q = pose_i.orientation;
            double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
            double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);

            pos_i = Eigen::Vector3d(pose_i.position.x, pose_i.position.y, pose_i.position.z);
            yaw_i = std::atan2(siny_cosp, cosy_cosp);
            vel_i = Eigen::Vector3d(0, 0, 0);

            cost_mat(i, j) = ViewNode::computeCost(pos_i, pos_j, yaw_i, yaw_j, vel_i, 0, path_ij);
            cost_mat(j, i) = cost_mat(i, j);
            // std::cout<<pos_i.transpose()<< "| "<< pos_j.transpose() <<std::endl;
            target_paths[i-1].push_back(path_ij);
            reverse(path_ij.begin(), path_ij.end());
            target_paths[j-1].push_back(path_ij);
          }
        }
      }    
}



// void FastExplorationManager::findTargetTour(Eigen::Matrix& cost_mat, vector<int>& indices){

//        // solve the TSP and read the tour as indices
//       solveTSPAndGetTour(cost_mat, indices, "/root/thesis_ws/src/thesis/sw/bringup/resource");


//       // Get the path of optimal tour from path matrix
//       getPathForTour(cur_pos, indices, ed_->global_tour_);

// }

void FastExplorationManager::getPathForTour(const Vector3d& pos, const vector<uint8_t>& ids, vector<Vector3d>& expanded_path){
  
  
  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  geometry_msgs::Pose pose = target_vpts[ids[0]];
  Eigen::Vector3d vpt_pos = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  ViewNode::searchPath(pos, vpt_pos, segment);
  expanded_path.insert(expanded_path.end(), segment.begin(), segment.end());
  // ROS_WARN("path1 len: %d", expanded_path.size());
  // Get paths of tour passing all clusters
  for (int i = 0; i < ids.size() - 1; ++i) {
    // Move to path to next cluster
    // auto path_iter = target_paths[ids[i]].begin();
    int next_idx = ids[i + 1];
    vector<Vector3d> segment;
    if (next_idx > ids[i]){ // because paths to self are never saved. So matrix is 1 less than actual size. this just compensates for that
      segment = target_paths[ids[i]][next_idx-1];
    }
    else{
      segment =  target_paths[ids[i]][next_idx];
    }
    expanded_path.insert(expanded_path.end(), segment.begin(), segment.end());
  }
}

void FastExplorationManager::refineLocalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_yaws,
    vector<Vector3d>& refined_pts, vector<double>& refined_yaws) {
  double create_time, search_time, parse_time;
  auto t1 = ros::Time::now();

  // Create graph for viewpoints selection
  GraphSearch<ViewNode> g_search;
  vector<ViewNode::Ptr> last_group, cur_group;

  // Add the current state
  ViewNode::Ptr first(new ViewNode(cur_pos, cur_yaw[0]));
  first->vel_ = cur_vel;
  g_search.addNode(first);
  last_group.push_back(first);
  ViewNode::Ptr final_node;

  // Add viewpoints
  std::cout << "Local tour graph: ";
  for (int i = 0; i < n_points.size(); ++i) {
    // Create nodes for viewpoints of one frontier
    for (int j = 0; j < n_points[i].size(); ++j) {
      ViewNode::Ptr node(new ViewNode(n_points[i][j], n_yaws[i][j]));
      g_search.addNode(node);
      // Connect a node to nodes in last group
      for (auto nd : last_group)
        g_search.addEdge(nd->id_, node->id_);
      cur_group.push_back(node);

      // Only keep the first viewpoint of the last local frontier
      if (i == n_points.size() - 1) {
        final_node = node;
        break;
      }
    }
    // Store nodes for this group for connecting edges
    std::cout << cur_group.size() << ", ";
    last_group = cur_group;
    cur_group.clear();
  }
  std::cout << "" << std::endl;
  create_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Search optimal sequence
  vector<ViewNode::Ptr> path;
  g_search.DijkstraSearch(first->id_, final_node->id_, path);

  search_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Return searched sequence
  for (int i = 1; i < path.size(); ++i) {
    refined_pts.push_back(path[i]->pos_);
    refined_yaws.push_back(path[i]->yaw_);
  }

  // Extract optimal local tour (for visualization)
  ed_->refined_tour_.clear();
  ed_->refined_tour_.push_back(cur_pos);
  ViewNode::astar_->lambda_heu_ = 1.0;
  ViewNode::astar_->setResolution(0.2);
  for (auto pt : refined_pts) {
    vector<Vector3d> path;
    if (ViewNode::searchPath(ed_->refined_tour_.back(), pt, path))
      ed_->refined_tour_.insert(ed_->refined_tour_.end(), path.begin(), path.end());
    else
      ed_->refined_tour_.push_back(pt);
  }
  ViewNode::astar_->lambda_heu_ = 10000;

  parse_time = (ros::Time::now() - t1).toSec();
  // ROS_WARN("create: %lf, search: %lf, parse: %lf", create_time, search_time, parse_time);
}

void FastExplorationManager::targetViewpointsCallback(const common_msgs::Viewpoints& msg){
  target_vpts = msg.viewpoints.poses;
  priorities = msg.priorities;
}

void FastExplorationManager::customPoseCallback(const geometry_msgs::PoseWithCovarianceStamped& msg){
  custom_goal_pose = msg.pose.pose;
}

}  // namespace fast_planner
