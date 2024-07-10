#ifndef TGT_PLNR_H
#define TGT_PLNR_H

#include <ros/ros.h>
#include <eigen3/Eigen/Eigen>
#include <tf/transform_datatypes.h>
#include <active_perception/frontier_finder.h>
#include <active_perception/object_finder.h>
#include <active_perception/diffuser.h>
#include <common/utils.h>
#include <common_msgs/Viewpoints.h>
#include <sensor_model/camera.h>
#include <plan_env/raycast.h>
#include <active_perception/perception_utils.h>
// #include <traj_utils/planning_visualization.h>

class Object;

struct TargetViewpoint: fast_planner::Viewpoint{
    float gain_;
    
    TargetViewpoint(){}
    ~TargetViewpoint(){}
    
    TargetViewpoint(Eigen::Vector3d pos, double yaw, float gain=0){
        pos_ = pos;
        yaw_ = yaw;
        gain_ = gain;
    }

    Eigen::Vector4d poseToEigen(){
        return Eigen::Vector4d(pos_(0), pos_(1),pos_(2), yaw_);
    }
    Eigen::Vector3d posToEigen(){
        return Eigen::Vector3d(pos_(0), pos_(1),pos_(2));
    }

    geometry_msgs::Pose toGeometryMsg(){
        geometry_msgs::Pose msg;
        
        msg.position.x = pos_(0);
        msg.position.y = pos_(1);
        msg.position.z = pos_(2);
        
        msg.orientation = rpyToQuaternionMsg(0, 0, yaw_);
        
        return msg;
    }   

    bool isClose(const TargetViewpoint& other){
        double dpsi = yaw_-other.yaw_;
        return (pos_-other.pos_).norm() < 0.4 && abs(std::min(dpsi, 2*M_PI-dpsi))<0.7;
    }
};

class TargetPlanner{
    public:
        // TargetPlanner(ros::NodeHandle& nh);
        TargetPlanner(){}
        ~TargetPlanner(){}
        void setObjectFinder(std::shared_ptr<ObjectFinder> obj_fnd_ptr){_obj_fnd = obj_fnd_ptr;}
        void setFrontierFinder(std::shared_ptr<fast_planner::FrontierFinder> ftr_fndr_ptr){_ftr_fndr = ftr_fndr_ptr;}
        void setDiffusionMap(std::shared_ptr<Diffuser> diff_map_ptr){_diff_map = diff_map_ptr;} 
        void setSDFMap(std::shared_ptr<fast_planner::SDFMap> sdf_map_ptr){_sdf_map = sdf_map_ptr;}
        void init(ros::NodeHandle& nh);

        std::list<std::vector<TargetViewpoint>> all_viewpoints;

    private:
        void informationGainTimer(const ros::TimerEvent& event);
        void sampleViewpoints(Object& object, std::vector<TargetViewpoint>& sampled_vpts);
        void findTopViewpoints(Object& object, std::vector<TargetViewpoint>& sampled_vpts);
        float computeInformationGain(Object& object, const Eigen::Vector3d& sample_pos, double yaw);
        bool isObjectInView(const Object& object, const Eigen::Vector3d& pos, const Eigen::Vector3d& orient);
        void sortViewpoints(std::vector<TargetViewpoint>& vpts);
        float infoTransfer(float gain);
        void filterSimilarPoses(std::list<std::vector<TargetViewpoint>>& myList);
        void publishTargetViewpoints();
        bool isPtInView(const Eigen::Vector3d& point_world, const Eigen::Vector3d& pos, double yaw);

        // References
        std::shared_ptr<ObjectFinder> _obj_fnd;
        std::shared_ptr<fast_planner::FrontierFinder> _ftr_fndr;
        std::shared_ptr<Diffuser> _diff_map;
        std::shared_ptr<fast_planner::SDFMap> _sdf_map;
        // fast_planner::PlanningVisualization viz;

        // Member variables
        std::unique_ptr<Camera> _camera;
        std::unique_ptr<RayCaster> _raycaster;
        std::unique_ptr<fast_planner::PerceptionUtils> _percep_utils;
        
        ros::Publisher vpt_pub;
        ros::Timer info_timer;
        common_msgs::Viewpoints vpts_msg;
        float _rmin, _min_vpt_clearance, _att_min, _min_info_gain;    
        Eigen::MatrixXd colormap;
};


#endif