#ifndef TGT_PLNR_H
#define TGT_PLNR_H

#include <eigen3/Eigen/Eigen>
#include <tf/transform_datatypes.h>
#include <active_perception/frontier_finder.h>
#include <common/utils.h>

struct TargetViewpoint: fast_planner::Viewpoint{
    float gain_;

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

    // common_msgs::Viewpoint toMsg(){
    //     target_search::Viewpoint msg;
        
    //     msg.position.x = pos_(0);
    //     msg.position.y = pos_(1);
    //     msg.position.z = pos_(2);
        
    //     msg.yaw = yaw_;
    //     msg.priority = gain_;
        
    //     return msg;
    // }  

    bool isClose(const TargetViewpoint& other){
        double dpsi = yaw_-other.yaw_;
        return (pos_-other.pos_).norm() < 0.4 && abs(std::min(dpsi, 2*M_PI-dpsi))<0.4;
    }

    Eigen::Vector4d getColor(float min, float max, Eigen::Matrix<double,4,4> colormap);

};

class TargetPlanner{
    public:
        void setObjectFinder(std::shared_ptr<ObjectFinder> obj_fnd_ptr){_obj_fnd = obj_fnd_ptr;}
        void setDiffusionMap(std::shared_ptr<Diffuser> diff_map_ptr){_diff_map = diff_map_ptr;}   

    private:
        


}


#endif