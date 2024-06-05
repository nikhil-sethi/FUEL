#ifndef OBJ_FNDR_H
#define OBJ_FNDR_H

#include <list>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <plan_env/priority_map.h>
#include <plan_env/sdf_map.h>
#include <active_perception/diffuser.h>
#include <pcl/point_types.h>
#include <active_perception/target_planner.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <common/utils.h>

class TargetViewpoint;

struct Object{
    int id;
    // pcl::PointCloud<pcl::PointXYZI> points;
    float max_gain;
    Eigen::Vector3d bbox_min_;
    Eigen::Vector3d bbox_max_;
    Eigen::Vector3d centroid_;
    Eigen::Vector3d scale_;
    std::vector<TargetViewpoint> viewpoints;
    std::vector<Eigen::Vector3d> projection_corners;

    // initialise projection corners to fixed length
    Object():projection_corners(8){}
    Object(const pcl::PointCloud<pcl::PointXYZI>& points):projection_corners(8){
        if (points.size() == 0) return;
        pcl::PointXYZI bbox_min;
        pcl::PointXYZI bbox_max;
        pcl::getMinMax3D(points, bbox_min, bbox_max); 

        bbox_min_ = Eigen::Vector3d(bbox_min.x-0.02, bbox_min.y-0.02, bbox_min.z-0.02);
        bbox_max_ = Eigen::Vector3d(bbox_max.x+0.02, bbox_max.y+0.02, bbox_max.z+0.02);

        computeInfo();
    }

    void computeInfo(){
        centroid_ = (bbox_min_ + bbox_max_)/2.0;
        scale_ = bbox_max_ - bbox_min_;
        computeBboxCorners();
    } 

    void computeBboxCorners(){
        // points 
        projection_corners[0] = bbox_min_;
        projection_corners[1] = bbox_min_ + Vector3d(scale_(0), 0, 0);
        projection_corners[2] = bbox_min_ + Vector3d(0, scale_(1), 0);
        projection_corners[3] = bbox_min_ + Vector3d(scale_(0), scale_(1), 0);
        
        projection_corners[4] = bbox_max_;
        projection_corners[5] = bbox_max_ - Vector3d(scale_(0), 0, 0);
        projection_corners[6] = bbox_max_ - Vector3d(0, scale_(1), 0);
        projection_corners[7] = bbox_max_ - Vector3d(scale_(0), scale_(1), 0);
    
    }

    // merge bbox corners
    void merge(const Object& other){
        bbox_min_ = bbox_min_.cwiseMin(other.bbox_min_);
        bbox_max_ = bbox_max_.cwiseMax(other.bbox_max_);
        computeInfo(); // update
    }

    bool isInBox(Eigen::Vector3d box_min, Eigen::Vector3d box_max){
        return isPtInBox(bbox_min_, box_min, box_max) && isPtInBox(bbox_max_, box_min, box_max);
    }

};


class ObjectFinder{
    public:
        ObjectFinder(ros::NodeHandle& nh);

        void setPriorityMap(std::shared_ptr<PriorityMap> att_map_ptr){_prio_map = att_map_ptr; _att_min = _prio_map->getAttMin();}
        void setSDFMap(std::shared_ptr<fast_planner::SDFMap> sdf_map_ptr){_sdf_map = sdf_map_ptr;}

        std::list<Object> global_objects;

    private:
        // Member functions
        void objectFusionTimer(const ros::TimerEvent& e);
        void createObjects(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::list<Object>& objects);
        void mergeObjects(std::list<Object> new_objects);
        bool areObjectsSimilar(const Object& obj_a, const Object& obj_b, double overlapThreshold);
        void publishObjectBoxes();

        // References
        std::shared_ptr<PriorityMap> _prio_map;
        std::shared_ptr<Diffuser> _diff_map;
        std::shared_ptr<fast_planner::SDFMap> _sdf_map;
        
        // Member variables
        // pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud;
        std::function<bool(const pcl::PointXYZI&, const pcl::PointXYZI&, float)> _cluster_condition;
        ros::Timer _obj_update_timer;

        // Parameters
        float _global_iou_thresh, _local_iou_thresh, _att_min;
        

};

#endif