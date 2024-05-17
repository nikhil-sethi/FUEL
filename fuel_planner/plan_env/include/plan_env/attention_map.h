#ifndef ATTENTION_MAP_H
#define ATTENTION_MAP_H

#include <ros/ros.h>
#include <vector>
#include <plan_env/sdf_map.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Eigen>


class AttentionMap{
    public:
        AttentionMap(){};
        ~AttentionMap(){};
        
        void init(ros::NodeHandle& nh);
        void setSDFMap(std::shared_ptr<fast_planner::SDFMap> sdf_map_ptr);
        void inputPointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud);
        float getAttMin(){return _att_min;}

        std::vector<float> priority_buffer;


    private:
        // Member functions
        void updatePriority(Eigen::Vector3d pos, float value);
        void publishMapTimer(const ros::TimerEvent& event);

        // Member variables
        std::shared_ptr<fast_planner::SDFMap> _map;
        pcl::PointCloud<pcl::PointXYZI>::Ptr global_cloud;

        ros::Timer _diffusion_timer, _pub_map_timer;
        ros::Publisher _map_pub;

        // Parameters
        float _att_min;
        float _learning_rate;

};

#endif