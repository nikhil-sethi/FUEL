#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <plan_env/edt_environment.h>
#include <plan_env/attention_map.h>
#include <active_perception/frontier_finder.h>

class fast_planner::SDFMap;

class Diffuser{
    public:
        Diffuser(const shared_ptr<fast_planner::EDTEnvironment>& edt, ros::NodeHandle& nh);
        void setFrontierFinder(std::shared_ptr<fast_planner::FrontierFinder>& ff_ptr);
        std::vector<float> diffusion_buffer;
        void inputPointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud);

    private:
        // Member functions
        void diffusionTimer(const ros::TimerEvent& event);
        void publishDiffusionMap();
        float partialConvolution(const Eigen::Vector3i& voxel);
        void updatePriority(Eigen::Vector3d pos, float new_priority);

        // Member variables
        std::shared_ptr<AttentionMap> _att_map;
        std::shared_ptr<fast_planner::SDFMap> _sdf_map;
        std::shared_ptr<fast_planner::FrontierFinder> _ff;
        std::vector<std::vector<std::vector<float>>> _kernel_weights;

        ros::Timer _diffusion_timer;
        ros::Publisher _map_pub;

        // float _diffusion_factor;
        float _att_min, _learning_rate;
        int _kernel_sigma;
        int _kernel_size;
        int _kernel_depth;
};

#endif