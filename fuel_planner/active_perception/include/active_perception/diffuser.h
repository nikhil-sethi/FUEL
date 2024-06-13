#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <plan_env/edt_environment.h>
#include <plan_env/priority_map.h>
#include <active_perception/frontier_finder.h>

class fast_planner::SDFMap;
class DiffusionMapGT;
class Diffuser{
    public:
        Diffuser(ros::NodeHandle& nh);
        void setFrontierFinder(std::shared_ptr<fast_planner::FrontierFinder>& ff_ptr);
        void setPriorityMap(std::shared_ptr<PriorityMap>& pm_ptr);
        void setSDFMap(std::shared_ptr<fast_planner::SDFMap>& sdf_ptr);

        void diffusionTimer(const ros::TimerEvent& event);
        std::vector<float> diffusion_buffer;
        friend DiffusionMapGT;
    private:
        // Member functions
        
        void publishDiffusionMap();
        float partialConvolution(const Eigen::Vector3i& voxel);

        // Member variables
        std::shared_ptr<PriorityMap> _att_map;
        std::shared_ptr<fast_planner::SDFMap> _sdf_map;
        std::shared_ptr<fast_planner::FrontierFinder> _ff;
        std::vector<std::vector<std::vector<float>>> _kernel_weights;

        ros::Timer _diffusion_timer;
        ros::Publisher _map_pub;

        // float _diffusion_factor;
        float _att_min;
        int _kernel_sigma;
        int _kernel_size;
        int _kernel_depth;
};

#endif