#include <plan_env/edt_environment.h>
#include <plan_env/attention_map.h>
#include <active_perception/frontier_finder.h>

class fast_planner::SDFMap;

class Diffuser{
    public:
        Diffuser(const shared_ptr<fast_planner::EDTEnvironment>& edt, ros::NodeHandle& nh);
        void setFrontierFinder(std::shared_ptr<fast_planner::FrontierFinder>& ff_ptr);
        std::vector<float> diffusion_buffer;

    private:
        void diffusionTimer(const ros::TimerEvent& event);
        void publishDiffusionMap();

        float diffuse(const Eigen::Vector3i& bu_voxel);

        std::shared_ptr<AttentionMap> _att_map;
        std::shared_ptr<fast_planner::SDFMap> _sdf_map;
        std::shared_ptr<fast_planner::FrontierFinder> _ff;
        // diffuse()

        ros::Timer _diffusion_timer;
        ros::Publisher _map_pub;

        float _diffusion_factor;
        float _att_min;
};

