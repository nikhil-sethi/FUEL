#include <active_perception/diffuser.h>
#include <active_perception/frontier_finder.h>
#include <common/io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
typedef std::chrono::duration<double> time_diff;

Diffuser::Diffuser(const shared_ptr<fast_planner::EDTEnvironment>& edt, ros::NodeHandle& nh){
    _att_map = edt->_att_map;
    _sdf_map = edt->sdf_map_;
    // _ff = edt->frontier_finder_;

    diffusion_buffer = std::vector<float>(_sdf_map->buffer_size, 0);

    // Params
    nh.param("/perception/attention_map/3d/diffusion_factor", _diffusion_factor, 0.9f); 
    _att_min = _att_map->getAttMin();
    _diffusion_timer = nh.createTimer(ros::Duration(0.1), &Diffuser::diffusionTimer, this);
    _map_pub = nh.advertise<sensor_msgs::PointCloud2>("/attention_map/diffused", 1);
}

void Diffuser::setFrontierFinder(std::shared_ptr<fast_planner::FrontierFinder>& ff_ptr){
    _ff = ff_ptr;
}

// Get diffused value from nearby voxels. Filters can be used
float Diffuser::diffuse(const Eigen::Vector3i& bu_voxel){
    auto nbrs = _ff->allNeighbors(bu_voxel, 2); // 124 neighbors
    float att_nbr = 0.0; 
    float count = 0;
    Eigen::Vector3d nbr_pos;
    for (auto nbr : nbrs) {
        _sdf_map->indexToPos(nbr, nbr_pos);
        int nbr_adr = _sdf_map->toAddress(nbr);

        if (!_sdf_map->isInMap(nbr) || nbr_pos(2)<0.1)
            continue;

        float attention = _att_map->priority_buffer[nbr_adr] + diffusion_buffer[nbr_adr];
        if (attention < _att_min) continue;
        
        att_nbr += attention;
        count++;    
    }
    float att_diffused = (count>0)? _diffusion_factor*att_nbr/count:0;
    return att_diffused;
}


/*Diffuse the priority buffer into nearby frontier voxels*/
void Diffuser::diffusionTimer(const ros::TimerEvent& event){
    /*
        for each frontier
            if frontier not near object: skip
            for each froniter cell 
                find dense neighbors
                for each nbr
                    sum attention
                attention at cell = discount*sum
    
    */

    // time_point start = std::chrono::high_resolution_clock::now();
    // time_diff dt;

    Eigen::Vector3i min_cut = _sdf_map->md_->local_bound_min_;
    Eigen::Vector3i max_cut = _sdf_map->md_->local_bound_max_;

    // std::fill(diffusion_buffer.begin(), diffusion_buffer.end(), 0); // TODO make this local 
    _sdf_map->boundIndex(min_cut);
    _sdf_map->boundIndex(max_cut);

    // print(max_cut-min_cut);
    // int sum = std::accumulate(_ff->frontier_flag_.begin(), _ff->frontier_flag_.end(), 0);
    // print(sum);
    for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y)
            for (int z = _sdf_map->mp_->box_min_(2); z < _sdf_map->mp_->box_max_(2); ++z) {
                
                int adr = _sdf_map->toAddress(x,y,z);
                // float attention = _att_map->priority_buffer[adr];
                
                if (_ff->frontier_flag_[adr]==0){ // this is the main thing that saves compute
                    diffusion_buffer[adr]=0;
                    continue;
                }
                
                // if (==1) {
                    float att_diffused = diffuse(Eigen::Vector3i(x,y,z)); // filter value
                    // print(att_diffused);
                    diffusion_buffer[adr] = att_diffused > _att_min?  (0.9*att_diffused + 0.1*diffusion_buffer[adr]):0;     // the weighted update just allows it to stabilise a bit, otherwise a lot of flickering
                // }
            }

    // time_point end = std::chrono::high_resolution_clock::now();
    // dt = end-start;
    // ROS_WARN("Time for diffusion: %f", dt);

    // time_point end2 = std::chrono::high_resolution_clock::now();
    // dt = end2-end;
    publishDiffusionMap();
    // ROS_WARN("Time for pub: %f", dt);

}

void Diffuser::publishDiffusionMap(){
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointXYZI pcl_pt;
    Eigen::Vector3d pos;
    for (int i=0; i<diffusion_buffer.size(); i++){
        float diff = diffusion_buffer[i];

        if (diff<=0.0f) // this is the main thing that saves compute
            continue;

        _sdf_map->indexToPos(i, pos);
        pcl_pt.x = pos[0];
        pcl_pt.y = pos[1];
        pcl_pt.z = pos[2];
        pcl_pt.intensity = diff;
        cloud.push_back(pcl_pt);  

    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.header.frame_id = "map";
    
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    _map_pub.publish(cloud_msg);

}   