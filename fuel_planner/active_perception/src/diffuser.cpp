#include <active_perception/diffuser.h>
#include <active_perception/frontier_finder.h>
#include <common/io.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chrono>
#include <cmath>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
typedef std::chrono::duration<double> time_diff;


/* for precomputing the 3D Gaussian kernel weights */ 
std::vector<std::vector<std::vector<float>>> calculateGaussianKernel(int size, double sigma, bool normalize=false) {
    std::vector<std::vector<std::vector<float>>> weights(size, std::vector<std::vector<float>>(size, std::vector<float>(size, 0.0)));

    double normalization = 0.0;
    for (int i = -size / 2; i <= size / 2; ++i)
        for (int j = -size / 2; j <= size / 2; ++j)
            for (int k = -size / 2; k <= size / 2; ++k) {
                float weight = std::exp(-(i * i + j * j + k * k) / (2 * sigma * sigma));
                weights[i + size / 2][j + size / 2][k + size / 2] = weight;
                normalization += weight;
            }

    if (normalize){
        for (int i = 0; i < size; ++i) 
            for (int j = 0; j < size; ++j) 
                for (int k = 0; k < size; ++k)
                    weights[i][j][k] /= normalization;
    }
    
    return weights;
}   

Diffuser::Diffuser(const shared_ptr<fast_planner::EDTEnvironment>& edt, ros::NodeHandle& nh){
    // References
    _att_map = edt->_att_map;
    _sdf_map = edt->sdf_map_;

    // Params
    nh.param("/diffusion/sigma", _kernel_sigma, 2); 
    nh.param("/diffusion/kernel_size", _kernel_size, 3); 
    _att_min = _att_map->getAttMin();

    // member variables
    diffusion_buffer = std::vector<float>(_sdf_map->buffer_size, 0.0f);
    _kernel_weights = calculateGaussianKernel(_kernel_size, _kernel_sigma, false);
    _kernel_depth = (_kernel_size - 1)/2;

    // ROS
    // _diffusion_timer = nh.createTimer(ros::Duration(0.1), &Diffuser::diffusionTimer, this);
    _map_pub = nh.advertise<sensor_msgs::PointCloud2>("/priority_map/diffused", 1);
}

void Diffuser::setFrontierFinder(std::shared_ptr<fast_planner::FrontierFinder>& ff_ptr){
    _ff = ff_ptr;
}

/*Get partial convolution filter value around a single voxel using kernel weights.
The partial convolution renormalises the weights for valid voxels. good for sparse data.
*/
float Diffuser::partialConvolution(const Eigen::Vector3i& voxel){
    /* 
    find dense neibhours in kernel size
    for each nbr
        sum = (top down priority) + (bottup diffusion)
        if sum > min
            weight the nbr using kernel weight
            count++
    convolved value = weighted sum/count
    */

    auto nbrs = _ff->allNeighbors(voxel, _kernel_depth); // 124 neighbors
    float att_nbr = 0.0; 
    float norm = 0;
    Eigen::Vector3d nbr_pos;
    for (auto nbr : nbrs) {
        _sdf_map->indexToPos(nbr, nbr_pos);
        int nbr_adr = _sdf_map->toAddress(nbr);

        if (!_sdf_map->isInMap(nbr) || nbr_pos(2)<=0.2)
            continue;

        float attention = _att_map->priority_buffer[nbr_adr] + diffusion_buffer[nbr_adr];
        if (attention < _att_min) 
            continue;

        Eigen::Vector3i idx = (nbr - voxel).cwiseAbs();
        float weight = _kernel_weights[idx(0)+ _kernel_depth][idx(1)+ _kernel_depth][idx(2)+ _kernel_depth]; 
        att_nbr += weight*attention;
        norm += weight;   // for partial convolution renormalisation
    }
    float att_diffused = (norm>0)? att_nbr/norm:_att_min; // clip at minimum attention for coverage 
    return att_diffused;
}

/*Diffuse the priority buffer into nearby frontier voxels*/
void Diffuser::diffusionTimer(const ros::TimerEvent& event){
    // time_point start = std::chrono::high_resolution_clock::now();
    // time_diff dt;

    // Only do calcs in a local window around robot, to save compute
    Eigen::Vector3i min_cut = _sdf_map->md_->local_bound_min_;
    Eigen::Vector3i max_cut = _sdf_map->md_->local_bound_max_;

    _sdf_map->boundIndex(min_cut);
    _sdf_map->boundIndex(max_cut);

    for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y)
            for (int z = 2; z < _sdf_map->mp_->box_max_(2); ++z) {
                
                int adr = _sdf_map->toAddress(x,y,z);            

                // diffusion map only contains frontiers
                if (_ff->frontier_flag_[adr]==0|| z<=2){ // z filter removes noise close to ground and avoids getting stuck
                    diffusion_buffer[adr]=0;
                    continue;
                }
                // get convolved filter value for this frontier voxel
                float att_conv = partialConvolution(Eigen::Vector3i(x,y,z));
                
                // the weighted update just allows it to stabilise a bit, otherwise a lot of flickering
                diffusion_buffer[adr] = 0.9f*att_conv + 0.1f*diffusion_buffer[adr];
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
    for (uint i=0; i<diffusion_buffer.size(); i++){
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