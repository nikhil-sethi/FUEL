#include <plan_env/priority_map.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>

void PriorityMap::init(ros::NodeHandle& nh){

    // Initialise variables
    priority_buffer = std::vector<float>(_map->buffer_size, 0);
    global_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    global_cloud->height = 1;
    global_cloud->is_dense = true;
    global_cloud->header.frame_id = "map";

    // ROS Parameters    
    nh.param("/priority_map/pmin", _att_min, 1.0f); 
    // nh.param("/perception/priority_map/3d/diffusion_factor", _diffusion_factor, 0.9f); 
    nh.param("/priority_map/learning_rate", _learning_rate, 0.5f); 

    // ROS pub, sub, timers    
    _map_pub = nh.advertise<sensor_msgs::PointCloud2>("/priority_map/global", 1);
    _pub_map_timer = nh.createTimer(ros::Duration(0.1), &PriorityMap::publishMapTimer, this);
}

void PriorityMap::setSDFMap(std::shared_ptr<fast_planner::SDFMap> sdf_map_ptr){
    _map = sdf_map_ptr;
}

/* Update voxel using new measurement
Currently just weighted update. 
Can be log odds as well in the future
*/
void PriorityMap::updatePriority(Eigen::Vector3d pos, float new_priority){
    Eigen::Vector3i idx;
    _map->posToIndex(pos, idx);
    if (!_map->isInMap(idx)) 
        return;
    
    // only update occupied cells
    if (!(_map->getOccupancy(idx) == fast_planner::SDFMap::OCCUPIED)) 
        return; 

    // perform weighted update
    int vox_adr = _map->toAddress(idx);
    priority_buffer[vox_adr] = _learning_rate*new_priority + (1-_learning_rate)*priority_buffer[vox_adr];
}

/* Update the priority buffer using the depth intensity point cloud*/
void PriorityMap::inputPointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud){
    // ===== Cleanup ====
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    *filtered_cloud = cloud;
    // ===== Intensity pass through
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(filtered_cloud);
    pass.setFilterFieldName("intensity");
    pass.setFilterLimits(_att_min, 10.0); 
    pass.filter(*filtered_cloud);

    if (filtered_cloud->size()>0){
        // ====== Density filtering =======
        pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;

        outrem.setInputCloud(filtered_cloud);
        outrem.setRadiusSearch(0.2);
        outrem.setMinNeighborsInRadius (3);
        outrem.setKeepOrganized(true);
        outrem.filter (*filtered_cloud);

        pcl::VoxelGrid<pcl::PointXYZI> sor;
        sor.setInputCloud (filtered_cloud);
        sor.setLeafSize (0.08f, 0.08f, 0.08f);
        sor.filter (*filtered_cloud);
    }


    // ======= Global buffer update ======

    // update the global buffer. Very small loop
    Eigen::Vector3d pos;
    
    for (auto& pt: filtered_cloud->points){
        // float priority = pt.intensity;
        // if (priority<_att_min) continue;

        pos[0] = pt.x;
        pos[1] = pt.y;
        pos[2] = pt.z;

        updatePriority(pos, pt.intensity);
    }

}

void PriorityMap::publishMapTimer(const ros::TimerEvent& event){

    global_cloud->points.clear();
    pcl::PointXYZI pcl_pt;
    Eigen::Vector3d pos;
    for (int i=0; i<priority_buffer.size(); i++){
        float priority = priority_buffer[i];

        if (priority<=0.0f) // this is the main thing that saves compute
            continue;

        _map->indexToPos(i, pos);
        pcl_pt.x = pos[0];
        pcl_pt.y = pos[1];
        pcl_pt.z = pos[2];
        pcl_pt.intensity = priority;
        global_cloud->push_back(pcl_pt);  

    }

    global_cloud->width = global_cloud->points.size();
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*global_cloud, cloud_msg);
    _map_pub.publish(cloud_msg);

}   
