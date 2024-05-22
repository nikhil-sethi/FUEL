#include <active_perception/target_planner.h>
#include <chrono>
#include <common/io.h>

void TargetPlanner::init(ros::NodeHandle& nh){    
    // Params
    nh.param("/target_planner/rmin", _rmin, 0.3f); 
    nh.param("/target_planner/min_vpt_clearance", _min_vpt_clearance, 0.3f); 
    nh.param("/priority_map/pmin", _att_min, 1.0f); 
    nh.param("/target_planner/min_info_gain", _min_info_gain, 1.0f); 

    // inits
    vpts_msg.viewpoints.header.stamp = ros::Time::now();
    vpts_msg.viewpoints.header.frame_id = "map";

    _raycaster.reset(new RayCaster);
    _raycaster->setParams(_sdf_map->mp_->resolution_, _sdf_map->mp_->map_origin_);
    _percep_utils.reset(new fast_planner::PerceptionUtils(nh));
    _camera.reset(new Camera(nh));
    // viz = fast_planner::PlanningVisualization(nh);

    vpt_pub = nh.advertise<common_msgs::Viewpoints>("/objects/target_vpts", 1);
    info_timer = nh.createTimer(ros::Duration(0.05), &TargetPlanner::informationGainTimer, this);
    
}

void TargetPlanner::informationGainTimer(const ros::TimerEvent& event){
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    Eigen::Vector3d local_box_min, local_box_max;
    _sdf_map->indexToPos(_sdf_map->md_->local_bound_min_, local_box_min);
    _sdf_map->indexToPos(_sdf_map->md_->local_bound_max_, local_box_max);

    // Sample viewpoints around all objects
    all_viewpoints.clear();
    for (Object& object: _obj_fnd->global_objects){   
        if (!object.isInBox(local_box_min, local_box_max)) continue;
          
        std::vector<TargetViewpoint> sample_vpts; 
        sampleViewpoints(object, sample_vpts);
        all_viewpoints.push_back(sample_vpts);
    }
    
    filterSimilarPoses(all_viewpoints);
    
    // Calculate priority of each viewpoint 
    std::list<std::vector<TargetViewpoint>>::iterator iter = all_viewpoints.begin();
    for (Object& object: _obj_fnd->global_objects){      
        if (!object.isInBox(local_box_min, local_box_max)) continue;

        // calculate information gain
        findTopViewpoints(object, *iter);
        iter++;
    }

    publishTargetViewpoints();

    std::chrono::duration<double> dt_vpt = std::chrono::high_resolution_clock::now() - start;
    // ROS_WARN("Time vpts: %f", dt_vpt);

}


void TargetPlanner::sortViewpoints(std::vector<TargetViewpoint>& vpts){
    sort(vpts.begin(), vpts.end(), [](const TargetViewpoint& v1, const TargetViewpoint& v2) { return v1.gain_ > v2.gain_; });   
}


/*
Sample valid viewpoints around the object
A viewpoint is valid if:
1. exists in map
2. Not too close to unknown or occupied space
3. Has the object in view
*/
void TargetPlanner::sampleViewpoints(Object& object, std::vector<TargetViewpoint>& sampled_vpts){


    // find the minimum radius for cylindrical sampling
        // find the smallest face of the bbox
        // find the AR of the smallest face
        // if this AR is larger than image ar
            // bound the face in image using the width of smallest face
        // else
            // bound using the height
        // find the rmin using the bound
    // object.viewpoint_candidates.clear();

    Eigen::Vector3d diag_3d = object.bbox_max_ - object.bbox_min_;
    int shortest_axis = diag_3d(1) > diag_3d(0) ? 0: 1;
    Eigen::Vector2d diag_2d = {diag_3d(shortest_axis), diag_3d(2)};
    
    // this distance might be obsolete now that we have isometric views. but still nice starting point
    double rmin = _camera->getMinDistance(diag_2d); 

    rmin += diag_3d(1-shortest_axis)/2; // add the longer axis to rmin because the cylinder starts at the centroid
    
    for (double rc = rmin, dr = 0.4; rc <= rmin + 0.5 + 1e-3; rc += dr)
        for (double phi = -M_PI; phi < M_PI-0.5235; phi += 0.5235) {
            Eigen::Vector3d sample_pos = object.centroid_ + rc * Eigen::Vector3d(cos(phi), sin(phi), 0);
            sample_pos[2] = sample_pos[2] + 0.1; // add a height to view the object isometrically. this will depend on the data from the sensor model

            if (!_sdf_map->isInBox(sample_pos) || _sdf_map->getInflateOccupancy(sample_pos) == 1 || _sdf_map->isNearUnknown(sample_pos, _min_vpt_clearance))
                continue;

            // === Check if object is in view            
            if (!isObjectInView(object, sample_pos, Eigen::Vector3d(0,0,phi + M_PI))){
                // print_eigen(sample_pos);
                continue;
            }

            TargetViewpoint vpt(TargetViewpoint(sample_pos, phi + M_PI, 0)); // 0 gain. will be computed later
            sampled_vpts.push_back(vpt);
        }
}


void TargetPlanner::findTopViewpoints(Object& object, std::vector<TargetViewpoint>& sampled_vpts){
    // find the closest distance that would have the object still in view
    // sample viewpoints around the object starting from the minimum distance
    // for each sample 
        // if sample is near unknown region (OR) on an occupied cell (OR) not in the volume --> discard
        // if the object's perspective view from the sample doesnt fit in the _camera --> discard 
        // evaluate information gain by ray casting (need attention buffer for this)
        // if gain is more than min gain, add viewpoint to candidate list
    // sort viewpoints by their gain
    // get some top viewpoints for the object

    object.viewpoints.clear();
    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    for (auto it=sampled_vpts.begin(); it!=sampled_vpts.end();++it){
            float gain = computeInformationGain(object, it->pos_, it->yaw_);
            if (gain<=_min_info_gain) continue;
            
            it->gain_ = gain;
            object.viewpoints.push_back(*it);
        }
        
    if (!object.viewpoints.empty())
        sortViewpoints(object.viewpoints);       // sort list wrt information gain 

    std::chrono::duration<double> dt_vpt = std::chrono::high_resolution_clock::now() - start;
    // ROS_INFO("Time vpt: %f", dt_vpt);

}       

float TargetPlanner::computeInformationGain(Object& object, const Eigen::Vector3d& sample_pos, const double& yaw){
    
    Eigen::Vector3i idx;
    Eigen::Vector3i start_idx;
    Eigen::Vector3d pos;
    float total_gain = 0;
    _percep_utils->setPose(sample_pos, yaw);

    Eigen::Vector3i bbox_min_idx;
    Eigen::Vector3i bbox_max_idx;

    //inflated bounding box around object bbox
    Eigen::Vector3d bbox_min = object.bbox_min_ - Eigen::Vector3d(1,1,0.5);
    Eigen::Vector3d bbox_max = object.bbox_max_ + Eigen::Vector3d(1,1,0.5);
    _sdf_map->boundBox(bbox_min, bbox_max);

    _sdf_map->posToIndex(bbox_min, bbox_min_idx);
    _sdf_map->posToIndex(bbox_max, bbox_max_idx);

    for (int x = bbox_min_idx(0); x <= bbox_max_idx(0); ++x)
        for (int y = bbox_min_idx(1); y <= bbox_max_idx(1); ++y)
            for (int z = _sdf_map->mp_->box_min_(2); z < _sdf_map->mp_->box_max_(2); ++z) {
        int adr = _sdf_map->toAddress(x,y,z);
        _sdf_map->indexToPos(Eigen::Vector3i(x,y,z), pos);

        // cell should be a frontier and in view
        if (_ftr_fndr->frontier_flag_[adr] == 0  || !_percep_utils->insideFOV(pos)) continue;
        
        _raycaster->input(pos, sample_pos);
        bool visib = true;
        _raycaster->nextId(idx); // because we're already on the surface, presumably
        // start_idx = idx;
        while (_raycaster->nextId(idx)) {
            int ray_adr = _sdf_map->toAddress(idx);
            if (
                // _ftr_fndr->frontier_flag_[ray_adr] == 1
                _diff_map->diffusion_buffer[ray_adr] > _att_min // the ray shouldnt have any other attentive cell in it's path
                // _sdf_map->getInflateOccupancy(idx) == 1   // not using inflation for now becauase most attentive cells will be missed out then
                || _sdf_map->getOccupancy(ray_adr) == fast_planner::SDFMap::OCCUPIED 
                || _sdf_map->getOccupancy(ray_adr) == fast_planner::SDFMap::UNKNOWN
                ) {

                visib = false;
                break;
            }
        }
        // if (visib) total_gain += infoTransfer(point.intensity)*point.intensity;
        if (visib) total_gain += _diff_map->diffusion_buffer[adr];
    }
    return total_gain; 
}

// returns true if a full 6D viewpoint can completely view set of points
bool TargetPlanner::isObjectInView(const Object& object, const Eigen::Vector3d& pos, const Eigen::Vector3d& orient){
    
    // Exstrinsic transformation (world --> viewpoint --> _camera)

    // World to viewpoint
    Eigen::Isometry3d T_world_sample; // world with respect to sample
    T_world_sample.translation() = pos;
      
    Eigen::Quaterniond quat;
    quat = Eigen::AngleAxisd(orient(0), Vector3d::UnitX())
        * Eigen::AngleAxisd(orient(1), Vector3d::UnitY())
        * Eigen::AngleAxisd(orient(2), Vector3d::UnitZ());
    
    T_world_sample.linear() = quat.toRotationMatrix();

    // Viewpoint to _camera
    Eigen::Isometry3d T_world_cam = T_world_sample*_camera->T_odom_cam; // this transform takes a point in world frame to _camera frame
    
    // Transform object corners to _camera frame
    std::vector<Eigen::Vector3d> corners_cam(8, Eigen::Vector3d::Zero());
    _camera->transform(object.projection_corners, corners_cam, T_world_cam.inverse()); 

    // project corners to image frame and check bounds
    return _camera->arePtsInView(corners_cam);
}

// nonlinear transfer function that modifies the gain that you see based on it's value
float infoTransfer(float gain){
    int alpha = 4; // higher if you want more local object region importance over object coverage. But small low attention objects might get missed out then
    return 1 + pow(alpha*(gain-0.5), 3);
}

void TargetPlanner::filterSimilarPoses(std::list<std::vector<TargetViewpoint>>& myList) {

     for (auto it = myList.begin(); it != myList.end(); ++it) {
        auto it2 = std::next(it);
        for (; it2 != myList.end(); ++it2) {
            for (auto it3 = it->begin(); it3!=it->end(); it3++) {
                for (auto it4 = it2->begin(); it4!=it2->end(); it4++) {
                    if (it3->isClose(*it4)) {
                        it2->erase(it4); // Remove the vector with similar pose
                        --it4; // Adjust the iterator after erasing
                        break; // No need to check other poses in vec2
                    }
                }
            }
        }
    }

}

void TargetPlanner::publishTargetViewpoints(){
    vpts_msg.viewpoints.poses.clear();
    vpts_msg.priorities.clear();
    for (Object& object: _obj_fnd->global_objects){    
        // Get first 3 three viewpoints for publishing
        for (int i=0; i<std::min((int)object.viewpoints.size(), 3); i++){
            vpts_msg.viewpoints.poses.push_back(object.viewpoints[i].toGeometryMsg());  
            vpts_msg.priorities.push_back(object.viewpoints[i].gain_);
        }  
    }
    vpt_pub.publish(vpts_msg); // publish poses for use by lkh tsp inside fuel

}