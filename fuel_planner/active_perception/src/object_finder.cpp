#include <active_perception/object_finder.h>
#include <common/utils.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/filters/extract_indices.h>
#include <string>

typedef std::pair<Vector3d, Vector3d> BoundingBox;

bool checkOverlap(const BoundingBox& box1, const BoundingBox& box2) {
    for (int i = 0; i < 3; ++i) {
        if (box1.second[i] < box2.first[i] || box1.first[i] > box2.second[i])
            return false; // No overlap along this axis
    }
    return true; // Overlap along all axes
}

double intersectionVolume(const BoundingBox& box1, const BoundingBox& box2) {
    Vector3d minCorner = box1.first.cwiseMax(box2.first);
    Vector3d maxCorner = box1.second.cwiseMin(box2.second);
    Vector3d dims = (maxCorner - minCorner).cwiseAbs();
    return dims.prod();
}


bool enforceIntensitySimilarity (const pcl::PointXYZI& point_a, const pcl::PointXYZI& point_b, float squared_distance)
{
  // close enough + discrete enough + similar enough
  
  if (squared_distance > 1 || std::abs (point_a.intensity - point_b.intensity) > 1e-1f)
    return (false);
  return (true);
}

ObjectFinder::ObjectFinder(ros::NodeHandle& nh){
    nh.param("/object_finder/global_iou_thresh", _global_iou_thresh, 0.5f); 
    nh.param("/object_finder/local_iou_thresh", _local_iou_thresh, 0.5f); 

    _obj_update_timer = nh.createTimer(ros::Duration(0.05), &ObjectFinder::objectFusionTimer, this);

}

void ObjectFinder::objectFusionTimer(const ros::TimerEvent& e){

    pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    Eigen::Vector3i min_cut = _sdf_map->md_->local_bound_min_;
    Eigen::Vector3i max_cut = _sdf_map->md_->local_bound_max_;

    _sdf_map->boundIndex(min_cut);
    _sdf_map->boundIndex(max_cut);
    
    Eigen::Vector3d pos;
    pcl::PointXYZI pcl_pt;
    
    for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y)
            for (int z = _sdf_map->mp_->box_min_(2); z < _sdf_map->mp_->box_max_(2); ++z) {

                int adr = _sdf_map->toAddress(x,y,z);
                float priority = _prio_map->priority_buffer[adr];

                if (priority < _att_min || !(_sdf_map->getOccupancy(adr) == fast_planner::SDFMap::OCCUPIED)) {
                    _prio_map->priority_buffer[adr] = 0; // cleanup
                    continue;
                }
                
                _sdf_map->indexToPos(adr, pos);
                pcl_pt.x = pos[0];
                pcl_pt.y = pos[1];
                pcl_pt.z = pos[2];
                pcl_pt.intensity = priority;
                
                local_cloud->push_back(pcl_pt);
            }

    std::list<Object> local_objects;    

    createObjects(local_cloud, local_objects); // cluster the global point cloud 
    mergeObjects(local_objects); // Fuse newly found objects into existing ones

}


void ObjectFinder::createObjects(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::list<Object>& objects){    
    // Apply clustering
    std::vector<pcl::PointIndices> cluster_indices;

    pcl::ConditionalEuclideanClustering<pcl::PointXYZI> cec (true);
    cec.setInputCloud (cloud);
    cec.setConditionFunction (enforceIntensitySimilarity);
    cec.setClusterTolerance (0.15);
    cec.setMinClusterSize (5);
    cec.setMaxClusterSize (100);
    cec.segment (cluster_indices);

    int i = 0;
    for (const auto& cluster_index : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        pcl::PointIndices::Ptr pcl_indices(new pcl::PointIndices(cluster_index));
        extract.setInputCloud(cloud);
        extract.setIndices(pcl_indices);
        extract.filter(*cluster_cloud);
        
        // create objects from clustered clouds
        Object object(*cluster_cloud);
        object.id = ++i;
        objects.push_back(object);
    }
}

// fuse newly detected objects into existing ones
void ObjectFinder::mergeObjects(std::list<Object> new_objects){
    // merge new objects into old ones
    for (Object obj_n: new_objects){
        bool merged = false;
        // find something to merge into
        for (auto& obj_o: global_objects){
            if (areObjectsSimilar(obj_n, obj_o, _local_iou_thresh)){
                obj_o.merge(obj_n);// 
                merged = true;
                break;
                
            }
        }
        if (!merged){   
            global_objects.push_back(obj_n);
        }
    }

    // merge old objects into old ones
    std::list<std::list<Object>::iterator> remove_iterators;
    for (auto it1 = global_objects.begin(); it1!=global_objects.end();it1++){
        for (auto it2 = std::next(it1); it2!=global_objects.end();it2++){
            if (areObjectsSimilar(*it1, *it2, _global_iou_thresh)){
                it1->merge(*it2);
                remove_iterators.push_back(it2);
                break;   
            }
        }    
    }
    for (auto it: remove_iterators)
        global_objects.erase(it);

}


bool ObjectFinder::areObjectsSimilar(const Object& obj_a, const Object& obj_b, double overlapThreshold){
    // if ((obj_a.bbox_min_-obj_b.bbox_min_).norm() < 0.1 && (obj_a.bbox_max_-obj_b.bbox_max_).norm() < 0.1)
    //     return true;

    BoundingBox box1(obj_a.bbox_min_, obj_a.bbox_max_);
    BoundingBox box2(obj_b.bbox_min_, obj_b.bbox_max_);

    // if overlap check IOU
    if (checkOverlap(box1, box2)){
        double intersectionVol = intersectionVolume(box1, box2);
        double totalVol1 = (box1.second - box1.first).prod();
        double totalVol2 = (box2.second - box2.first).prod();
        double overlapRatio1 = intersectionVol / totalVol1;
        double overlapRatio2 = intersectionVol / totalVol2;

        // Check if overlap ratio exceeds the threshold for either bounding box
        if (overlapRatio1 > overlapThreshold || overlapRatio2 > overlapThreshold)
            return true;
    }
    // 
    // if (checkAdjacent(box1, box2)){
    //     return true;
    // }
        
    return false;
}

// void ObjectFinder::publishObjectBoxes(){
//     int i=0;
//     for (Object& object: global_objects ){
//         _viz.drawBox(object.centroid_, object.scale_, Eigen::Vector4d(0.5, 0, 1, 0.3), "box"+std::to_string(i), i, 7);
//         i++;
//     }

// }
