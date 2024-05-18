#include <active_perception/target_planner.h>

Eigen::Vector4d TargetViewpoint::getColor(float min, float max, Eigen::Matrix<double, 4,4> colormap){
    float rel_gain  = std::max(0.0f, std::min((gain_-min)/(max-min), 1.0f)); // bounds prevent nan values
    int idx = (int)(rel_gain*2.999); // size of colormap is 3
    
    Eigen::Vector4d cmin = colormap.row(idx);
    Eigen::Vector4d cmax = colormap.row(idx+1);
    Eigen::Vector4d color = cmin + (rel_gain*3 - idx)*(cmax-cmin); 
    color(3) = 1;
    return color;
}

void removeSimilarPosesFromList(std::list<std::vector<TargetViewpoint>>& myList) {

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

void sortViewpoints(std::vector<TargetViewpoint>& vpts){
    sort(vpts.begin(), vpts.end(), [](const TargetViewpoint& v1, const TargetViewpoint& v2) { return v1.gain_ > v2.gain_; });   
}


