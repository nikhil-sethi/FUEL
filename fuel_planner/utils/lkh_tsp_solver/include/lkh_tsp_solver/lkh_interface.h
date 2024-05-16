#ifndef _LKH_INTERFACE_H
#define _LKH_INTERFACE_H
#include <vector>
#include <string>
#include <eigen3/Eigen/Eigen>

extern "C" {
#include "LKH.h"
#include "Genetic.h"
}

namespace lkh_interface{
int solveTSPLKH(const char* input_file);
void readTourFromFile(std::vector<int>& indices, const std::string& file_dir);
void writeCostMatToFile(const Eigen::MatrixXd& cost_mat, const std::string& file_dir);
}
#endif