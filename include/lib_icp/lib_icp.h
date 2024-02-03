#ifndef LIBICP_H
#define LIBICP_H

#include <Eigen/Dense>
#include <tuple>
#include <flann/flann.hpp>

class LibICP 
{
    public:

        // Constructor
        LibICP(); 
        std::tuple<int, Eigen::MatrixXd> icp(Eigen::MatrixXd A, Eigen::MatrixXd B, int max_iterations, double tolerance);
        
    private:
        std::tuple<std::vector<double>, std::vector<int>> nearest_neighbor(Eigen::MatrixXd src, Eigen::MatrixXd dst);
        Eigen::MatrixXd best_fit_transform(Eigen::MatrixXd A, Eigen::MatrixXd B);
};

#endif