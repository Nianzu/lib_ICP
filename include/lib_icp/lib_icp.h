#ifndef LIBICP_H
#define LIBICP_H

#include <Eigen/Dense>
#include <tuple>
#include <flann/flann.hpp>

class LibICP 
{
    public:

        std::tuple<int, Eigen::MatrixXd> icp(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, int max_iterations, double tolerance);
        
    private:
        std::tuple<std::vector<double>, std::vector<int>> nearest_neighbor(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);
        Eigen::MatrixXd best_fit_transform(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);
};

#endif