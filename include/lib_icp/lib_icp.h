#ifndef LIBICP_H
#define LIBICP_H

#include <Eigen/Dense>
#include <tuple>

class LibICP 
{
    public:

        // Constructor
        LibICP(); 
        Eigen::MatrixXd icp(Eigen::MatrixXd A, Eigen::MatrixXd B, int max_iterations, double tolerance);
        
    private:
        std::tuple<double, int> nearest_neighbor(Eigen::MatrixXd src, Eigen::MatrixXd dst);
        Eigen::MatrixXd best_fit_transform(Eigen::MatrixXd A, Eigen::MatrixXd B);
};

#endif