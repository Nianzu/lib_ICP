#include <lib_icp/lib_icp.h>

LibICP::LibICP()
{

}


Eigen::MatrixXd LibICP::icp(Eigen::MatrixXd A, Eigen::MatrixXd B, int max_iterations, double tolerance)
{
    printf("Working!");
    Eigen::MatrixXd output;
    return output;
}


std::tuple<double, int> LibICP::nearest_neighbor(Eigen::MatrixXd src, Eigen::MatrixXd dst)
{
    std::tuple<double, int> output;
    return output;
}

Eigen::MatrixXd LibICP::best_fit_transform(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
    Eigen::MatrixXd output;
    return output;
}