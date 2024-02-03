#include <lib_icp/lib_icp.h>

LibICP::LibICP()
{

}


std::tuple<int, Eigen::MatrixXd> LibICP::icp(Eigen::MatrixXd A, Eigen::MatrixXd B, int max_iterations, double tolerance)
{

    // Check that these point clouds are good
    if (A.cols() != B.cols())
    {
        Eigen::MatrixXd output;
        return std::tuple<int, Eigen::MatrixXd> {-1,output};
    }

    // Transform to homogenous Matrix
    // https://stackoverflow.com/questions/16280218/convert-a-vector-of-3d-point-to-homogeneous-representation-in-eigen
    Eigen::MatrixXd src = A.colwise().homogeneous();
    Eigen::MatrixXd dst = B.colwise().homogeneous();

    double prev_error = 0;

    for (int i = 0; i < max_iterations; i++)
    {
        auto[distances, indicies] = nearest_neighbor(src,dst);
        Eigen::MatrixXd dst_mapped(dst.rows(),indicies.size());
        for (int index : indicies)
        {
            dst_mapped.col(i) = dst.col(index);
        }
        Eigen::MatrixXd T = best_fit_transform(src,dst_mapped);

        // Update src
        src = T*src;

        // Check the error
        double&& sum = 0;
        for (double dist : distances)
        {
            sum += dist;
        }
        double mean_error = sum/distances.size();
        double change_error = prev_error - mean_error;

        // Easier than dealing with Abs
        if (change_error < tolerance && change_error > -tolerance)
        {
            break;
        }
        prev_error = mean_error;
    }

    Eigen::MatrixXd T = best_fit_transform(A,src);

    return std::tuple<int, Eigen::MatrixXd> {0,T};
}


std::tuple<std::vector<double>, std::vector<int>> LibICP::nearest_neighbor(Eigen::MatrixXd src, Eigen::MatrixXd dst)
{
    std::tuple<std::vector<double>, std::vector<int>> output;
    return output;
}

Eigen::MatrixXd LibICP::best_fit_transform(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
    Eigen::MatrixXd output;
    return output;
}