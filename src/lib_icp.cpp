#include <lib_icp/lib_icp.h>

std::tuple<int, Eigen::MatrixXd> LibICP::icp(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, int max_iterations, double tolerance)
{

    // Check that these point clouds are of the same dimension
    if (src.rows() != dst.rows())
    {
        Eigen::MatrixXd output;
        return std::tuple<int, Eigen::MatrixXd> {-1,output};
    }

    // Transform to homogenous Matrix
    // https://stackoverflow.com/questions/16280218/convert-a-vector-of-3d-point-to-homogeneous-representation-in-eigen
    Eigen::MatrixXd src_H = src.colwise().homogeneous();
    Eigen::MatrixXd dst_H = dst.colwise().homogeneous();

    double prev_error = 0;
    int iterations;
    for (iterations = 0; iterations < max_iterations; iterations++)
    {
        auto[distances, indicies] = nearest_neighbor(src_H.topRows(src_H.rows()-1),dst_H.topRows(dst_H.rows()-1));
        
        Eigen::MatrixXd dst_mapped(dst_H.rows(),indicies.size());
        for (int j = 0; j < indicies.size(); j++)
        {
            dst_mapped.col(j) = dst_H.col(indicies[j]);
        }
        Eigen::MatrixXd T = best_fit_transform(src_H.topRows(src_H.rows()-1),dst_mapped.topRows(dst_mapped.rows()-1));

        // Update src
        src_H = T*src_H;

        // Check the error
        double sum = 0;
        for (double dist : distances)
        {
            sum += dist;
        }
        double mean_error = sum/distances.size();
        double change_error = prev_error - mean_error;

        // Check if we are within tolerance
        if (std::abs(change_error) < tolerance)
        {
            break;
        }
        prev_error = mean_error;
    }

    Eigen::MatrixXd T = best_fit_transform(src,src_H.topRows(src_H.rows()-1));

    return std::tuple<int, Eigen::MatrixXd> {iterations,T};
}


std::tuple<std::vector<double>, std::vector<int>> LibICP::nearest_neighbor(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst)
{
    // FLANN expects row-major input, but Eigen uses column-major
    Eigen::MatrixXd src_transposed = src.transpose();
    Eigen::MatrixXd dst_transposed = dst.transpose();

    // FLANN also uses its own matrix type
    flann::Matrix<double> dataset(dst_transposed.data(),dst_transposed.rows(),dst_transposed.cols());
    flann::Matrix<double> query(src_transposed.data(),src_transposed.rows(),src_transposed.cols());
    
    // Create FLANN index, using L2 norm and KD Trees
    flann::Index<flann::L2<double>> index(dataset, flann::KDTreeIndexParams(1));
    index.buildIndex();
    
    // Prepare for calculation
    std::vector<int> indices_vec(query.rows);
    std::vector<double> dists_vec(query.rows);
    flann::Matrix<int> indices(indices_vec.data(), query.rows, 1);
    flann::Matrix<double> dists(dists_vec.data(), query.rows, 1);

    // Run FLANN
    index.knnSearch(query, indices, dists, 1, flann::SearchParams(128));

    return std::tuple<std::vector<double>, std::vector<int>> {dists_vec, indices_vec};
}

Eigen::MatrixXd LibICP::best_fit_transform(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst)
{
    // Get the centroids
    // https://stackoverflow.com/questions/43196545/eigen-class-to-take-mean-of-each-row-of-matrix-compute-centroid-of-the-column-v
    Eigen::VectorXd centroid_src = src.rowwise().mean();
    Eigen::VectorXd centroid_dst = dst.rowwise().mean();

    // Translate the points to their centroids
    Eigen::MatrixXd src_translated = src.colwise() - centroid_src;
    Eigen::MatrixXd dst_translated = dst.colwise() - centroid_dst;

    // Get the rotation matrix from SVD
    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    Eigen::MatrixXd H = src_translated * dst_translated.transpose(); // Covariance matrix
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    // Ensure a right-handed coordinate system and correct for reflection if necessary
    Eigen::MatrixXd VU = V * U.transpose();
    double det = VU.determinant();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(src.rows(), src.rows());
    if(det < 0) {
        I(src.rows() - 1, src.rows() - 1) = det; // Adjust the sign in the diagonal matrix for reflection
    }

    // Compute the rotation matrix R
    Eigen::MatrixXd R = V * I * U.transpose();

    // Get the translation matrix
    Eigen::VectorXd t = centroid_dst.transpose() - (R * centroid_src.transpose());

    // Compute the homogeneous transformation matrix
    int dim = R.rows();
    Eigen::MatrixXd T = Eigen::MatrixXd::Identity(dim+1,dim+1);
    T.block(0,0,dim,dim) = R;
    T.block(0,dim,dim,1) = t;

    return T;
}