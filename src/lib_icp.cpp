#include <lib_icp/lib_icp.h>

LibICP::LibICP()
{

}


std::tuple<int, Eigen::MatrixXd> LibICP::icp(Eigen::MatrixXd A, Eigen::MatrixXd B, int max_iterations, double tolerance)
{

    // Check that these point clouds are of the same dimension
    if (A.rows() != B.rows())
    {
        printf("A.rows() = %ld != B.rows() = %lib_icp\n",A.cols(),B.cols());
        Eigen::MatrixXd output;
        return std::tuple<int, Eigen::MatrixXd> {-1,output};
    }

    // Transform to homogenous Matrix
    // https://stackoverflow.com/questions/16280218/convert-a-vector-of-3d-point-to-homogeneous-representation-in-eigen
    Eigen::MatrixXd src = A.colwise().homogeneous();
    Eigen::MatrixXd dst = B.colwise().homogeneous();

    double prev_error = 0;
    int iterations;
    for (iterations = 0; iterations < max_iterations; iterations++)
    {
        auto[distances, indicies] = nearest_neighbor(src.topRows(src.rows()-1),dst.topRows(dst.rows()-1));
        
        std::cout << "Indicies:" << std::endl;
        for (int index : indicies)
        {
            printf("%d\n",index);
        }
        Eigen::MatrixXd dst_mapped(dst.rows(),indicies.size());
        for (int j = 0; j < indicies.size(); j++)
        {
            dst_mapped.col(j) = dst.col(indicies[j]);
        }
        std::cout << "dst_mapped:" << std::endl;
        std::cout << dst_mapped << std::endl;
        Eigen::MatrixXd T = best_fit_transform(src.topRows(src.rows()-1),dst_mapped.topRows(dst_mapped.rows()-1));

        std::cout << "Transformation Matrix:\n" << T << std::endl;

        // Update src
        src = T*src;

        // Check the error
        double sum = 0;
        for (double dist : distances)
        {
            sum += dist;
        }
        double mean_error = sum/distances.size();
        double change_error = prev_error - mean_error;
        std::cout << "mean_error: " << mean_error << std::endl << "change_error: " <<change_error <<std::endl;

        // Check if we are within tolerance
        if (std::abs(change_error) < tolerance)
        {
            break;
        }
        prev_error = mean_error;
    }

    Eigen::MatrixXd T = best_fit_transform(A,src.topRows(src.rows()-1));

    return std::tuple<int, Eigen::MatrixXd> {iterations,T};
}


std::tuple<std::vector<double>, std::vector<int>> LibICP::nearest_neighbor(Eigen::MatrixXd src, Eigen::MatrixXd dst)
{
    // FLANN expects row-major input, but Eigen uses column-major
    Eigen::MatrixXd src_transposed = src.transpose();
    Eigen::MatrixXd dst_transposed = dst.transpose();

    std::cout << "src_transposed:\n" << src_transposed << std::endl;
    std::cout << "dst_transposed:\n" << dst_transposed << std::endl;

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

Eigen::MatrixXd LibICP::best_fit_transform(Eigen::MatrixXd A, Eigen::MatrixXd B)
{
    // Get the centroids
    // https://stackoverflow.com/questions/43196545/eigen-class-to-take-mean-of-each-row-of-matrix-compute-centroid-of-the-column-v
    Eigen::VectorXd centroid_A = A.rowwise().mean();
    Eigen::VectorXd centroid_B = B.rowwise().mean();

    // Translate the points to their centroids
    Eigen::MatrixXd AA = A.colwise() - centroid_A;
    Eigen::MatrixXd BB = B.colwise() - centroid_B;


    std::cout << "A_translated:\n" << AA << std::endl;
    std::cout << "B_translated:\n" << BB << std::endl;

    // Get the rotation matrix from SVD
    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    Eigen::MatrixXd H = AA * BB.transpose(); // Covariance matrix
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    // Ensure a right-handed coordinate system and correct for reflection if necessary
    Eigen::MatrixXd VU = V * U.transpose();
    double det = VU.determinant();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.rows());
    if(det < 0) {
        I(A.rows() - 1, A.rows() - 1) = det; // Adjust the sign in the diagonal matrix for reflection
    }

    // Compute the rotation matrix R
    Eigen::MatrixXd R = V * I * U.transpose();

    std::cout << "Rotation Matrix:\n" << R << std::endl;

    // Get the translation matrix
    Eigen::VectorXd t = centroid_B.transpose() - (R * centroid_A.transpose());

    // Compute the homogeneous transformation matrix
    int dim = R.rows();
    Eigen::MatrixXd T = Eigen::MatrixXd::Identity(dim+1,dim+1);
    T.block(0,0,dim,dim) = R;
    T.block(0,dim,dim,1) = t;

    return T;
}