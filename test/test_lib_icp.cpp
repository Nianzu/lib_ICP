#include <lib_icp/lib_icp.h>
#include <time.h>

int main(int argc, char* argv[])
{
    // Random seed
    // https://stackoverflow.com/questions/20201141/same-random-numbers-generated-every-time-in-c
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    srand((time_t)ts.tv_nsec);

    int num_points = 5;
    Eigen::MatrixXd src = Eigen::MatrixXd::Random(3,num_points);
    
    // Create a rotation matrix
    const double max_theta = 0.2;
    const double min_theta = -0.2;
    // https://www.geeksforgeeks.org/generate-random-double-numbers-in-cpp/#
    double yaw = min_theta + (max_theta - min_theta) * (double)std::rand() / RAND_MAX;
    double pitch = min_theta + (max_theta - min_theta) * (double)std::rand() / RAND_MAX;
    double roll = min_theta + (max_theta - min_theta) * (double)std::rand() / RAND_MAX;
    
    // https://stackoverflow.com/questions/21412169/creating-a-rotation-matrix-with-pitch-yaw-roll-using-eigen
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d rotation_matrix = q.matrix();

    // Create a translation vector
    const double max_translation = 0.5;
    const double min_translation = -0.5;
    double x = min_translation + (max_translation - min_translation) * (double)std::rand() / RAND_MAX;
    double y = min_translation + (max_translation - min_translation) * (double)std::rand() / RAND_MAX;
    double z = min_translation + (max_translation - min_translation) * (double)std::rand() / RAND_MAX;
    Eigen::Vector3d translation_vector(x,y,z);

    // Create B as a rotation and translation of A
    Eigen::MatrixXd dst = (rotation_matrix * src).colwise() + translation_vector;

    std::cout << "A:\n" << src << std::endl;
    std::cout << "B:\n" << dst << std::endl;


    LibICP icp;
    auto [iterations, transformation] = icp.icp(src,dst,20,0.001);
    printf("ICP iterations: %d\n",iterations);
    printf("Transformation Matrix:\n");
    std::cout << transformation << std::endl;
}