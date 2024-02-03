#include <lib_icp/lib_icp.h>

int main(int argc, char* argv[])
{
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;


    LibICP icp;
    icp.icp(A,B,20,0.001);
}