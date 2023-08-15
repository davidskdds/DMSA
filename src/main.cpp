#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "dmsa.h"
#include "IO_functions.h"
#include "config.h"
#include <thread>

using namespace Eigen;
using namespace pcl;

std::mutex viewer_mutex;


void updateWindow(pcl::visualization::PCLVisualizer* viewer_ptr)
{
    while(true)
    {
        viewer_mutex.lock();
        viewer_ptr->spinOnce(1,true);
        viewer_mutex.unlock();
        usleep(10000);
    }

}


int main(int argc, char** argv) {

    std::string directory;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory-path>" << std::endl;
        //return -1;
        directory = "/home/david/Software/catkin_ws/src/DMSA/data";

    }
    else
    {
        directory = argv[1];
    }

    // init DMSA map and config
    MapManagement Map;
    config dmsa_conf;
    int numKeyframesAdded = 0;

    // load poses in tum format
    Eigen::MatrixXd Poses_tum = readPosesFromFile(directory);

    // load point clouds
    std::vector<pcl::PointXYZ> point_clouds;

    std::vector<std::string> files = listFilesInDirectory(directory);

    for (const std::string& file : files) {
        if (file.length() > 4 && file.substr(file.length() - 4) == ".pcd") {
            std::string fullFilePath = directory + "/" + file;

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

            if (pcl::io::loadPCDFile<pcl::PointXYZ>(fullFilePath, *cloud) == -1) {
                std::cerr << "Couldn't read file: " << fullFilePath << std::endl;
                continue;
            }

            std::cout << "Loaded " << cloud->width * cloud->height << " data points from " << fullFilePath << std::endl;

            // downsample point cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr pc_filtered (new pcl::PointCloud<pcl::PointXYZ>);

            gridRandomDownsampling(*cloud, *pc_filtered, dmsa_conf.grid_size_downsampling);
            std::cout << "Downsampled point cloud to " << pc_filtered->width * pc_filtered->height << " points" << std::endl;

            // convert pcl point cloud to eigen matrix
            Matrix3Xd pc_filtered_eig = convertPclToEigen(pc_filtered);

            if (numKeyframesAdded > Poses_tum.rows()-1)
            {
                std::cerr << "Number of keyframes in your poses file is smaller than the number of point clouds. Continue with first "<< numKeyframesAdded<<" clouds"<< std::endl;
                break;
            }

            // get position and axang w. r. t. world frame from poses in tum format
            Vector3d position_w = Poses_tum.block(numKeyframesAdded,1,0,3).transpose();
            
            // convert quaternion to axangle
            Quaterniond q;

            q.x() = Poses_tum(numKeyframesAdded,4);
            q.y() = Poses_tum(numKeyframesAdded,5);
            q.z() = Poses_tum(numKeyframesAdded,6);
            q.w() = Poses_tum(numKeyframesAdded,7);

            Vector3d axang_w = rotm2axang(q.normalized().toRotationMatrix());

            // add pc to dmsa map
            Map.addKeyframe(position_w,axang_w,pc_filtered_eig,Poses_tum(numKeyframesAdded,0));
            

            ++numKeyframesAdded;

            std::cout << "Added keyframe no. " << numKeyframesAdded << " points" << std::endl;
        }
    }

    // init viewer
    pcl::visualization::PCLVisualizer *viewer;

    if (dmsa_conf.live_view)
    {
        std::cout << "Start point cloud viewer " << std::endl;

        // init viewer on seperate thread
        viewer = new pcl::visualization::PCLVisualizer ("DMSA Point Cloud Viewer");

        std::thread t1(updateWindow, viewer);
        t1.detach();
    }

    // optimization loop
    double score = 0;

    dmsa DMSA_cloud_optimzer(dmsa_conf.grid_size_dmsa);

    std::cout << "Start DMSA optimization with "<< Map.numPoints << " points . . . "<<std::endl;

    for (int iter = 1; iter < dmsa_conf.num_iter; ++iter)
    {
        // visualize global cloud
        if (dmsa_conf.live_view)
        {
            viewer_mutex.lock();
            visualizePc(*viewer,Map.trajParamsGlo.Translations ,Map.globalCloud.XYZ,Vector3f(0.,0.,0.));
            viewer_mutex.unlock();
        }

        std::cout << "Optimization step "<< iter << std::endl;

        // optimize one iteration
        score = DMSA_cloud_optimzer.optimizeMap(&Map,1,0.4,true,false,true,0.97);

        // update cloud
        Map.updateGlobalCloud();

    }

    return 0;
}
