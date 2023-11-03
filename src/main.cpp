#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "dmsa.h"
#include "IO_functions.h"
#include "config.h"
#include <thread>
#include <cstdlib>
#include <ctime>

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

    std::cout << "Load point clouds " << std::endl;

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
            //Vector3d position_w = Poses_tum.block(numKeyframesAdded,1,0,3).transpose();

            Vector3d position_w;

            position_w(0) = Poses_tum(numKeyframesAdded,1);
            position_w(1) = Poses_tum(numKeyframesAdded,2);
            position_w(2) = Poses_tum(numKeyframesAdded,3);

            
            // convert quaternion to axangle
            Quaterniond q;

            q.x() = Poses_tum(numKeyframesAdded,4);
            q.y() = Poses_tum(numKeyframesAdded,5);
            q.z() = Poses_tum(numKeyframesAdded,6);
            q.w() = Poses_tum(numKeyframesAdded,7);

            Vector3d axang_w = rotm2axang(q.normalized().toRotationMatrix());

            // add pc to dmsa map
            if (numKeyframesAdded%3 == 0) Map.addKeyframe(position_w,axang_w,pc_filtered_eig,Poses_tum(numKeyframesAdded,0));


            // update cloud
            Map.updateGlobalCloud();

            // visualize global cloud
            if (dmsa_conf.live_view)
            {
                viewer_mutex.lock();
                visualizePc(*viewer,Map.trajParamsGlo.Translations ,Map.globalCloud.XYZ,Vector3f(0.,0.,0.));
                viewer_mutex.unlock();
            }
            
            ++numKeyframesAdded;

            std::cout << "Added keyframe no. " << numKeyframesAdded << " points" << std::endl;
        }
    }

    savePosesToTxt(Map.trajParamsGlo.Translations, Map.trajParamsGlo.Orientations, Map.getKeyframeStamps(), directory,"Original_poses");

    // add some noise
    Map.global2relative();

    for(int k = 1; k < Map.getNumKeyframes(); ++k)
    {
        // get random values
        double sigma_p = 0.1;
        double sigma_r = 0.008;

        srand (static_cast <unsigned> (time(0)));
        double r1 = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX));
        srand (static_cast <unsigned> (time(0)));
        double r2 = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX));
        srand (static_cast <unsigned> (time(0)));
        double r3 = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX));

        Map.trajParamsRel.Translations.col(k) += sigma_p*2.*Eigen::Vector3d(r1-0.5,r2-0.5,r3-0.5);

        Map.trajParamsRel.Orientations.col(k) = rotm2axang( axang2rotm(sigma_r*2.*Eigen::Vector3d(r1-0.5,r2-0.5,r3-0.5))*axang2rotm(Map.trajParamsRel.Orientations.col(k)));

    }

    Map.relative2global();

    std::cout << "Added noise to keyframes " << std::endl;

    // update cloud
    Map.updateGlobalCloud();

    // visualize global cloud
    if (dmsa_conf.live_view)
    {
        viewer_mutex.lock();
        visualizePc(*viewer,Map.trajParamsGlo.Translations ,Map.globalCloud.XYZ,Vector3f(0.,0.,0.));
        viewer_mutex.unlock();
    }

    // save poses with additional noise
    savePosesToTxt(Map.trajParamsGlo.Translations, Map.trajParamsGlo.Orientations, Map.getKeyframeStamps(), directory,"noisy_poses");



    // optimization loop
    double score = 0;
    double alpha = 0.5;

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
        score = DMSA_cloud_optimzer.optimizeMap2(&Map,1,alpha,true,false,true,0.97,true);

        // update cloud
        Map.updateGlobalCloud();

        // save new poses
        savePosesToTxt(Map.trajParamsGlo.Translations, Map.trajParamsGlo.Orientations, Map.getKeyframeStamps(), directory,"Optimized_poses");

    }

    return 0;
}
