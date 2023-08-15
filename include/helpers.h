
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdio>

#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <algorithm>
#include <pcl/pcl_base.h>
#include <math.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <string>

#include <iostream>

#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/io/pcd_io.h>

using namespace Eigen;
using namespace pcl;

//#include <Eigen/Dense>
//using VectorXd;

inline void savePointCloudAsPCD(const Matrix3Xd& points, const std::string& filename)
{
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);

    // Fill in the cloud data
    cloud->width    = points.cols();
    cloud->height   = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].x = points(0, i);
        cloud->points[i].y = points(1, i);
        cloud->points[i].z = points(2, i);
    }

    // Save the cloud to a .pcd file
    if (io::savePCDFileASCII(filename, *cloud) == -1)
    {
        PCL_ERROR("Failed to save PCD file\n");
        return;
    }

    std::cout << "Saved " << cloud->points.size () << " data points to " << filename << std::endl;
}

inline Matrix3Xd convertPclToEigen(PointCloud<PointXYZ>::Ptr pcl_cloud)
{

    Matrix3Xd EigenCloud;

    EigenCloud.resize(3,pcl_cloud->size() );

    for (int i = 0; i < EigenCloud.cols(); i++) {


        EigenCloud(0, i) = pcl_cloud->points[i].x;
        EigenCloud(1, i) = pcl_cloud->points[i].y;
        EigenCloud(2, i) = pcl_cloud->points[i].z;
    }

    return EigenCloud;
}



inline void convertEigenToPointCloud(const Matrix3Xd& matrix,PointCloud<PointXYZ>::Ptr cloud)
{

    if(cloud->size() != matrix.cols())cloud->points.resize(matrix.cols());

    for (int i = 0; i < matrix.cols(); i++) {
        cloud->points[i].x = matrix(0, i);
        cloud->points[i].y = matrix(1, i);
        cloud->points[i].z = matrix(2, i);
    }

    cloud->width = matrix.cols();
    cloud->height = 1;
}

// Function to compute Gaussian kernel
inline std::vector<double> gaussianKernel(int size, double sigma) {
    std::vector<double> kernel(size);
    double mean = size/2;
    double sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < size; x++) {
        kernel[x] = std::exp(-0.5 * std::pow((x-mean)/sigma, 2)) / (sigma * std::sqrt(2*M_PI));
        // Accumulate the kernel values
        sum += kernel[x];
    }
    // Normalize the kernel
    for (int x = 0; x < size; x++)
        kernel[x] /= sum;
    return kernel;
}

inline Matrix3Xd applyGaussianFilter(const Matrix3Xd &Acc, int n, double sigma) {
    std::vector<double> kernel = gaussianKernel(n, sigma);
    Matrix3Xd filtered = Acc;
    int padding = n / 2;

    int colApply = 0;

    for (int row = 0; row < Acc.rows(); row++) {
        for (int col = 0; col < Acc.cols(); col++) {
            double sum = 0.0;
            for (int k = -padding; k <= padding; k++) {

                // limit bounds
                colApply = std::min( std::max(0,col + k), (int) (Acc.cols()-1) );
                sum += Acc(row, colApply) * kernel[k + padding];
            }
            filtered(row, col) = sum;
        }
    }
    return filtered;
}



inline VectorXd linspace(double x_start, double x_end, uint64_t N)
{
    VectorXd result(N);

    double delta = (x_end - x_start)/(  (double) (N-1)  );

    for (uint64_t k = 0; k < N; ++k) result(k) = x_start + (double) k * delta;

    return result;
}

inline Matrix3d axang2rotm(Vector3d axang)
{
    // calc skew symmetric
    Matrix3d skewSym;

    skewSym << 0, -axang(2), axang(1),\
            axang(2), 0, -axang(0),\
            -axang(1), axang(0),0;


    return skewSym.exp();
}

inline Vector3d rotm2axang(Matrix3d rotm)
{
    // calc skew symmetric
    Matrix3d skewSym = rotm.log();

    return Vector3d(skewSym(2,1),skewSym(0,2),skewSym(1,0));
}



inline void gridRandomDownsampling( PointCloud<PointXYZ> &pc_in, PointCloud<PointXYZ> &pc_filtered, float gridSize)
{
    /* downsampling by using a random point within a grid cell */

    octree::OctreePointCloud<PointXYZ> octree(gridSize); // set voxel size

    // Set the input point cloud to the octree
    octree.setInputCloud(pc_in.makeShared() );

    // Construct the octree
    octree.addPointsFromInputCloud();

    // resize filtered pc
    pc_filtered.resize(octree.getLeafCount () );

    int pc_fil_id = 0;
    double r;
    int id;

    for (auto it = octree.leaf_depth_begin(); it != octree.leaf_end(); ++it)
    {
        std::vector<int> indices;
        it.getLeafContainer().getPointIndices(indices);

        // get random number
        r = ((double) rand() / (RAND_MAX));
        id = (int) (r * (double) (indices.size()-1) );

        // add point
        pc_filtered.points[pc_fil_id] = pc_in.points[indices[id]];

        // update index
        ++pc_fil_id;
    }
}



struct pc_elem
{
    void set_size( uint64_t n )
    {
        XYZ.resize(3,n);
        Stamps.resize(n);
        Ids.resize(n);
        numPoints = n;
    }

    double getMinStamp() const { return Stamps.minCoeff(); }
    double getMaxStamp() const { return Stamps.maxCoeff(); }
    VectorXd getStamps() const {return Stamps;}
    VectorXi getRings() const {return Ids;}

    Matrix3Xd XYZ;
    VectorXd Stamps;
    VectorXi Ids;
    uint32_t numPoints=0;
};


struct optimParams
{
    void reset()
    {

        Orientations.setZero();
        Translations.setZero();
        
    }

    void setNumParams(uint64_t numParams, double horizon)
    {
        Orientations.resize(3,numParams);
        Translations.resize(3,numParams);

        Orientations.setZero();
        Translations.setZero();

        paramStamps = linspace(0.0, horizon, numParams);

    }

    int getNumElemsVector() const
    {
        return (paramStamps.size() - 1)*6;
    }

    VectorXd getParamsAsVector() const
    {

        VectorXd params(Orientations.cols()*Orientations.rows() + Translations.cols()*Translations.rows()-6);
        params << Orientations.block(0,1,3,Orientations.cols()-1).reshaped(Orientations.cols()*Orientations.rows()-3,1),\
                Translations.block(0,1,3,Translations.cols()-1).reshaped(Translations.cols()*Translations.rows()-3,1);

        return params;
    }

    void setParamsFromVector(VectorXd& params)
    {
        Orientations.block(0,1,3,Orientations.cols()-1) = params.head(Orientations.cols()*Orientations.rows()-3).reshaped(Orientations.rows(),Orientations.cols()-1);
        Translations.block(0,1,3,Translations.cols()-1) = params.tail(Translations.cols()*Translations.rows()-3).reshaped(Translations.rows(),Translations.cols()-1);
    }

    VectorXd getParamsAsVectorWithGravWithoutRot() const
    {

        VectorXd params(Orientations.rows()-1+Translations.cols()*Translations.rows()-3);

        // extract gravity as euler angles
        Vector3d eulerOrig = axang2rotm(Orientations.col(0)).eulerAngles(0,1,2);

        params << eulerOrig.head(2),\
                Translations.block(0,1,3,Translations.cols()-1).reshaped(Translations.cols()*Translations.rows()-3,1);

        return params;
    }

    void setParamsFromVectorWithGravWithoutRot(VectorXd& params)
    {
        // get new axangle from euler
        Vector3d eulerOrig = axang2rotm(Orientations.col(0)).eulerAngles(0,1,2);

        // recalc axang
        Matrix3d R;
        R= AngleAxisd(params[0], Vector3d::UnitX())
                * AngleAxisd(params[1], Vector3d::UnitY())
                * AngleAxisd(eulerOrig[2], Vector3d::UnitZ());

        Orientations.col(0) = rotm2axang(R);

        Translations.block(0,1,3,Translations.cols()-1) = params.tail(Translations.cols()*Translations.rows()-3).reshaped(Translations.rows(),Translations.cols()-1);
    }

    VectorXd getParamsAsVectorWithGrav() const
    {

        VectorXd params(Orientations.cols()*Orientations.rows() + Translations.cols()*Translations.rows()-4);


        params << 0.0, 0.0, Orientations.block(0,1,3,Orientations.cols()-1).reshaped(Orientations.cols()*Orientations.rows()-3,1),\
                Translations.block(0,1,3,Translations.cols()-1).reshaped(Translations.cols()*Translations.rows()-3,1);

        return params;
    }

    void setParamsFromVectorWithGrav(VectorXd& params)
    {
        // get new axangle from euler
        Matrix3d R_xy = axang2rotm( Vector3d(params(0),params(1),0.0) );

        // combine
        Matrix3d R_combined = axang2rotm( Orientations.col(0))*R_xy;

        Orientations.col(0) = rotm2axang(R_combined);


        Orientations.block(0,1,3,Orientations.cols()-1) = params.segment(2,Orientations.cols()*Orientations.rows()-3).reshaped(Orientations.rows(),Orientations.cols()-1);
        Translations.block(0,1,3,Translations.cols()-1) = params.tail(Translations.cols()*Translations.rows()-3).reshaped(Translations.rows(),Translations.cols()-1);
    }

    Matrix3Xd Orientations;
    Matrix3Xd Translations;

    VectorXd paramStamps;

};

class TransformHandler
{
public:
    optimParams trajParamsRel;
    optimParams trajParamsGlo;

    void setNumParams(int n)
    {
        trajParamsRel.setNumParams(n,1.0);
        trajParamsGlo.setNumParams(n,1.0);
    }


    void relative2global()
    {

        // shift parameters
        Matrix3d R = Matrix3d::Identity();
        Vector3d T(0.0f,0.0f,0.0f);

        for (uint32_t k = 0; k < trajParamsRel.Orientations.cols(); ++k)
        {
            // update translation
            T = T + R*trajParamsRel.Translations.col(k);
            trajParamsGlo.Translations.col(k) = T;

            // update rotation
            R = R * axang2rotm( trajParamsRel.Orientations.col(k) );

            // save global rotation
            trajParamsGlo.Orientations.col(k) = rotm2axang(R);
        }

    }

    void global2relative()
    {
        // copy init values
        trajParamsRel.Orientations.col(0) = trajParamsGlo.Orientations.col(0);
        trajParamsRel.Translations.col(0) = trajParamsGlo.Translations.col(0);

        // init params
        Matrix3d R1,R2;;
        Vector3d T1,T2;

        for (uint32_t k = trajParamsRel.Orientations.cols()-1; k > 0; --k)
        {
            R1 = axang2rotm( trajParamsGlo.Orientations.col(k-1) );
            T1 = trajParamsGlo.Translations.col(k-1);

            R2 = axang2rotm( trajParamsGlo.Orientations.col(k) );
            T2 = trajParamsGlo.Translations.col(k);

            // save relative rotation and translations
            trajParamsRel.Orientations.col(k) = rotm2axang(R1.transpose()*R2);
            trajParamsRel.Translations.col(k) = R1.transpose()*(T2-T1);
        }

    }



};

struct Keyframe
{
    Matrix3Xd cloud_imu;
    double keyframeStamp;

};

class MapManagement : public TransformHandler
{
public:
    MapManagement()
    {  
        resetMap();
    }

    Vector3d getLastKeyframePosition()
    {
        return trajParamsGlo.Translations.col(getNumKeyframes()-1);
    }

    Vector3d getLastKeyframeOrientation()
    {
        return trajParamsGlo.Orientations.col(getNumKeyframes()-1);
    }


    void updateFromSubmap(int from_id, int to_id, MapManagement& submap)
    {
        submap.global2relative();

        global2relative();

        int numParams = to_id - from_id + 1;

        trajParamsRel.Translations.block(0,from_id+1,3,numParams-1) = submap.trajParamsRel.Translations.block(0,1,3,numParams-1);
        trajParamsRel.Translations.block(0,from_id+1,3,numParams-1) = submap.trajParamsRel.Translations.block(0,1,3,numParams-1);

        relative2global();
    }

    MapManagement getSubmap(int from_id, int to_id)
    {
        MapManagement submap;

        int numParams = to_id - from_id + 1;

        submap.trajParamsGlo.setNumParams(numParams,0.0);
        submap.trajParamsRel.setNumParams(numParams,0.0);

        // copy parameters
        submap.trajParamsGlo.Translations.block(0,0,3,numParams) = trajParamsGlo.Translations.block(0,from_id,3,numParams);
        submap.trajParamsGlo.Orientations.block(0,0,3,numParams) = trajParamsGlo.Orientations.block(0,from_id,3,numParams);

        // update relative params
        submap.global2relative();

        // copy keyframes
        //submap.Keyframes.reserve(numParams);
        for(int k = 0; k< numParams; ++k) submap.Keyframes.push_back(Keyframes[from_id+k]);

        return submap;
    }


    void addKeyframe(Vector3d pos_w, Vector3d axang_w, Matrix3Xd points_imu, double stamp)
    {
        // update number of points
        numPoints += (uint64_t)  points_imu.cols();

        if (IS_INITIALIZED == false)
        {
            // init
            trajParamsGlo.Translations.resize(3,1);
            trajParamsGlo.Orientations.resize(3,1);
            trajParamsRel.Translations.resize(3,1);
            trajParamsRel.Orientations.resize(3,1);

            // update status
            IS_INITIALIZED = true;
        }
        else
        {
            // add columns
            trajParamsGlo.Translations.conservativeResize(3,trajParamsGlo.Translations.cols()+1);
            trajParamsGlo.Orientations.conservativeResize(3,trajParamsGlo.Orientations.cols()+1);
            trajParamsRel.Translations.conservativeResize(3,trajParamsRel.Translations.cols()+1);
            trajParamsRel.Orientations.conservativeResize(3,trajParamsRel.Orientations.cols()+1);

        }

        // save transform
        trajParamsGlo.Translations.col(trajParamsGlo.Translations.cols()-1) = pos_w;
        trajParamsGlo.Orientations.col(trajParamsGlo.Translations.cols()-1) = axang_w;

        // update relative params
        global2relative();

        // transform points in imu frame
        Keyframe newKeyframe;

        newKeyframe.cloud_imu = points_imu;
        newKeyframe.keyframeStamp = stamp;

        // add keyframe
        Keyframes.push_back(newKeyframe);

    }

    void updateGlobalCloud()
    {
        uint64_t numPointsSparse=0;
        for (int k=0; k < Keyframes.size(); ++k) numPointsSparse += Keyframes[k].cloud_imu.cols();

        // init global cloud
        globalCloud.set_size(numPointsSparse);

        // update transforms
        relative2global();

        // init index
        uint64_t startId = 0;

        for (int k=0; k < Keyframes.size(); ++k)
        {
            // transform cloud and copy in global cloud matrix
            globalCloud.XYZ.block(0,startId,3,Keyframes[k].cloud_imu.cols()) = axang2rotm(trajParamsGlo.Orientations.col(k))*Keyframes[k].cloud_imu;
            globalCloud.XYZ.block(0,startId,3,Keyframes[k].cloud_imu.cols()).colwise() += trajParamsGlo.Translations.col(k);

            // save keyframe id
            globalCloud.Ids.segment(startId,Keyframes[k].cloud_imu.cols()).setConstant(k);

            // update start id
            startId += Keyframes[k].cloud_imu.cols();
        }

        // update number of points
        numPoints = globalCloud.XYZ.cols();

    }

    void resetMap()
    {
        IS_INITIALIZED = false;
        Keyframes.clear();
        numPoints = 0;
    }

    int getNumKeyframes() { return Keyframes.size(); }


    void setKeyframeSize(int n)
    {
        Keyframes.resize(n);

        setNumParams(n);

    }

    std::vector<double> getKeyframeStamps()
    {
        std::vector<double> Stamps;

        Stamps.resize(getNumKeyframes());

        for(int k=0; k<getNumKeyframes();++k) Stamps[k] = Keyframes[k].keyframeStamp;

        return Stamps;
    }

    std::vector<Keyframe> Keyframes;

    bool IS_INITIALIZED = false;

    pc_elem globalCloud;
    uint64_t numPoints = 0;
};



template<typename Derived>
inline bool is_nan(const MatrixBase<Derived>& x)
{
    return ((x.array() == x.array())).all();
}



inline void visualizePc(visualization::PCLVisualizer& viewer,Matrix3Xd path,Matrix3Xd& globalCloud, Vector3f origin)
{
    // path
    PointCloud<PointXYZ>::Ptr cloudP(new PointCloud<PointXYZ>);
    cloudP->resize(path.cols());

    for (int i = 0; i < path.cols(); i++)
    {
        PointXYZ point;
        point.x = path(0, i)-origin(0);
        point.y = path(1, i)-origin(1);
        point.z = path(2, i)-origin(2);
        cloudP->points[i] = point;
    }

    // globalCloud
    PointCloud<PointXYZ>::Ptr globalCloudPCL(new PointCloud<PointXYZ>);
    globalCloudPCL->resize(globalCloud.cols());

    for (int i = 0; i < globalCloud.cols(); i++)
    {
        PointXYZ point;
        point.x = globalCloud(0, i)-origin(0);
        point.y = globalCloud(1, i)-origin(1);
        point.z = globalCloud(2, i)-origin(2);
        globalCloudPCL->points[i] = point;
    }

    visualization::PointCloudColorHandlerCustom<PointXYZ> single_colorR(cloudP, 255, 0, 0);
    single_colorR = visualization::PointCloudColorHandlerCustom<PointXYZ>(cloudP, 255, 0, 0);

    viewer.setBackgroundColor(0, 0, 0);
    viewer.removePointCloud("path");
    viewer.addPointCloud(cloudP,single_colorR, "path");
    viewer.removePointCloud("global");
    viewer.addPointCloud(globalCloudPCL, "global");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "path");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, "global");


    //viewer.registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

    //while (!viewer.wasStopped())
    //{
     //   viewer.spinOnce();
    //}
   // viewer.spinOnce(1);




}



