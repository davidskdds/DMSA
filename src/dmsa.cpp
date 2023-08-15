#include "dmsa.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <algorithm>    // std::sort
#include <numeric>
#include <omp.h>


double dmsa::optimizeMap(MapManagement *Map, int numIter, double alpha, bool REBALANCE, bool select_best_set, bool limited_cov, double inlierRatio,bool reduce_gain)
{
    // SAVE INITIAL TRANSLATION
    Map->global2relative();

    Eigen::Vector3d initialTranslation = Map->trajParamsRel.Translations.col(0);
    Map->trajParamsRel.Translations.col(0).setZero();
    Map->relative2global();
    // END


    // update cloud
    Map->updateGlobalCloud();
    currentGauss.rebalancingWeights.setConstant(1.0);

    currentGauss.limited_cov = limited_cov;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTraj_pcl_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::MatrixXd Jacobian(Map->numPoints+Map->getNumKeyframes(),(Map->getNumKeyframes())*6-6);

    // get sparse anchor points
    convertEigenToPointCloud(Map->globalCloud.XYZ,cloudTraj_pcl_ptr);

    // Create an octree object
    pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(GRID_OCTREE_MAP); // set voxel size

    // Set the input point cloud to the octree
    octree.setInputCloud(cloudTraj_pcl_ptr);

    // Construct the octree
    octree.addPointsFromInputCloud();

    // Create an octree object
    //pcl::octree::OctreePointCloud<pcl::PointXYZ> octreeCoarse(1.5); // set voxel size


    // Set the input point cloud to the octree
    //octreeCoarse.setInputCloud(cloudTraj_pcl_ptr);

    // Construct the octree
    //octreeCoarse.addPointsFromInputCloud();

    // init best set selection
    Eigen::VectorXd bestSetRel = Map->trajParamsRel.getParamsAsVector();
    double maxEntropy;
    bool minEntropyInit = false;
    double currError;
    int iter;

    for(iter = 0; iter < numIter+1; ++iter)
    {

        // iterate over leaf nodes
        currentGauss.reset();

        //for (auto it = octree.leaf_begin(); it != octree.leaf_end(); ++it)
        for (auto it = octree.leaf_depth_begin(); it != octree.leaf_end(); ++it)
        {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);

            // check rings
            Eigen::VectorXi IdsSet = Map->globalCloud.Ids(Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size()));

            if ( (indices.size() > 6) && (IdsSet.maxCoeff()-IdsSet.minCoeff() != 0) )
            {
                currentGauss.addPointSet(indices,Map->globalCloud,octree.getResolution());
            }

        }

        /*
        // octree coarse
        for (auto it = octreeCoarse.leaf_depth_begin(); it != octreeCoarse.leaf_end(); ++it)
        {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);

            // check rings
            Eigen::VectorXi IdsSet = Map->globalCloud.Ids(Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size()));

            if ( (indices.size() > 6) && (IdsSet.maxCoeff()-IdsSet.minCoeff() != 0) )
            {
                currentGauss.addPointSet(indices,Map->globalCloud,octreeCoarse.getResolution());
            }

        }
        */

        // update info mats and weights
        //currentGauss.updateInfosMats(Map->globalCloud,GRID_OCTREE_MAP);
        if (REBALANCE) currentGauss.updateRebalancingWeights();

        // init error vector
        Eigen::VectorXd Error(currentGauss.numPoints);
        Eigen::VectorXd BalancedPointToOriginError(currentGauss.numPoints);
        Error.setZero();
        int vecId = 0;
        Eigen::Vector3d mean,diffVec;
        Eigen::Matrix3Xd subset;


        // loop over point sets
        for(int k = 0; k < currentGauss.numPointSets; ++k)
        {
            // get subset
            subset = Map->globalCloud.XYZ(Eigen::all,currentGauss.connectedPointIds[k]);

            // calc mean
            mean = subset.rowwise().mean();

            // loop over points within sets
            for(int j = 0; j < currentGauss.connectedPointIds[k].size(); ++j)
            {
                diffVec = Map->globalCloud.XYZ.col(currentGauss.connectedPointIds[k](j)) - mean;

                Error(vecId) = currentGauss.rebalancingWeights(k) * diffVec.transpose() * currentGauss.infoMats[k] * diffVec ;
                BalancedPointToOriginError(vecId) = currentGauss.rebalancingWeights(k)*diffVec.transpose() * diffVec ;

                // update vector id
                ++vecId;
            }

        }

        /* BEST SET SELECTION */

        if(!minEntropyInit)
        {
            maxEntropy = currentGauss.differentialEntropy;
            bestSetRel = Map->trajParamsRel.getParamsAsVector();
            minEntropyInit = true;
            //std::cout << "\nStart: Map LiDAR error / imu error [%]: "<< 100.0*(errorSum-imuError)/errorSum << " / " << 100.0*(imuError)/errorSum <<std::endl;
            //std::cout << "Map LiDAR error / imu error abs: "<< (errorSum-imuError) << " / " << (imuError) <<std::endl;
            //std::cout << "Entropy: "<< maxEntropy <<std::endl;
            //std::cout << "\nimu error: "<< Map->imuFactorError.transpose() <<" imu_factor_weight: "<<imu_factor_weight <<std::endl;

        }

        // update best set
        if (currentGauss.differentialEntropy > maxEntropy)
        {
            bestSetRel = Map->trajParamsRel.getParamsAsVector();
            maxEntropy = currentGauss.differentialEntropy;
        }
        else if(reduce_gain)
        {
            alpha = std::max(0.9*alpha,0.2);
        }

        // update trajectory with best set in last iteration
        if (iter == numIter)
        {
            if(select_best_set) Map->trajParamsRel.setParamsFromVector(bestSetRel);

            // update global cloud
            Map->updateGlobalCloud();
            //std::cout << "End: Map LiDAR error / imu error [%]: "<< 100.0*(errorSum-imuError)/errorSum << " / " << 100.0*(imuError)/errorSum <<std::endl;
            //std::cout << "Map LiDAR error / imu error abs: "<< (errorSum-imuError) << " / " << (imuError) <<std::endl;
            //std::cout << "Entropy: "<< maxEntropy <<"\n"<<std::endl;

            break;
        }

        // calculate Jacobian
        getNumericJacobianMap(*Map,Error, Jacobian.topRows(currentGauss.numPoints+Map->getNumKeyframes()));

        /* OUTLIER REMOVAL */
        int thresholdIndex = (int) floor( (float) currentGauss.numPoints * inlierRatio);

        // get sorted indices
        std::vector<int> Indices_std(currentGauss.numPoints);
        std::iota(Indices_std.begin(),Indices_std.end(),0); //Initializing
        sort( Indices_std.begin(),Indices_std.end(), [&](int i,int j){return Error[i]<Error[j];} );

        Eigen::VectorXi Indices = Eigen::Map<Eigen::VectorXi>(Indices_std.data(), Indices_std.size());

        // remove outliers
        Eigen::VectorXd ErrorWoOutliers = Error(Indices.head(thresholdIndex));
        Eigen::MatrixXd JacobianWoOutliers = Jacobian(Indices.head(thresholdIndex),Eigen::placeholders::all);

        Eigen::VectorXd optimStep;


        // optimization step
        Eigen::MatrixXd H = JacobianWoOutliers.transpose()*JacobianWoOutliers;

        // add lambda
        H.diagonal().array() += 0.00001;

        // optimize
        optimStep = -alpha* H.inverse() * JacobianWoOutliers.transpose() * ErrorWoOutliers;


        // stop in case of nan
        if  ( !is_nan(optimStep) )
        {
            ROS_ERROR_STREAM( "optimStep contains nan: "<<optimStep.transpose() );
            if(select_best_set) Map->trajParamsRel.setParamsFromVector(bestSetRel);

            break;
        }

        // epsilon stop condition
        if  ( std::max( optimStep.maxCoeff(),-optimStep.minCoeff()) < epsilon_keyframe_opt  )
        {
            ROS_INFO_STREAM("Epsilon stop condition after "<<iter<<" iterations in map optimization . . .");
            if(select_best_set) Map->trajParamsRel.setParamsFromVector(bestSetRel);
            break;
        }

        // update relative poses
        Eigen::VectorXd optimVec = Map->trajParamsRel.getParamsAsVector();
        optimVec = optimVec + optimStep;
        Map->trajParamsRel.setParamsFromVector(optimVec);

        // update global cloud
        Map->updateGlobalCloud();
        //std::cout << "NumSkips / numSets: "<<currentGauss.numSkips<<" / "<<currentGauss.numPointSets<<std::endl;
    }
    //std::cout<< "Map optimization quit after iteration: "<<iter<<std::endl;

    // ADD INITIAL TRANSLATION
    Map->trajParamsRel.Translations.col(0) = initialTranslation;
    Map->relative2global();
    // END

    //std::cout << "Max / mean / n_elem gravity error [deg]: "<< 180.0/M_PI * sqrt(Map->GravityEstimError.maxCoeff()/(Map->infoGrav*Map->balancingFactorGrav))<<" / " <<180.0/M_PI * sqrt(Map->GravityEstimError.mean()/(Map->infoGrav*Map->balancingFactorGrav))<< " / "<< Map->GravityEstimError.size()<<std::endl;

    // reset
    //cloudTraj_pcl_ptr.reset();

    // return mean error
    if(select_best_set) return maxEntropy;
    else return currentGauss.differentialEntropy;

}

void dmsa::getNumericJacobianMap(MapManagement &Map, const Eigen::VectorXd& Error0, Eigen::Ref<Eigen::MatrixXd> Jacobian)
{
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(4); // Use 4 threads for all consecutive parallel regions

    // set zero
    Jacobian.setZero();

    // get parameter vector
    Eigen::VectorXd oldRelativeParamVec = Map.trajParamsRel.getParamsAsVector();

    // main loop
    //#pragma omp parallel num_threads(4)
    for (int k = 0; k < oldRelativeParamVec.size(); ++k)
    {

        /* MODIFY TRAJECTORY AND TRANSFORM POINTS */

        // step size tuning parameter
        double deltaStep = 1.0 * sqrt( std::numeric_limits<double>::epsilon() );

        // get "clean" vector
        Eigen::VectorXd optimVec = oldRelativeParamVec;

        // add step
        optimVec(k) += deltaStep;

        // use modified vector
        Map.trajParamsRel.setParamsFromVector(optimVec);

        // update reference poses and global cloud
        Map.updateGlobalCloud();

        /* CALCULATE NEW ERROR */

        // init error vector
        Eigen::VectorXd Error(currentGauss.numPoints);
        Error.setZero();
        int vecId = 0;

        // loop over point sets
        #pragma omp parallel num_threads(4)
        for(int i = 0; i < currentGauss.numPointSets; ++i)
        {
            Eigen::Vector3d mean,diffVec;

            // calculate mean
            mean.setZero();

            for(int j = 0; j < currentGauss.connectedPointIds[i].size(); ++j)
            {
                mean = mean + Map.globalCloud.XYZ.col(currentGauss.connectedPointIds[i](j));
            }

            mean = mean / (double) (currentGauss.connectedPointIds[i].size());

            // loop over points within sets
            for(int j = 0; j < currentGauss.connectedPointIds[i].size(); ++j)
            {
                diffVec = Map.globalCloud.XYZ.col(currentGauss.connectedPointIds[i](j)) - mean;

                Error(currentGauss.errorVecIds[i](j)) = currentGauss.rebalancingWeights(i) * diffVec.transpose() * currentGauss.infoMats[i] * diffVec;
            }

        }



        /* ADD TO JACOBIAN */
        Jacobian.col(k) = ( Error - Error0 ) / deltaStep;

    }

    // reset with old params
    Map.trajParamsRel.setParamsFromVector(oldRelativeParamVec);

    // update reference poses and global cloud
    Map.updateGlobalCloud();

}
