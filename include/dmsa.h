


#include "helpers.h"


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_search.h>
#include <eigen3/Eigen/Eigenvalues>

#include <Eigen/SparseCore>

class Gaussians
{
    public:
    Gaussians()
    {
        // init
        connectedPointIds.resize(maxSets);
        errorVecIds.resize(maxSets);
        infoMats.resize(maxSets);
        numPointsPerSet.resize(maxSets);
        numPointsPerSet.setZero();
        rebalancingWeights.resize(maxSets);
        obervationWeights.resize(maxSets);
        obervationWeights.setZero();

        Scores.resize(maxSets);
        Scores.setZero();

        numPointSets = 0;
        numPoints = 0;
    }

    void reset()
    {
        numPointsPerSet.setZero();
        obervationWeights.setZero();

        numPointSets = 0;
        numPoints = 0;

        sumOfInfoNorms = 0.000;
        sumOfCovNorms = 0.000;
        sumOfWeightedInfoNorms = 0.000;
        differentialEntropy = 0.000;
        realDifferentialEntropy = 0.000;
        numSkips = 0;
    }

    void addPointSet(std::vector<int> ids, const pc_elem &cloudTraj, float gridSize, double observationWeight=1.0)
    {
        // prevent overflow
        if (numPointSets>=maxSets)
        {
            std::cerr <<  "Point set limit reached in Gaussians!" << std::endl;
            return;
        }

        /* calculate info mat */
        Eigen::MatrixX3d subset = cloudTraj.XYZ(Eigen::all,Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(ids.data(), ids.size())).transpose();

        MatrixXd centered = subset.rowwise() - subset.colwise().mean();
        Matrix3d cov = (centered.adjoint() * centered) / double(subset.rows() - 1);
        Matrix3d cov_lim;

        sumOfCovNorms = sumOfCovNorms + cov.norm();

        double addScore = std::log(1+cov.inverse().determinant());

        // generate pseudo cov and check if voxel contains information
        if ( generatePseudoCov(cov,limited_cov,gridSize) == 0)
        {
            ++numSkips;
            return;
        }

        differentialEntropy = differentialEntropy + addScore;

        // save information matrix
        infoMats[numPointSets] = cov.inverse();

        sumOfInfoNorms = sumOfInfoNorms + infoMats[numPointSets].norm();

        //sumOfWeightedInfoNorms = sumOfWeightedInfoNorms + (double) connectedPointIds[k].size() * infoMats[k].norm();



        realDifferentialEntropy = realDifferentialEntropy + log(cov.determinant() );

        // save observation weight
        obervationWeights(numPointSets) = observationWeight;

        // copy ids
        connectedPointIds.at(numPointSets) = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(ids.data(), ids.size());

        // add ids in error vector
        int startVecId = 0;

        // get id from last set
        if(numPointSets>0) startVecId = errorVecIds[numPointSets-1](errorVecIds[numPointSets-1].size()-1) + 1;

        // resize
        errorVecIds[numPointSets].resize(ids.size());

        // save ids
        for (int k = 0; k < ids.size(); ++k) errorVecIds[numPointSets](k) = startVecId + k;

        // save number of points
        numPointsPerSet(numPointSets) = ids.size();

        // update
        ++numPointSets;
        numPoints += ids.size();
    }

    void updateScores(const pc_elem &cloudTraj, float gridSize)
    {

        #pragma omp parallel num_threads(8)
        for (int k = 0; k < numPointSets; ++k)
        {
            Eigen::MatrixX3d subset = cloudTraj.XYZ(Eigen::all,connectedPointIds[k]).transpose();

            MatrixXd centered = subset.rowwise() - subset.colwise().mean();
            Matrix3d cov = (centered.adjoint() * centered) / double(subset.rows() - 1);

            int i = generatePseudoCov(cov,false,gridSize);

            //Scores(k) =  std::log(1.0+cov.inverse().determinant()*1e-4);
            Scores(k) =  std::sqrt(cov.inverse().determinant());
            //Scores(k) =  -cov.determinant();


        }


    }

    void updateRebalancingWeights()
    {
        rebalancingWeights.head(numPointSets) = numPointsPerSet.head(numPointSets).cast<double>().array().pow(-1).matrix();

        // add information weights
        rebalancingWeights.head(numPointSets) = rebalancingWeights.head(numPointSets).cwiseProduct(obervationWeights.head(numPointSets));


        rebalancingWeights.head(numPointSets) = rebalancingWeights.head(numPointSets)/rebalancingWeights.head(numPointSets).mean();
        //rebalancingWeights.setConstant(1.0);
    }

    int generatePseudoCov(Matrix3d& cov,bool limit_cov,float& gridSize)
    {
        Eigen::EigenSolver<Eigen::Matrix3d> eigensolver;
        eigensolver.compute(cov);
        int dimensionsWithInformation = 3;

         Eigen::Vector3d eigen_values = eigensolver.eigenvalues().real();
         Eigen::Matrix3d eigen_vectors = eigensolver.eigenvectors().real();

         // modify eigen values
         for (int k = 0; k < 3; ++k)
         {
             if(limit_cov) eigen_values(k) = std::max(eigen_values(k), 0.00010);

             // limit acc. to grid size
             if(eigen_values(k) > std::pow(gridSize*0.25,2) )
             {
                 eigen_values(k) = 10.0*gridSize;
                 dimensionsWithInformation = dimensionsWithInformation - 1;
             }
         }


         // create diagonal matrix
         Eigen::DiagonalMatrix<double, 3> diagonal_matrix( eigen_values(0),eigen_values(1),eigen_values(2) );

         // update covariance
         cov = eigen_vectors*diagonal_matrix*eigen_vectors.inverse();

         // return number of dimensions with information
         return dimensionsWithInformation;

    }

    

    std::vector<Eigen::VectorXi> connectedPointIds;
    std::vector<Eigen::VectorXi> errorVecIds;
    std::vector<Eigen::Matrix3d> infoMats;

    Eigen::VectorXd Scores;

    double oneScore;

    Eigen::VectorXi numPointsPerSet;
    Eigen::VectorXd rebalancingWeights;
    Eigen::VectorXd obervationWeights;

    double sumOfInfoNorms;
    double sumOfWeightedInfoNorms;
    double sumOfCovNorms;
    double differentialEntropy;
    double realDifferentialEntropy;
    int numSkips=0;

    Eigen::Matrix3d InformationBalance;

    bool limited_cov = false;

    int numPointSets;
    int maxSets = 50000;
    uint64_t numPoints;
};

class dmsa {
    public:
    dmsa(float gridSize)
    {
        GRID_OCTREE_MAP = gridSize;
    };

    float GRID_OCTREE_MAP = 1.0f;

    double epsilon_keyframe_opt=0.0005;

    double meanError;
    double meanUnbalancedError;



Gaussians currentGauss;

double optimizeMap(MapManagement *Map, int numIter, double& alpha, bool REBALANCE, bool select_best_set, bool limited_cov, double inlierRatio, bool reduce_gain=true);

double optimizeMap2(MapManagement *Map, int numIter, double& alpha, bool REBALANCE, bool select_best_set, bool limited_cov, double inlierRatio,bool reduce_gain=true);

void getNumericJacobianMap(MapManagement &Map, const Eigen::VectorXd& Error0, Eigen::Ref<MatrixXd> Jacobian);

void getNumericJacobianMap2(MapManagement &Map, const Eigen::VectorXd& Error0, Eigen::Ref<Eigen::MatrixXd> Jacobian);


};
