#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <algorithm>

inline Eigen::MatrixXd readPosesFromFile(const std::string& directory) {
    std::ifstream infile(directory + "/poses.txt");
    if (!infile.is_open()) {
        std::cerr << "Failed to open poses.txt" << std::endl;
        return Eigen::MatrixXd();  // return an empty matrix
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double value;
        std::vector<double> row;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    if (data.empty() || data[0].empty()) {
        std::cerr << "The file is empty or improperly formatted." << std::endl;
        return Eigen::MatrixXd();  // return an empty matrix
    }

    Eigen::MatrixXd matrix(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

inline std::vector<std::string> listFilesInDirectory(const std::string& directory) {
    DIR* dir;
    struct dirent* ent;
    std::vector<std::string> files;

    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            files.push_back(ent->d_name);
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open directory" << std::endl;
        return files;
    }

    std::sort(files.begin(), files.end());
    return files;
}


