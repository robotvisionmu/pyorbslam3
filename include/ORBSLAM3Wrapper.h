#ifndef ORB_SLAM3_PYTHON_H
#define ORB_SLAM3_PYTHON_H

#include <memory>
#include <System.h>
#include <Tracking.h>
#include <iostream>
#include <map>

namespace py = pybind11;
class ORBSLAM3Python
{
public:
    ORBSLAM3Python(std::string vocabFile, std::string settingsFile, ORB_SLAM3::System::eSensor sensorMode = ORB_SLAM3::System::eSensor::RGBD);
    ~ORBSLAM3Python();

    bool initialize();
    bool processMono(cv::Mat image, double timestamp);
    bool processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp);
    bool processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp);
    void reset();
    void shutdown();
    bool isShutDown();
    bool isRunning();
    void setUseViewer(bool useViewer);
    std::vector<Eigen::Matrix4f> getTrajectory() const;
    bool ViewerShouldQuit();
    int GetNumMapsInAtlas();
    int GetActiveMapID();
    std::vector<double> GetAllKeyFrameTimes();
    std::vector<std::array<float,16>> GetAllKeyFramePoses();
    py::array_t<float> GetAllKeyFramePosesNP();
    std::vector<int> GetAllKeyFrameMapIDs();
    py::tuple GetAllKeyFrameData();
    py::array_t<float> GetActiveKeyFramePosesNP();

private:
    std::string vocabluaryFile;
    std::string settingsFile;
    ORB_SLAM3::System::eSensor sensorMode;
    std::shared_ptr<ORB_SLAM3::System> system;
    bool bUseViewer;
    bool bUseRGB;
};

#endif // ORB_SLAM3_PYTHON_H