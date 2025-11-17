#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <KeyFrame.h>
#include <Converter.h>
#include <Tracking.h>
#include <MapPoint.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ORBSLAM3Wrapper.h"
#include "NDArrayConverter.h"

namespace py = pybind11;

ORBSLAM3Python::ORBSLAM3Python(std::string vocabFile, std::string settingsFile, ORB_SLAM3::System::eSensor sensorMode): 
    vocabluaryFile(vocabFile),
    settingsFile(settingsFile),
    sensorMode(sensorMode),
    system(nullptr),
    bUseViewer(false)
{
}

ORBSLAM3Python::~ORBSLAM3Python()
{
}

bool ORBSLAM3Python::initialize()
{
    system = std::make_shared<ORB_SLAM3::System>(vocabluaryFile, settingsFile, sensorMode, bUseViewer);
    return true;
}

bool ORBSLAM3Python::processMono(cv::Mat image, double timestamp)
{
    if (!system)
    {
        return false;
    }
    if (image.data)
    {
        Sophus::SE3f pose = system->TrackMonocular(image, timestamp);
        return !system->isLost();
    }
    else
    {
        return false;
    }
}

bool ORBSLAM3Python::processStereo(cv::Mat leftImage, cv::Mat rightImage, double timestamp)
{
    if (!system)
    {
        std::cout << "you must call initialize() first!" << std::endl;
        return false;
    }
    if (leftImage.data && rightImage.data)
    {
        auto pose = system->TrackStereo(leftImage, rightImage, timestamp);
        return !system->isLost();
    }
    else
    {
        return false;
    }
}

bool ORBSLAM3Python::processRGBD(cv::Mat image, cv::Mat depthImage, double timestamp)
{
    if (!system)
    {
        std::cout << "you must call initialize() first!" << std::endl;
        return false;
    }
    if (image.data && depthImage.data)
    {
        auto pose = system->TrackRGBD(image, depthImage, timestamp);
        return !system->isLost();
    }
    else
    {
        return false;
    }
}

void ORBSLAM3Python::reset()
{
    if (system)
    {
        system->Reset();
    }
}

void ORBSLAM3Python::shutdown()
{
    if (system)
    {
        system->Shutdown();
    }
}

bool ORBSLAM3Python::isShutDown()
{
    if(system)
    {
        return system->isShutDown();
    }
}

bool ORBSLAM3Python::isRunning()
{
    return system != nullptr;
}

void ORBSLAM3Python::setUseViewer(bool useViewer)
{
    bUseViewer = useViewer;
}

std::vector<Eigen::Matrix4f> ORBSLAM3Python::getTrajectory() const
{
    return system->GetCameraTrajectory();
}

bool ORBSLAM3Python::ViewerShouldQuit()
{
    return system->ViewerShouldQuit();
}

int ORBSLAM3Python::GetNumMapsInAtlas()
{
    return system->GetNumMapsInAtlas();
}

int ORBSLAM3Python::GetActiveMapID()
{
    return system->GetActiveMapID();
}

std::vector<double> ORBSLAM3Python::GetAllKeyFrameTimes()
{
    std::vector<double> vTimes = system->GetAllKeyFrameTimes();
   return vTimes;
}

std::vector<std::array<float,16>> ORBSLAM3Python::GetAllKeyFramePoses()
{
    return system->GetAllKeyFramePoses();
}

py::array_t<float> ORBSLAM3Python::GetAllKeyFramePosesNP()
{
    auto poses = GetAllKeyFramePoses();
    py::ssize_t N = poses.size();
    py::array_t<float> arr({N,(py::ssize_t)4,(py::ssize_t)4});
    auto buf = arr.mutable_unchecked<3>();
    for(py::ssize_t i=0;i<N;++i)
        std::memcpy(buf.mutable_data(i,0,0), poses[i].data(), 16*sizeof(float));
    return arr;
}

std::vector<int> ORBSLAM3Python::GetAllKeyFrameMapIDs()
{
    std::vector<int> vMapIDs = system->GetAllKeyFrameMapIDs();
    return vMapIDs;
}

py::tuple ORBSLAM3Python::GetAllKeyFrameData()
{
    std::vector<double> times;
    std::vector<std::array<float,16>> poses_aos;
    std::vector<int> mapIDs;
    system->GetAllKeyFrameData(times, poses_aos, mapIDs);

    py::ssize_t N = static_cast<py::ssize_t>(poses_aos.size());
    py::array_t<float> poses_np({N, (py::ssize_t)4, (py::ssize_t)4});
    auto buf = poses_np.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < N; ++i)
    {
        // Copy 16 floats row-major into the numpy slice
        std::memcpy(buf.mutable_data(i, 0, 0), poses_aos[i].data(), 16*sizeof(float));
    }

    return py::make_tuple(times, poses_np, mapIDs);
}

PYBIND11_MODULE(orbslam3, m)
{
    NDArrayConverter::init_numpy();
    py::enum_<ORB_SLAM3::Tracking::eTrackingState>(m, "TrackingState")
        .value("SYSTEM_NOT_READY", ORB_SLAM3::Tracking::eTrackingState::SYSTEM_NOT_READY)
        .value("NO_IMAGES_YET", ORB_SLAM3::Tracking::eTrackingState::NO_IMAGES_YET)
        .value("NOT_INITIALIZED", ORB_SLAM3::Tracking::eTrackingState::NOT_INITIALIZED)
        .value("OK", ORB_SLAM3::Tracking::eTrackingState::OK)
        .value("RECENTLY_LOST", ORB_SLAM3::Tracking::eTrackingState::RECENTLY_LOST)
        .value("LOST", ORB_SLAM3::Tracking::eTrackingState::LOST)
        .value("OK_KLT", ORB_SLAM3::Tracking::eTrackingState::OK_KLT);

    py::enum_<ORB_SLAM3::System::eSensor>(m, "Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::eSensor::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::eSensor::STEREO)
        .value("RGBD", ORB_SLAM3::System::eSensor::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::eSensor::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::eSensor::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::eSensor::IMU_RGBD);

    py::class_<ORBSLAM3Python>(m, "system")
        .def(py::init<std::string, std::string, ORB_SLAM3::System::eSensor>(), py::arg("vocab_file"), py::arg("settings_file"), py::arg("sensor_type"))
        .def("initialize", &ORBSLAM3Python::initialize)
        .def("process_image_mono", &ORBSLAM3Python::processMono, py::arg("image"), py::arg("time_stamp"))
        .def("process_image_stereo", &ORBSLAM3Python::processStereo, py::arg("left_image"), py::arg("right_image"), py::arg("time_stamp"))
        .def("process_image_rgbd", &ORBSLAM3Python::processRGBD, py::arg("image"), py::arg("depth"), py::arg("time_stamp"))
        .def("reset", &ORBSLAM3Python::reset)
        .def("shutdown", &ORBSLAM3Python::shutdown)
        .def("is_shutdown", &ORBSLAM3Python::isShutDown)
        .def("is_running", &ORBSLAM3Python::isRunning)        
        .def("viewer_should_quit", &ORBSLAM3Python::ViewerShouldQuit)
        .def("set_use_viewer", &ORBSLAM3Python::setUseViewer)
        .def("get_trajectory", &ORBSLAM3Python::getTrajectory)
        
        // ---------------------------------------------------------------
        // Custom Functions
        // ---------------------------------------------------------------
        .def("get_num_maps_in_atlas", &ORBSLAM3Python::GetNumMapsInAtlas)
        .def("get_active_map_id", &ORBSLAM3Python::GetActiveMapID)
        .def("get_all_keyframe_times", &ORBSLAM3Python::GetAllKeyFrameTimes, py::return_value_policy::copy)
        .def("get_all_keyframe_map_ids", &ORBSLAM3Python::GetAllKeyFrameMapIDs, py::return_value_policy::copy)
        .def("get_all_keyframe_poses", &ORBSLAM3Python::GetAllKeyFramePosesNP);
}