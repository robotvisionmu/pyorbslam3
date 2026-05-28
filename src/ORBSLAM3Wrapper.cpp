#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstring>
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

namespace
{
py::array_t<float> Array16ToNumpy(const std::array<float,16>& data)
{
    py::array_t<float> arr({(py::ssize_t)4, (py::ssize_t)4});
    auto buf = arr.mutable_unchecked<2>();
    std::memcpy(buf.mutable_data(0, 0), data.data(), 16 * sizeof(float));
    return arr;
}

const char* MapEventTypeToString(ORB_SLAM3::Atlas::MapEventType type)
{
    switch(type)
    {
        case ORB_SLAM3::Atlas::MapEventType::MapCreated:
            return "map_created";
        case ORB_SLAM3::Atlas::MapEventType::MapMerged:
            return "map_merged";
        case ORB_SLAM3::Atlas::MapEventType::MapRemoved:
            return "map_removed";
        default:
            return "unknown";
    }
}
}

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

void ORBSLAM3Python::ChangeDataset()
{
    if(system)
    {
        system->ChangeDataset();
    }
}

void ORBSLAM3Python::ResetActiveMap()
{
    if(system)
    {
        system->ResetActiveMap();
    }
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

py::array_t<float> ORBSLAM3Python::GetActiveFramePosesNP() const
{
    auto trajectory = system->GetActiveFramePoses();
    py::ssize_t N = static_cast<py::ssize_t>(trajectory.size());
    py::array_t<float> arr({N, (py::ssize_t)4, (py::ssize_t)4});
    auto buf = arr.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < N; ++i)
    {
        std::memcpy(buf.mutable_data(i, 0, 0), trajectory[i].data(), 16 * sizeof(float));
    }
    return arr;
}

py::list ORBSLAM3Python::GetMapEvents()
{
    py::list events;
    if(!system)
    {
        return events;
    }

    std::vector<ORB_SLAM3::Atlas::MapEvent> nativeEvents = system->PopMapEvents();
    for (const auto& event : nativeEvents)
    {
        py::dict item;
        item["type"] = MapEventTypeToString(event.type);
        item["from_map_id"] = event.fromMapId;
        item["to_map_id"] = event.toMapId;
        item["timestamp"] = event.timestamp;
        item["moved_keyframe_ids"] = py::cast(event.movedKeyframeIds);
        if(event.hasTransform)
        {
            item["transform"] = Array16ToNumpy(event.transform);
        }
        else
        {
            item["transform"] = py::none();
        }
        events.append(item);
    }
    return events;
}

py::list ORBSLAM3Python::PeekMapEvents()
{
    py::list events;
    if(!system)
    {
        return events;
    }

    std::vector<ORB_SLAM3::Atlas::MapEvent> nativeEvents = system->PeekMapEvents();
    for (const auto& event : nativeEvents)
    {
        py::dict item;
        item["type"] = MapEventTypeToString(event.type);
        item["from_map_id"] = event.fromMapId;
        item["to_map_id"] = event.toMapId;
        item["timestamp"] = event.timestamp;
        item["moved_keyframe_ids"] = py::cast(event.movedKeyframeIds);
        if(event.hasTransform)
        {
            item["transform"] = Array16ToNumpy(event.transform);
        }
        else
        {
            item["transform"] = py::none();
        }
        events.append(item);
    }
    return events;
}

bool ORBSLAM3Python::processRGBD_IMU(cv::Mat image, cv::Mat depthImage, double timestamp, py::array_t<float> vImuMeas)
{
    if (!system)
    {
        std::cout << "you must call initialize() first!" << std::endl;
        return false;
    }
    if (sensorMode != ORB_SLAM3::System::IMU_RGBD)
    {
        std::cout << "Error: you called processRGBD_IMU but the system was not initialized in IMU_RGBD mode." << std::endl;
        return false;
    }
    if (!image.data || !depthImage.data)
    {
        std::cout << "Error: image or depth image is empty." << std::endl;
        return false;
    }
    if (vImuMeas.ndim() != 2)
    {
        std::cout << "Error: vImuMeas should be a 2D array." << std::endl;
        return false;
    }
    
    auto values = vImuMeas.unchecked<2>();
    pybind11::ssize_t N = values.shape(0);
    pybind11::ssize_t w = values.shape(1);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeasurements;
    vImuMeasurements.reserve(N);

    if (w != 7)
    {
        std::cout << "IMU measurement should have 7 values [ax, ay, az, wx, wy, wz, t]" << std::endl;
        return false;
    }

    for (pybind11::ssize_t i = 0; i < N; i++)
    {
        float ax = values(i, 0);
        float ay = values(i, 1);
        float az = values(i, 2);
        float wx = values(i, 3);
        float wy = values(i, 4);
        float wz = values(i, 5);
        double t = static_cast<double>(values(i, 6));
        vImuMeasurements.emplace_back(ax, ay, az, wx, wy, wz, t);
    }

    // Print IMU measurements
    // std::cout << "IMU Measurements (" << vImuMeasurements.size() << " total):" << std::endl;
    // for (size_t i = 0; i < vImuMeasurements.size(); i++)
    // {
    //     const auto& imu = vImuMeasurements[i];
    //     std::cout << "  [" << i << "] t=" << imu.t 
    //               << " acc=(" << imu.a(0) << ", " << imu.a(1) << ", " << imu.a(2) << ")"
    //               << " gyro=(" << imu.w(0) << ", " << imu.w(1) << ", " << imu.w(2) << ")"
    //               << std::endl;
    // }

    system->TrackRGBD(image, depthImage, timestamp, vImuMeasurements);
    return !system->isLost();
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
        
        // ---------------------------------------------------------------
        // Custom Functions
        // ---------------------------------------------------------------
        .def("get_num_maps_in_atlas", &ORBSLAM3Python::GetNumMapsInAtlas)
        .def("get_active_map_id", &ORBSLAM3Python::GetActiveMapID)
        .def("change_dataset", &ORBSLAM3Python::ChangeDataset)
        .def("reset_active_map", &ORBSLAM3Python::ResetActiveMap)
        .def("get_all_keyframe_times", &ORBSLAM3Python::GetAllKeyFrameTimes, py::return_value_policy::copy)
        .def("get_all_keyframe_map_ids", &ORBSLAM3Python::GetAllKeyFrameMapIDs, py::return_value_policy::copy)
        .def("get_all_keyframe_poses", &ORBSLAM3Python::GetAllKeyFramePosesNP)
        .def("get_all_keyframe_data", &ORBSLAM3Python::GetAllKeyFrameData)
        .def("get_active_frame_poses", &ORBSLAM3Python::GetActiveFramePosesNP)
        .def("get_map_events", &ORBSLAM3Python::GetMapEvents)
        .def("peek_map_events", &ORBSLAM3Python::PeekMapEvents)
        .def("process_image_rgbd_imu", &ORBSLAM3Python::processRGBD_IMU, py::arg("image"), py::arg("depth"), py::arg("time_stamp"), py::arg("vImuMeas"));
}