// Gazebo timestamp probe for Scout calibration validation.
// The optional --image-metadata mode subscribes to full Image messages (but
// only records metadata), which can add transport/memory load because the
// image payload is still delivered to this process.

#include <gz/msgs/camera_info.pb.h>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/laserscan.pb.h>
#include <gz/transport/Node.hh>

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{
volatile std::sig_atomic_t stopRequested = 0;

void StopHandler(int)
{
  stopRequested = 1;
}

int64_t MonotonicNanoseconds()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string Sequence(const gz::msgs::Header &header)
{
  for (const auto &entry : header.data())
  {
    if ((entry.key() == "seq" || entry.key() == "sequence") && entry.value_size() > 0)
      return entry.value(0);
  }
  return "";
}

std::string Frame(const gz::msgs::Header &header)
{
  for (const auto &entry : header.data())
  {
    if (entry.key() == "frame_id" && entry.value_size() > 0)
      return entry.value(0);
  }
  return "";
}

std::string Csv(std::string value)
{
  std::string result = "\"";
  for (char c : value)
  {
    if (c == '"') result += '"';
    result += c;
  }
  result += '"';
  return result;
}

class Recorder
{
 public:
  explicit Recorder(const std::string &path) : output(path)
  {
    if (!output)
      throw std::runtime_error("cannot open output CSV: " + path);
    output << "host_monotonic_ns,type,sim_sec,sim_nsec,sequence,frame,width,height,range_count,vertical_count,pixel_format,step,image_data_bytes\n";
  }

  void Write(const std::string &type, int64_t sec, int32_t nsec,
             const std::string &sequence, const std::string &frame,
             uint32_t width = 0, uint32_t height = 0,
             uint32_t rangeCount = 0, uint32_t verticalCount = 0,
             int pixelFormat = 0, uint32_t step = 0,
             uint64_t imageDataBytes = 0)
  {
    const auto mono = MonotonicNanoseconds();
    std::lock_guard<std::mutex> lock(mutex);
    output << mono << ',' << type << ',' << sec << ',' << nsec << ','
           << Csv(sequence) << ',' << Csv(frame) << ',' << width << ','
           << height << ',' << rangeCount << ',' << verticalCount << ','
           << pixelFormat << ',' << step << ',' << imageDataBytes << '\n';
    if (++writes % 30 == 0)
      output.flush();
  }

 private:
  std::ofstream output;
  std::mutex mutex;
  uint64_t writes{0};
};
}

int main(int argc, char **argv)
{
  if (argc < 3 || argc > 5)
  {
    std::cerr << "Usage: " << argv[0]
              << " OUTPUT.csv WORLD_NAME [DURATION_S] [--image-metadata]\n";
    return 2;
  }

  const std::string output = argv[1];
  const std::string world = argv[2];
  const bool captureImageMetadata = argc == 5 &&
      std::string(argv[4]) == "--image-metadata";
  if ((argc == 4 && std::string(argv[3]) == "--image-metadata") ||
      (argc == 5 && !captureImageMetadata))
  {
    std::cerr << "The optional fifth argument must be --image-metadata, after DURATION_S.\n";
    return 2;
  }
  const double duration = argc >= 4 ? std::stod(argv[3]) : 15.0;
  if (duration <= 0.0)
  {
    std::cerr << "Duration must be positive.\n";
    return 2;
  }

  const std::string cameraInfoTopic = "/world/" + world +
      "/model/x500_mono_cam_down_lidar_0/link/camera_link/sensor/imager/camera_info";
  const std::string cameraImageTopic = "/world/" + world +
      "/model/x500_mono_cam_down_lidar_0/link/camera_link/sensor/imager/image";
  const std::string lidarTopic = "/scout/lidar/points";
  const std::string clockTopic = "/world/" + world + "/clock";

  try
  {
    Recorder recorder(output);
    gz::transport::Node node;
    const bool cameraSubscribed = node.Subscribe<gz::msgs::CameraInfo>(cameraInfoTopic,
        std::function<void(const gz::msgs::CameraInfo &)>([&recorder](const gz::msgs::CameraInfo &msg)
        {
          const auto &stamp = msg.header().stamp();
          recorder.Write("camera_info", stamp.sec(), stamp.nsec(),
                         Sequence(msg.header()), Frame(msg.header()),
                         msg.width(), msg.height());
        }));
    const bool imageSubscribed = !captureImageMetadata ||
        node.Subscribe<gz::msgs::Image>(cameraImageTopic,
        std::function<void(const gz::msgs::Image &)>([&recorder](const gz::msgs::Image &msg)
        {
          const auto &stamp = msg.header().stamp();
          recorder.Write("image", stamp.sec(), stamp.nsec(),
                         Sequence(msg.header()), Frame(msg.header()),
                         msg.width(), msg.height(), 0, 0,
                         static_cast<int>(msg.pixel_format_type()), msg.step(),
                         static_cast<uint64_t>(msg.data().size()));
        }));
    const bool lidarSubscribed = node.Subscribe<gz::msgs::LaserScan>(lidarTopic,
        std::function<void(const gz::msgs::LaserScan &)>([&recorder](const gz::msgs::LaserScan &msg)
        {
          const auto &stamp = msg.header().stamp();
          recorder.Write("lidar", stamp.sec(), stamp.nsec(),
                         Sequence(msg.header()), msg.frame(), 0, 0,
                         msg.count(), msg.vertical_count());
        }));
    const bool clockSubscribed = node.Subscribe<gz::msgs::Clock>(clockTopic,
        std::function<void(const gz::msgs::Clock &)>([&recorder](const gz::msgs::Clock &msg)
        {
          const auto &stamp = msg.sim();
          recorder.Write("world_clock", stamp.sec(), stamp.nsec(), "", "");
        }));
    if (!cameraSubscribed || !imageSubscribed || !lidarSubscribed || !clockSubscribed)
    {
      std::cerr << "Failed to subscribe to one or more Gazebo topics.\n"
                << cameraInfoTopic << "\n" << cameraImageTopic << "\n"
                << lidarTopic << "\n" << clockTopic << "\n";
      return 1;
    }

    std::signal(SIGINT, StopHandler);
    std::signal(SIGTERM, StopHandler);
    std::cerr << "Recording low-volume Gazebo stamps for " << duration << " s\n"
              << "camera_info=" << cameraInfoTopic << "\n"
              << "lidar=" << lidarTopic << "\n"
              << "clock=" << clockTopic << "\n"
              << "output=" << output << "\n";
    if (captureImageMetadata)
      std::cerr << "image metadata subscription enabled (image payload is received but not stored): "
                << cameraImageTopic << "\n";
    const auto start = std::chrono::steady_clock::now();
    while (!stopRequested && std::chrono::duration<double>(
               std::chrono::steady_clock::now() - start).count() < duration)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
