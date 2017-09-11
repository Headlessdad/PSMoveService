#ifndef VIRTUAL_STEREO_TRACKER_H
#define VIRTUAL_STEREO_TRACKER_H

// -- includes -----
#include "PSMoveConfig.h"
#include "DeviceEnumerator.h"
#include "DeviceInterface.h"
#include <string>
#include <vector>
#include <array>
#include <deque>

// -- pre-declarations -----
namespace PSMoveProtocol
{
    class Response_ResultTrackerSettings;
};

// -- definitions -----
class VirtualStereoTrackerConfig : public PSMoveConfig
{
public:
    VirtualStereoTrackerConfig(const std::string &fnamebase = "VirtualStereoTrackerConfig");
    
    virtual const boost::property_tree::ptree config2ptree();
    virtual void ptree2config(const boost::property_tree::ptree &pt);

	const CommonHSVColorRangeTable *getColorRangeTable(const std::string &table_name) const;
	inline CommonHSVColorRangeTable *getOrAddColorRangeTable(const std::string &table_name);
    
    bool is_valid;
    long max_poll_failure_count;

	double frame_width;
	double frame_height;
	double frame_rate;
    double exposure;
	double gain;

    double hfov;
    double vfov;
    double zNear;
    double zFar;

    std::string left_camera_usb_path;
    std::string right_camera_usb_path;

    std::array<double, 3*3> left_intrinsic_matrix;
    std::array<double, 3*3> right_intrinsic_matrix;

    std::array<double, 5> left_distortion_cofficients;
    std::array<double, 5> right_distortion_cofficients;

    std::array<double, 3*3> left_rectification_rotation;
    std::array<double, 3*3> right_rectification_rotation;

    std::array<double, 3*4> left_rectification_projection;
    std::array<double, 3*4> right_rectification_projection;

    std::array<double, 3*3> rotation_between_cameras;
    std::array<double, 3> translation_between_cameras;
    std::array<double, 3*3> essential_matrix;
    std::array<double, 3*3> fundamental_matrix;

    CommonDevicePose pose;
	CommonHSVColorRangeTable SharedColorPresets;
	std::vector<CommonHSVColorRangeTable> DeviceColorPresets;

    static const int CONFIG_VERSION;
};

struct VirtualStereoTrackerState : public CommonDeviceState
{   
    VirtualStereoTrackerState()
    {
        clear();
    }
    
    void clear()
    {
        CommonDeviceState::clear();
        DeviceType = CommonDeviceState::VirtualStereoCamera;
    }
};

class VirtualStereoTracker : public ITrackerInterface {
public:
    VirtualStereoTracker();
    virtual ~VirtualStereoTracker();
        
    // Stereo Tracker
    bool open(); // Opens the first virtual stereo tracker
    
    // -- IDeviceInterface
    bool matchesDeviceEnumerator(const DeviceEnumerator *enumerator) const override;
    bool open(const DeviceEnumerator *enumerator) override;
    bool getIsOpen() const override;
    bool getIsReadyToPoll() const override;
    IDeviceInterface::ePollResult poll() override;
    void close() override;
    long getMaxPollFailureCount() const override;
    static CommonDeviceState::eDeviceType getDeviceTypeStatic()
    { return CommonDeviceState::VirtualStereoCamera; }
    CommonDeviceState::eDeviceType getDeviceType() const override;
    const CommonDeviceState *getState(int lookBack = 0) const override;
    
    // -- ITrackerInterface
    ITrackerInterface::eDriverType getDriverType() const override;
    std::string getUSBDevicePath() const override;
    bool getVideoFrameDimensions(int *out_width, int *out_height, int *out_stride) const override;
    bool getIsStereoCamera() const override { return true; }
    const unsigned char *getVideoFrameBuffer(ITrackerInterface::eTrackerVideoSection section) const override;
    void loadSettings() override;
    void saveSettings() override;
	void setFrameWidth(double value, bool bUpdateConfig) override;
	double getFrameWidth() const override;
	void setFrameHeight(double value, bool bUpdateConfig) override;
	double getFrameHeight() const override;
	void setFrameRate(double value, bool bUpdateConfig) override;
	double getFrameRate() const override;
    void setExposure(double value, bool bUpdateConfig) override;
    double getExposure() const override;
	void setGain(double value, bool bUpdateConfig) override;
	double getGain() const override;
    void getCameraIntrinsics(
        ITrackerInterface::eTrackerVideoSection section,
        float &outFocalLengthX, float &outFocalLengthY,
        float &outPrincipalX, float &outPrincipalY,
        float &outDistortionK1, float &outDistortionK2, float &outDistortionK3,
        float &outDistortionP1, float &outDistortionP2) const override;
    void setCameraIntrinsics(
        ITrackerInterface::eTrackerVideoSection section,
        float focalLengthX, float focalLengthY,
        float principalX, float principalY,
        float distortionK1, float distortionK2, float distortionK3,
        float distortionP1, float distortionP2) override;
    CommonDevicePose getTrackerPose() const override;
    void setTrackerPose(const struct CommonDevicePose *pose) override;
    void getFOV(float &outHFOV, float &outVFOV) const override;
    void getZRange(float &outZNear, float &outZFar) const override;
    void gatherTrackerOptions(PSMoveProtocol::Response_ResultTrackerSettings* settings) const override;
    bool setOptionIndex(const std::string &option_name, int option_index) override;
    bool getOptionIndex(const std::string &option_name, int &out_option_index) const override;
    void gatherTrackingColorPresets(const std::string &controller_serial, PSMoveProtocol::Response_ResultTrackerSettings* settings) const override;
    void setTrackingColorPreset(const std::string &controller_serial, eCommonTrackingColorID color, const CommonHSVColorRange *preset) override;
    void getTrackingColorPreset(const std::string &controller_serial, eCommonTrackingColorID color, CommonHSVColorRange *out_preset) const override;

    // -- Getters
    inline const VirtualStereoTrackerConfig &getConfig() const
    { return cfg; }

private:
    VirtualStereoTrackerConfig cfg;
    std::string device_identifier;

    class ITrackerInterface *LeftTracker;
    class ITrackerInterface *RightTracker;
    class VirtualStereoCaptureData *CaptureData;
    ITrackerInterface::eDriverType DriverType;    
    
    // Read Tracker State
    int NextPollSequenceNumber;
    std::deque<VirtualStereoTrackerState> TrackerStates;
};
#endif // PS3EYE_TRACKER_H
