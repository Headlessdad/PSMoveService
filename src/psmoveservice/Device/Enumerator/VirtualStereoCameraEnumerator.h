#ifndef VIRTUAL_STEREO_CAMERA_ENUMERATOR_H
#define VIRTUAL_STEREO_CAMERA_ENUMERATOR_H

// -- includes -----
#include "DeviceEnumerator.h"
#include <vector>
#include <string>

// -- definitions -----
class VirtualStereoCameraEnumerator : public DeviceEnumerator
{
public:
    VirtualStereoCameraEnumerator();

    bool is_valid() const override;
    bool next() override;
	int get_vendor_id() const override;
	int get_product_id() const override;
    const char *get_path() const override;

    inline int get_device_identifier() const { return m_device_index; }

    // Assigned by the tracker manager on startup
    static int virtual_stereo_camera_count;

private:
	std::string m_current_device_identifier;
    int m_device_index;
    int m_device_count;
};

#endif // VIRTUAL_STEREO_CAMERA_ENUMERATOR_H

