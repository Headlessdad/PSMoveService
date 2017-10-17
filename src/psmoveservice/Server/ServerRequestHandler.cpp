//-- includes -----
#include "ServerRequestHandler.h"

#include "BluetoothRequests.h"
#include "BluetoothQueries.h"
#include "ControllerManager.h"
#include "DeviceManager.h"
#include "DeviceEnumerator.h"
#include "MathEigen.h"
#include "HMDManager.h"
#include "MorpheusHMD.h"
#include "VirtualHMD.h"
#include "OrientationFilter.h"
#include "PositionFilter.h"
#include "ProtocolVersion.h"
#include "PS3EyeTracker.h"
#include "PSDualShock4Controller.h"
#include "PSMoveController.h"
#include "PSNaviController.h"
#include "PSMoveProtocol.pb.h"
#include "ServerControllerView.h"
#include "ServerDeviceView.h"
#include "ServerNetworkManager.h"
#include "ServerTrackerView.h"
#include "ServerHMDView.h"
#include "ServerLog.h"
#include "ServerUtility.h"
#include "TrackerManager.h"
#include "VirtualController.h"

#include <cassert>
#include <bitset>
#include <map>
#include <boost/shared_ptr.hpp>

//-- pre-declarations -----
class ServerRequestHandlerImpl;
typedef boost::shared_ptr<ServerRequestHandlerImpl> ServerRequestHandlerImplPtr;

//-- definitions -----
struct RequestConnectionState
{
    int connection_id;
    std::bitset<ControllerManager::k_max_devices> active_controller_streams;
    std::bitset<TrackerManager::k_max_devices> active_tracker_streams;
    std::bitset<HMDManager::k_max_devices> active_hmd_streams;
    AsyncBluetoothRequest *pending_bluetooth_request;
    ControllerStreamInfo active_controller_stream_info[ControllerManager::k_max_devices];
    TrackerStreamInfo active_tracker_stream_info[TrackerManager::k_max_devices];
    HMDStreamInfo active_hmd_stream_info[HMDManager::k_max_devices];

    RequestConnectionState()
        : connection_id(-1)
        , active_controller_streams()
        , active_tracker_streams()
        , active_hmd_streams()
        , pending_bluetooth_request(nullptr)
    {
        for (int index = 0; index < ControllerManager::k_max_devices; ++index)
        {
            active_controller_stream_info[index].Clear();
        }

        for (int index = 0; index < TrackerManager::k_max_devices; ++index)
        {
            active_tracker_stream_info[index].Clear();
        }
        
        for (int index = 0; index < HMDManager::k_max_devices; ++index)
        {
            active_hmd_stream_info[index].Clear();
        }
    }
};
typedef boost::shared_ptr<RequestConnectionState> RequestConnectionStatePtr;
typedef std::map<int, RequestConnectionStatePtr> t_connection_state_map;
typedef std::map<int, RequestConnectionStatePtr>::iterator t_connection_state_iter;
typedef std::map<int, RequestConnectionStatePtr>::const_iterator t_connection_state_const_iter;
typedef std::pair<int, RequestConnectionStatePtr> t_id_connection_state_pair;

struct RequestContext
{
    RequestConnectionStatePtr connection_state;
    RequestPtr request;
};

//-- private implementation -----
class ServerRequestHandlerImpl
{
public:
    ServerRequestHandlerImpl(DeviceManager &deviceManager)
        : m_device_manager(deviceManager)
        , m_connection_state_map()
    {
    }

    virtual ~ServerRequestHandlerImpl()
    {
        // Without this we get a warning for deletion:
        // "Delete called on 'class ServerRequestHandlerImpl' that has virtual functions but non-virtual destructor"
    }

    bool any_active_bluetooth_requests() const
    {
        bool any_active= false;

        for (t_connection_state_const_iter iter= m_connection_state_map.begin(); iter != m_connection_state_map.end(); ++iter)
        {
            RequestConnectionStatePtr connection_state= iter->second;

            if (connection_state->pending_bluetooth_request != nullptr)
            {
                any_active= true;
                break;
            }
        }

        return any_active;
    }

    void update()
    {
        for (t_connection_state_iter iter= m_connection_state_map.begin(); iter != m_connection_state_map.end(); ++iter)
        {
            int connection_id= iter->first;
            RequestConnectionStatePtr connection_state= iter->second;

            // Update any asynchronous bluetooth requests
            if (connection_state->pending_bluetooth_request != nullptr)
            {
                bool delete_request;

                connection_state->pending_bluetooth_request->update();

                switch(connection_state->pending_bluetooth_request->getStatusCode())
                {
                case AsyncBluetoothRequest::running:
                    {
                        // Don't delete. Still have work to do
                        delete_request= false;
                    } break;
                case AsyncBluetoothRequest::succeeded:
                    {
                        SERVER_LOG_INFO("ServerRequestHandler") 
                            << "Async bluetooth request(" 
                            << connection_state->pending_bluetooth_request->getDescription() 
                            << ") completed.";
                        delete_request= true;
                    } break;
                case AsyncBluetoothRequest::failed:
                    {
                        SERVER_LOG_ERROR("ServerRequestHandler") 
                            << "Async bluetooth request(" 
                            << connection_state->pending_bluetooth_request->getDescription() 
                            << ") failed!";
                        delete_request= true;
                    } break;
                default:
                    assert(0 && "unreachable");
                }

                if (delete_request)
                {
                    delete connection_state->pending_bluetooth_request;
                    connection_state->pending_bluetooth_request= nullptr;
                }
            }
        }
    }

    ResponsePtr handle_request(int connection_id, RequestPtr request)
    {
        // The context holds everything a handler needs to evaluate a request
        RequestContext context;
        context.request= request;
        context.connection_state= FindOrCreateConnectionState(connection_id);

        // All responses track which request they came from
        PSMoveProtocol::Response *response= nullptr;

        switch (request->type())
        {
            // Controller Requests
            case PSMoveProtocol::Request_RequestType_GET_CONTROLLER_LIST:
                response = new PSMoveProtocol::Response;
                handle_request__get_controller_list(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_START_CONTROLLER_DATA_STREAM:
                response = new PSMoveProtocol::Response;
                handle_request__start_controller_data_stream(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_STOP_CONTROLLER_DATA_STREAM:
                response = new PSMoveProtocol::Response;
                handle_request__stop_controller_data_stream(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_RESET_ORIENTATION:
                response = new PSMoveProtocol::Response;
                handle_request__reset_orientation(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_UNPAIR_CONTROLLER:
                response = new PSMoveProtocol::Response;
                handle_request__unpair_controller(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_PAIR_CONTROLLER:
                response = new PSMoveProtocol::Response;
                handle_request__pair_controller(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_CANCEL_BLUETOOTH_REQUEST:
                response = new PSMoveProtocol::Response;
                handle_request__cancel_bluetooth_request(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_LED_TRACKING_COLOR:
                response = new PSMoveProtocol::Response;
                handle_request__set_led_tracking_color(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_CONTROLLER_MAGNETOMETER_CALIBRATION:
                response = new PSMoveProtocol::Response;
                handle_request__set_controller_magnetometer_calibration(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_CONTROLLER_ACCELEROMETER_CALIBRATION:
                response = new PSMoveProtocol::Response;
                handle_request__set_controller_accelerometer_calibration(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_CONTROLLER_GYROSCOPE_CALIBRATION:
                response = new PSMoveProtocol::Response;
                handle_request__set_controller_gyroscope_calibration(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_OPTICAL_NOISE_CALIBRATION:
                response = new PSMoveProtocol::Response;
                handle_request__set_optical_noise_calibration(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_ORIENTATION_FILTER:
                response = new PSMoveProtocol::Response;
                handle_request__set_orientation_filter(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_POSITION_FILTER:
                response = new PSMoveProtocol::Response;
                handle_request__set_position_filter(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_CONTROLLER_PREDICTION_TIME:
                response = new PSMoveProtocol::Response;
                handle_request__set_controller_prediction_time(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_ATTACHED_CONTROLLER:
                response = new PSMoveProtocol::Response;
                handle_request__set_attached_controller(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_GAMEPAD_INDEX:
                response = new PSMoveProtocol::Response;
                handle_request__set_gamepad_index(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_CONTROLLER_DATA_STREAM_TRACKER_INDEX:
                response = new PSMoveProtocol::Response;
                handle_request__set_controller_data_stream_tracker_index(context, response);
                break;

            // Tracker Requests
            case PSMoveProtocol::Request_RequestType_GET_TRACKER_LIST:
                response = new PSMoveProtocol::Response;
                handle_request__get_tracker_list(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_START_TRACKER_DATA_STREAM:
                response = new PSMoveProtocol::Response;
                handle_request__start_tracker_data_stream(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_STOP_TRACKER_DATA_STREAM:
                response = new PSMoveProtocol::Response;
                handle_request__stop_tracker_data_stream(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_GET_TRACKER_SETTINGS:
                response = new PSMoveProtocol::Response;
                handle_request__get_tracker_settings(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_FRAME_WIDTH:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_frame_width(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_FRAME_HEIGHT:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_frame_height(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_FRAME_RATE:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_frame_rate(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_EXPOSURE:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_exposure(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_GAIN:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_gain(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_OPTION:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_option(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_COLOR_PRESET:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_color_preset(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_POSE:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_pose(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_TRACKER_INTRINSICS:
                response = new PSMoveProtocol::Response;
                handle_request__set_tracker_intrinsics(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SAVE_TRACKER_PROFILE:
                response = new PSMoveProtocol::Response;
                handle_request__save_tracker_profile(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_RELOAD_TRACKER_SETTINGS:
                response = new PSMoveProtocol::Response;
                handle_request__reload_tracker_settings(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_APPLY_TRACKER_PROFILE:
                response = new PSMoveProtocol::Response;
                handle_request__apply_tracker_profile(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SEARCH_FOR_NEW_TRACKERS:
                response = new PSMoveProtocol::Response;
                handle_request__search_for_new_trackers(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_GET_TRACKING_SPACE_SETTINGS:
                response = new PSMoveProtocol::Response;
                handle_request__get_tracking_space_settings(context, response);
                break;

            // HMD Requests
            case PSMoveProtocol::Request_RequestType_GET_HMD_LIST:
                response = new PSMoveProtocol::Response;
                handle_request__get_hmd_list(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_START_HMD_DATA_STREAM:
                response = new PSMoveProtocol::Response;
                handle_request__start_hmd_data_stream(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_STOP_HMD_DATA_STREAM:
                response = new PSMoveProtocol::Response;
                handle_request__stop_hmd_data_stream(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_LED_TRACKING_COLOR:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_led_tracking_color(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_ACCELEROMETER_CALIBRATION:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_accelerometer_calibration(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_GYROSCOPE_CALIBRATION:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_gyroscope_calibration(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_ORIENTATION_FILTER:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_orientation_filter(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_POSITION_FILTER:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_position_filter(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_PREDICTION_TIME:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_prediction_time(context, response);
                break;
            case PSMoveProtocol::Request_RequestType_SET_HMD_DATA_STREAM_TRACKER_INDEX:
                response = new PSMoveProtocol::Response;
                handle_request__set_hmd_data_stream_tracker_index(context, response);
                break;

            // General Service Requests
            case PSMoveProtocol::Request_RequestType_GET_SERVICE_VERSION:
                response = new PSMoveProtocol::Response;
                handle_request__get_service_version(context, response);
                break;

            default:
                assert(0 && "Whoops, bad request!");
        }

        if (response != nullptr)
        {
            response->set_request_id(request->request_id());
        }

        return ResponsePtr(response);
    }

    void handle_input_data_frame(DeviceInputDataFramePtr data_frame)
    {
        // The context holds everything a handler needs to evaluate a request
        RequestConnectionStatePtr connection_state = FindOrCreateConnectionState(data_frame->connection_id());

        switch (data_frame->device_category())
        {
        case PSMoveProtocol::DeviceInputDataFrame::DeviceCategory::DeviceInputDataFrame_DeviceCategory_CONTROLLER:
            {
                handle_data_frame__controller_packet(connection_state, data_frame);
            } break;
        }
    }

    void handle_client_connection_stopped(int connection_id)
    {
        t_connection_state_iter iter= m_connection_state_map.find(connection_id);

        if (iter != m_connection_state_map.end())
        {
            int connection_id= iter->first;
            RequestConnectionStatePtr connection_state= iter->second;

            // Cancel any pending asynchronous bluetooth requests
            if (connection_state->pending_bluetooth_request != nullptr)
            {
                assert(connection_state->pending_bluetooth_request->getStatusCode() == AsyncBluetoothRequest::running);

                connection_state->pending_bluetooth_request->cancel(AsyncBluetoothRequest::connectionClosed);

                delete connection_state->pending_bluetooth_request;
                connection_state->pending_bluetooth_request= nullptr;
            }

            // Clean up any controller state related to this connection
            for (int controller_id = 0; controller_id < ControllerManager::k_max_devices; ++controller_id)
            {
                const ControllerStreamInfo &streamInfo = connection_state->active_controller_stream_info[controller_id];
                ServerControllerViewPtr controller_view = m_device_manager.getControllerViewPtr(controller_id);

                if (controller_view->getIsOpen())
                {
                    // Clear any LED overrides we had active
                    //###HipsterSlot $HACK 
                    // This implicitly assumes that only one connection had an LED override color active.
                    // This is technically true right now because only the config tool sets the override
                    // color for purposes of tracking color calibration, but this could change in the future.
                    if (controller_view->getIsLEDOverrideActive())
                    {
                        controller_view->clearLEDOverride();
                    }

                    // If this connection had ROI disabled, pop the ROI supression request
                    if (streamInfo.disable_roi)
                    {
                        controller_view->popDisableROI();
                    }

                    // Halt any controller tracking this connection had going on
                    if (streamInfo.include_position_data)
                    {
                        m_device_manager.getControllerViewPtr(controller_id)->stopTracking();
                    }
                }
            }

            
            for (int tracker_id = 0; tracker_id < TrackerManager::k_max_devices; ++tracker_id)
            {
                // Restore any overridden camera settings from the config
                if (connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override)
                {
                    m_device_manager.getTrackerViewPtr(tracker_id)->loadSettings();
                }

                // Halt any shared memory streams this connection has going
                if (connection_state->active_tracker_stream_info[tracker_id].streaming_video_data)
                {
                    m_device_manager.getTrackerViewPtr(tracker_id)->stopSharedMemoryVideoStream();
                }
            }

            // Clean up any hmd state related to this connection
            for (int hmd_id = 0; hmd_id < HMDManager::k_max_devices; ++hmd_id)
            {
                const HMDStreamInfo &streamInfo = connection_state->active_hmd_stream_info[hmd_id];
                ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);

                // Undo the ROI suppression
                if (streamInfo.disable_roi)
                {
                    m_device_manager.getHMDViewPtr(hmd_id)->popDisableROI();
                }

                // Halt any hmd tracking this connection had going on
                if (streamInfo.include_position_data)
                {
                    m_device_manager.getHMDViewPtr(hmd_id)->stopTracking();
                }
            }

            // Remove the connection state from the state map
            m_connection_state_map.erase(iter);
        }
    }

    void publish_controller_data_frame(
         ServerControllerView *controller_view, 
         ServerRequestHandler::t_generate_controller_data_frame_for_stream callback)
    {
        int controller_id= controller_view->getDeviceID();

        // Notify any connections that care about the controller update
        for (t_connection_state_iter iter= m_connection_state_map.begin(); iter != m_connection_state_map.end(); ++iter)
        {
            int connection_id= iter->first;
            RequestConnectionStatePtr connection_state= iter->second;

            if (connection_state->active_controller_streams.test(controller_id))
            {
                const ControllerStreamInfo &streamInfo=
                    connection_state->active_controller_stream_info[controller_id];

                // Fill out a data frame specific to this stream using the given callback
                DeviceOutputDataFramePtr data_frame(new PSMoveProtocol::DeviceOutputDataFrame);
                callback(controller_view, &streamInfo, data_frame.get());

                // Send the controller data frame over the network
                ServerNetworkManager::get_instance()->send_device_data_frame(connection_id, data_frame);
            }
        }
    }

    void publish_tracker_data_frame(
        class ServerTrackerView *tracker_view,
            ServerRequestHandler::t_generate_tracker_data_frame_for_stream callback)
    {
        int tracker_id = tracker_view->getDeviceID();

        // Notify any connections that care about the tracker update
        for (t_connection_state_iter iter = m_connection_state_map.begin(); iter != m_connection_state_map.end(); ++iter)
        {
            int connection_id = iter->first;
            RequestConnectionStatePtr connection_state = iter->second;

            if (connection_state->active_tracker_streams.test(tracker_id))
            {
                const TrackerStreamInfo &streamInfo =
                    connection_state->active_tracker_stream_info[tracker_id];

                // Fill out a data frame specific to this stream using the given callback
                DeviceOutputDataFramePtr data_frame(new PSMoveProtocol::DeviceOutputDataFrame);
                callback(tracker_view, &streamInfo, data_frame);

                // Send the tracker data frame over the network
                ServerNetworkManager::get_instance()->send_device_data_frame(connection_id, data_frame);
            }
        }
    }

    void publish_hmd_data_frame(
        class ServerHMDView *hmd_view,
        ServerRequestHandler::t_generate_hmd_data_frame_for_stream callback)
    {
        int hmd_id = hmd_view->getDeviceID();

        // Notify any connections that care about the tracker update
        for (t_connection_state_iter iter = m_connection_state_map.begin(); iter != m_connection_state_map.end(); ++iter)
        {
            int connection_id = iter->first;
            RequestConnectionStatePtr connection_state = iter->second;

            if (connection_state->active_hmd_streams.test(hmd_id))
            {
                const HMDStreamInfo &streamInfo =
                    connection_state->active_hmd_stream_info[hmd_id];

                // Fill out a data frame specific to this stream using the given callback
                DeviceOutputDataFramePtr data_frame(new PSMoveProtocol::DeviceOutputDataFrame);
                callback(hmd_view, &streamInfo, data_frame);

                // Send the hmd data frame over the network
                ServerNetworkManager::get_instance()->send_device_data_frame(connection_id, data_frame);
            }
        }
    }    

protected:
    RequestConnectionStatePtr FindOrCreateConnectionState(int connection_id)
    {
        t_connection_state_iter iter= m_connection_state_map.find(connection_id);
        RequestConnectionStatePtr connection_state;

        if (iter == m_connection_state_map.end())
        {
            connection_state= RequestConnectionStatePtr(new RequestConnectionState());
            connection_state->connection_id= connection_id;

            m_connection_state_map.insert(t_id_connection_state_pair(connection_id, connection_state));
        }
        else
        {
            connection_state= iter->second;
        }

        return connection_state;
    }

    // -- Controller Requests -----
    inline ServerControllerView *get_controller_view_or_null(int controller_id)
    {
        ServerControllerView *controller_view= nullptr;

        if (ServerUtility::is_index_valid(controller_id, m_device_manager.getControllerViewMaxCount()))
        {
            ServerControllerViewPtr controller_view_ptr= m_device_manager.getControllerViewPtr(controller_id);
            
            if (controller_view_ptr->getIsOpen())
            {
                controller_view= controller_view_ptr.get();
            }
        }

        return controller_view;
    }

    void handle_request__get_controller_list(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const PSMoveProtocol::Request_RequestGetControllerList& request =
            context.request->request_get_controller_list();
        PSMoveProtocol::Response_ResultControllerList* list= response->mutable_result_controller_list();

        response->set_type(PSMoveProtocol::Response_ResponseType_CONTROLLER_LIST);

        // Get the address of the bluetooth adapter cached at startup
        list->set_host_serial(m_device_manager.m_controller_manager->getCachedBluetoothHostAddress());
        list->set_gamepad_count(m_device_manager.m_controller_manager->getGamepadCount());

        // Add of the open controllers matching the filter constraints
        for (int controller_id= 0; controller_id < m_device_manager.getControllerViewMaxCount(); ++controller_id)
        {
            ServerControllerViewPtr controller_view= m_device_manager.getControllerViewPtr(controller_id);
            const bool bIncludeUSB = request.include_usb_controllers();
            const bool bIsNonUSB = controller_view->getIsBluetooth() || controller_view->getIsVirtualController();

            if (controller_view->getIsOpen() && (bIncludeUSB || bIsNonUSB))
            {
                PSMoveProtocol::Response_ResultControllerList_ControllerInfo *controller_info= list->add_controllers();

                int firmware_version = 0;
                int firmware_revision = 0;
                bool has_magnetometer = false;

                std::string parent_controller_serial = "";
                std::string orientation_filter = "";
                std::string position_filter = "";
                std::string gyro_gain_setting = "";

                float prediction_time = 0.f;

                int gamepad_index= -1;

                switch(controller_view->getControllerDeviceType())
                {
                case CommonControllerState::PSMove:
                    {
                        const PSMoveController *controller = controller_view->castCheckedConst<PSMoveController>();
                        const PSMoveControllerConfig *config = controller->getConfig();

                        orientation_filter = config->orientation_filter_type;
                        position_filter = config->position_filter_type;
                        firmware_version = config->firmware_version;
                        firmware_revision = config->firmware_revision;
                        prediction_time = config->prediction_time;
                        has_magnetometer = controller->getSupportsMagnetometer();

                        controller_info->set_controller_type(PSMoveProtocol::PSMOVE);
                    }
                    break;
                case CommonControllerState::PSNavi:
                    {
                        const PSNaviController *controller = controller_view->castCheckedConst<PSNaviController>();
                        const PSNaviControllerConfig &config = controller->getConfig();

                        controller_info->set_controller_type(PSMoveProtocol::PSNAVI);
                        parent_controller_serial = config.attached_to_controller;					
                    }
                    break;
                case CommonControllerState::PSDualShock4:
                    {
                        const PSDualShock4Controller *controller = controller_view->castCheckedConst<PSDualShock4Controller>();
                        const PSDualShock4ControllerConfig *config = controller->getConfig();

                        float radian_gain_divisor = safe_divide_with_default(1.f, config->gyro_gain, 1.f);
                        float degree_gain_divisor = radian_gain_divisor * k_degrees_to_radians;

                        // Gyro gain mode can vary from controller to controller
                        // Sensitivity values from Pg.15 of:
                        // https://ae-bst.resource.bosch.com/media/_tech/media/datasheets/BST-BMI055-DS000-08.pdf
                        if (is_nearly_equal(degree_gain_divisor, 262.4f, 1.f))
                        {
                            gyro_gain_setting = "125deg/s";
                        }
                        else if (is_nearly_equal(degree_gain_divisor, 131.2f, 1.f))
                        {
                            gyro_gain_setting = "250deg/s";
                        }
                        else if (is_nearly_equal(degree_gain_divisor, 65.6f, 1.f))
                        {
                            gyro_gain_setting = "500deg/s";
                        }
                        else if (is_nearly_equal(degree_gain_divisor, 32.8f, 1.f))
                        {
                            gyro_gain_setting = "1000deg/s";
                        }
                        else if (is_nearly_equal(degree_gain_divisor, 16.4f, 1.f))
                        {
                            gyro_gain_setting = "2000deg/s";
                        }
                        else
                        {
                            gyro_gain_setting = "custom";
                        }

                        orientation_filter = config->orientation_filter_type;
                        position_filter = config->position_filter_type;
                        prediction_time = config->prediction_time;

                    controller_info->set_controller_type(PSMoveProtocol::PSDUALSHOCK4);
                    }
                    break;
                case CommonControllerState::VirtualController:
                    {
                        const VirtualController *controller = controller_view->castCheckedConst<VirtualController>();
                        const VirtualControllerConfig *config = controller->getConfig();

                        position_filter = config->position_filter_type;
                        prediction_time = config->prediction_time;

                        controller_info->set_controller_type(PSMoveProtocol::VIRTUALCONTROLLER);
                        gamepad_index= config->gamepad_index;
                    }
                    break;
                default:
                    assert(0 && "Unhandled controller type");
                }

                controller_info->set_controller_id(controller_id);
                controller_info->set_connection_type(
                    bIsNonUSB
                    ? PSMoveProtocol::Response_ResultControllerList_ControllerInfo_ConnectionType_BLUETOOTH
                    : PSMoveProtocol::Response_ResultControllerList_ControllerInfo_ConnectionType_USB);  
                controller_info->set_tracking_color_type(
                    static_cast<PSMoveProtocol::TrackingColorType>(controller_view->getTrackingColorID()));
                controller_info->set_device_path(controller_view->getUSBDevicePath());
                controller_info->set_device_serial(controller_view->getSerial());
                controller_info->set_assigned_host_serial(controller_view->getAssignedHostBluetoothAddress());
                controller_info->set_parent_controller_serial(parent_controller_serial);
                controller_info->set_firmware_version(firmware_version);
                controller_info->set_firmware_revision(firmware_revision);
                controller_info->set_has_magnetometer(has_magnetometer);
                controller_info->set_orientation_filter(orientation_filter);
                controller_info->set_position_filter(position_filter);
                controller_info->set_gyro_gain_setting(gyro_gain_setting);
                controller_info->set_prediction_time(prediction_time);
                controller_info->set_gamepad_index(gamepad_index);
            }
        }

        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }

    void handle_request__start_controller_data_stream(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const PSMoveProtocol::Request_RequestStartPSMoveDataStream& request=
            context.request->request_start_psmove_data_stream();
        int controller_id= request.controller_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_CONTROLLER_STREAM_STARTED);

        if (ServerUtility::is_index_valid(controller_id, m_device_manager.getControllerViewMaxCount()))
        {
            ServerControllerViewPtr controller_view = m_device_manager.getControllerViewPtr(controller_id);

            // Some controllers can only be streamed when connected via bluetooth
            if (controller_view->getIsStreamable())
            {
                ControllerStreamInfo &streamInfo =
                    context.connection_state->active_controller_stream_info[controller_id];

                // The controller manager will always publish updates regardless of who is listening.
                // All we have to do is keep track of which connections care about the updates.
                context.connection_state->active_controller_streams.set(controller_id, true);

                // Set control flags for the stream
                streamInfo.Clear();
                streamInfo.include_position_data = request.include_position_data();
                streamInfo.include_physics_data = request.include_physics_data();
                streamInfo.include_raw_sensor_data = request.include_raw_sensor_data();
                streamInfo.include_calibrated_sensor_data = request.include_calibrated_sensor_data();
                streamInfo.include_raw_tracker_data = request.include_raw_tracker_data();
                streamInfo.disable_roi = request.disable_roi();

                SERVER_LOG_INFO("ServerRequestHandler") << "Start controller(" << controller_id << ") stream ("
                    << "pos=" << streamInfo.include_position_data
                    << ",phys=" << streamInfo.include_physics_data
                    << ",raw_sens=" << streamInfo.include_raw_sensor_data
                    << ",cal_sens=" << streamInfo.include_calibrated_sensor_data
                    << ",trkr=" << streamInfo.include_raw_tracker_data
                    << ",roi=" << streamInfo.disable_roi
                    << ")";

                if (streamInfo.include_position_data)
                {
                    controller_view->startTracking();
                }
                
                // Attach the initial state of the controller
                {
                    auto *stream_started_response= response->mutable_result_controller_stream_started();
                    PSMoveProtocol::DeviceOutputDataFrame* data_frame= stream_started_response->mutable_initial_data_frame();
                    
                    ServerControllerView::generate_controller_data_frame_for_stream(controller_view.get(), &streamInfo, data_frame);
                }

                if (streamInfo.disable_roi)
                {
                    controller_view->pushDisableROI();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                SERVER_LOG_INFO("ServerRequestHandler") << "Failed to start controller(" << controller_id << ") stream: Not on stream-able connection.";

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            SERVER_LOG_INFO("ServerRequestHandler") << "Failed to start controller(" << controller_id << ") stream: Invalid controller id.";
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__stop_controller_data_stream(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        int controller_id= context.request->request_stop_psmove_data_stream().controller_id();

        if (ServerUtility::is_index_valid(controller_id, m_device_manager.getControllerViewMaxCount()))
        {
            ServerControllerViewPtr controller_view = m_device_manager.getControllerViewPtr(controller_id);
            ControllerStreamInfo &streamInfo =
                context.connection_state->active_controller_stream_info[controller_id];

            if (controller_view->getIsStreamable())
            {
                if (streamInfo.disable_roi)
                {
                    controller_view->popDisableROI();
                }

                if (streamInfo.include_position_data)
                {
                    controller_view->stopTracking();
                }

                if (streamInfo.led_override_active)
                {
                    controller_view->clearLEDOverride();
                }

                SERVER_LOG_INFO("ServerRequestHandler") << "Stop controller(" << controller_id << ") stream";

                context.connection_state->active_controller_streams.set(controller_id, false);
                context.connection_state->active_controller_stream_info[controller_id].Clear();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__reset_orientation(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const int controller_id= context.request->reset_orientation().controller_id();
        ServerControllerViewPtr controllerView = m_device_manager.getControllerViewPtr(controller_id);

        CommonDeviceQuaternion q_pose;
        q_pose.w= context.request->reset_orientation().orientation().w();
        q_pose.x= context.request->reset_orientation().orientation().x();
        q_pose.y= context.request->reset_orientation().orientation().y();
        q_pose.z= context.request->reset_orientation().orientation().z();

        if (controllerView->getIsOpen())
        {
            if (controllerView->recenterOrientation(q_pose))
            {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__unpair_controller(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const int connection_id= context.connection_state->connection_id;
        const int controller_id= context.request->unpair_controller().controller_id();        

        if (context.connection_state->pending_bluetooth_request == nullptr)
        {
            ServerControllerViewPtr controllerView= m_device_manager.getControllerViewPtr(controller_id);

            if (controllerView->getIsOpen())
            {
            context.connection_state->pending_bluetooth_request =
                new AsyncBluetoothUnpairDeviceRequest(connection_id, controllerView);

            std::string description = context.connection_state->pending_bluetooth_request->getDescription();

            if (context.connection_state->pending_bluetooth_request->start())
            {
                SERVER_LOG_INFO("ServerRequestHandler") << "Async bluetooth request(" << description << ") started.";

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                SERVER_LOG_ERROR("ServerRequestHandler") << "Async bluetooth request(" << description << ") failed to start!";

                delete context.connection_state->pending_bluetooth_request;
                context.connection_state->pending_bluetooth_request = nullptr;

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            SERVER_LOG_ERROR("ServerRequestHandler") 
                    << "Can't start unpair request. Controller not open. Controller ID: "
                    << controller_id;
            }
        }
        else
        {
            SERVER_LOG_ERROR("ServerRequestHandler") 
                << "Can't start unpair request due to existing request: " 
                << context.connection_state->pending_bluetooth_request->getDescription();

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__pair_controller(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const int connection_id= context.connection_state->connection_id;
        const int controller_id= context.request->pair_controller().controller_id();        

        if (context.connection_state->pending_bluetooth_request == nullptr)
        {
            ServerControllerViewPtr controllerView= m_device_manager.getControllerViewPtr(controller_id);

            if (controllerView->getIsOpen())
            {
            context.connection_state->pending_bluetooth_request = 
                new AsyncBluetoothPairDeviceRequest(connection_id, controllerView);

            if (context.connection_state->pending_bluetooth_request->start())
            {
                SERVER_LOG_INFO("ServerRequestHandler") 
                    << "Async bluetooth request(" 
                    << context.connection_state->pending_bluetooth_request->getDescription() 
                    << ") started.";

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                SERVER_LOG_ERROR("ServerRequestHandler") 
                    << "Async bluetooth request(" 
                    << context.connection_state->pending_bluetooth_request->getDescription() 
                    << ") failed to start!";

                delete context.connection_state->pending_bluetooth_request;
                context.connection_state->pending_bluetooth_request= nullptr;

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            SERVER_LOG_ERROR("ServerRequestHandler") 
                    << "Can't start pair request. Controller not open. Controller ID: "
                    << controller_id;
            }
        }
        else
        {
            SERVER_LOG_ERROR("ServerRequestHandler") 
                << "Can't start pair request due to existing request: " 
                << context.connection_state->pending_bluetooth_request->getDescription();

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__cancel_bluetooth_request(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const int connection_id= context.connection_state->connection_id;
        const int controller_id= context.request->cancel_bluetooth_request().controller_id();        

        if (context.connection_state->pending_bluetooth_request != nullptr)
        {
            SERVER_LOG_INFO("ServerRequestHandler") 
                << "Async bluetooth request(" 
                << context.connection_state->pending_bluetooth_request->getDescription() 
                << ") Canceled.";

            context.connection_state->pending_bluetooth_request->cancel(AsyncBluetoothRequest::userRequested);
            
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
        }
        else
        {
            SERVER_LOG_ERROR("ServerRequestHandler") << "No active bluetooth operation active";

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_led_tracking_color(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int connection_id = context.connection_state->connection_id;
        const int controller_id = context.request->set_led_tracking_color_request().controller_id();
        const eCommonTrackingColorID newColorID=
            static_cast<eCommonTrackingColorID>(context.request->set_led_tracking_color_request().color_type());

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);

        if (ControllerView && 
            ControllerView->getIsStreamable() &&
            (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove ||
             ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4 ||
             ControllerView->getControllerDeviceType() == CommonDeviceState::VirtualController))
        {
            const eCommonTrackingColorID oldColorID = ControllerView->getTrackingColorID();

            if (newColorID != oldColorID)
            {
                // Give up control of our existing tracking color
                if (oldColorID != eCommonTrackingColorID::INVALID_COLOR)
                {
                    m_device_manager.m_tracker_manager->freeTrackingColorID(oldColorID);
                }

                // Take the color from any other controller that might have it
                if (m_device_manager.m_tracker_manager->claimTrackingColorID(ControllerView.get(), newColorID))
                {
                    // Assign the new color to ourselves
                    ControllerView->setTrackingColorID(newColorID);
                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
                }
                else 
                {
                    if (oldColorID != eCommonTrackingColorID::INVALID_COLOR)
                    {
                        m_device_manager.m_tracker_manager->claimTrackingColorID(ControllerView.get(), oldColorID);
                    }

                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
                }
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    inline void set_config_vector(
        const PSMoveProtocol::FloatVector &source_vector,
        CommonDeviceVector &target_vector)
    {
        target_vector.i = source_vector.i();
        target_vector.j = source_vector.j();
        target_vector.k = source_vector.k();
    }

    void handle_request__set_controller_magnetometer_calibration(
        const RequestContext &context, 
        PSMoveProtocol::Response *response)
    {
        const int controller_id= context.request->set_controller_magnetometer_calibration_request().controller_id();

        ServerControllerViewPtr ControllerView= m_device_manager.getControllerViewPtr(controller_id);

        if (ControllerView && ControllerView->getIsOpen() && ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
        {
            PSMoveController *controller= ControllerView->castChecked<PSMoveController>();
            PSMoveControllerConfig *config= controller->getConfigMutable();

            const auto &request= context.request->set_controller_magnetometer_calibration_request();

            set_config_vector(request.ellipse_center(), config->magnetometer_center);
            set_config_vector(request.ellipse_extents(), config->magnetometer_extents);
            set_config_vector(request.magnetometer_identity(), config->magnetometer_identity);
            config->magnetometer_fit_error = request.ellipse_fit_error();
            config->magnetometer_variance= request.magnetometer_variance();

            {
                CommonDeviceVector basis_x, basis_y, basis_z;

                set_config_vector(request.ellipse_basis_x(), basis_x);
                set_config_vector(request.ellipse_basis_y(), basis_y);
                set_config_vector(request.ellipse_basis_z(), basis_z);

                config->magnetometer_basis_x = basis_x;
                config->magnetometer_basis_y = basis_y;
                config->magnetometer_basis_z = basis_z;
            }

            config->save();

            // Reset the orientation filter state the calibration changed
            ControllerView->resetPoseFilter();

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_controller_accelerometer_calibration(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->set_controller_accelerometer_calibration_request().controller_id();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);

        if (ControllerView && ControllerView->getIsOpen())
        {
            if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
            {
                PSMoveController *controller = ControllerView->castChecked<PSMoveController>();
                PSMoveControllerConfig *config = controller->getConfigMutable();

                const auto &request = context.request->set_controller_accelerometer_calibration_request();

                // Save the noise radius in controller config
                config->accelerometer_noise_radius= request.noise_radius();
                    config->accelerometer_variance = request.variance();
                config->save();

                    ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4)
            {
                PSDualShock4Controller *controller = ControllerView->castChecked<PSDualShock4Controller>();
                PSDualShock4ControllerConfig *config = controller->getConfigMutable();

                const auto &request = context.request->set_controller_accelerometer_calibration_request();

                // Save the noise radius in controller config
                config->accelerometer_noise_radius= request.noise_radius();
                    config->accelerometer_variance = request.variance();
                config->save();

                    ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_controller_gyroscope_calibration(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->set_controller_gyroscope_calibration_request().controller_id();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);

        if (ControllerView && ControllerView->getIsOpen())
        {
            if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4)
            {
                PSDualShock4Controller *controller = ControllerView->castChecked<PSDualShock4Controller>();
                PSDualShock4ControllerConfig *config = controller->getConfigMutable();

                const auto &request = context.request->set_controller_gyroscope_calibration_request();

                bool bChanged = false;

                if (request.drift() > 0.f)
                {
                    config->gyro_drift= request.drift();
                    bChanged = true;
                }

                if (request.variance() > 0.f)
                {
                    config->gyro_variance= request.variance();
                    bChanged = true;
                }

                const std::string gyro_gain_setting = request.gyro_gain_setting();
                if (gyro_gain_setting.length() > 0)
                {
                    // Sensitivity values from Pg.15 of:
                    // https://ae-bst.resource.bosch.com/media/_tech/media/datasheets/BST-BMI055-DS000-08.pdf
                    if (gyro_gain_setting == "125deg/s")
                    {
                        config->gyro_gain = 1.f / (262.4f / k_degrees_to_radians);
                        bChanged = true;
                    }
                    if (gyro_gain_setting == "250deg/s")
                    {
                        config->gyro_gain = 1.f / (131.2f / k_degrees_to_radians);
                        bChanged = true;
                    }
                    if (gyro_gain_setting == "500deg/s")
                    {
                        config->gyro_gain = 1.f / (65.6f / k_degrees_to_radians);
                        bChanged = true;
                    }
                    else if (gyro_gain_setting == "1000deg/s")
                    {
                        config->gyro_gain = 1.f / (32.8f / k_degrees_to_radians);
                        bChanged = true;
                    }
                    else if (gyro_gain_setting == "2000deg/s")
                    {
                        config->gyro_gain = 1.f / (16.4f / k_degrees_to_radians);
                        bChanged = true;
                    }
                }

                if (bChanged)
                {
                    config->save();
                }

                ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
            {
                PSMoveController *controller = ControllerView->castChecked<PSMoveController>();
                PSMoveControllerConfig *config = controller->getConfigMutable();

                const PSMoveProtocol::Request_RequestSetControllerGyroscopeCalibration &request =
                    context.request->set_controller_gyroscope_calibration_request();

                bool bChanged = false;

                if (request.drift() > 0.f)
                {
                    config->gyro_drift= request.drift();
                    bChanged = true;
                }

                if (request.variance() > 0.f)
                {
                    config->gyro_variance= request.variance();
                    bChanged = true;
                }

                if (bChanged)
                {
                    config->save();
                }

                ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_optical_noise_calibration(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->request_set_optical_noise_calibration().controller_id();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);
        const PSMoveProtocol::Request_RequestSetOpticalNoiseCalibration &request =
            context.request->request_set_optical_noise_calibration();

        if (ControllerView && ControllerView->getIsOpen())
        {
            if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4)
            {
                PSDualShock4Controller *controller = ControllerView->castChecked<PSDualShock4Controller>();
                PSDualShock4ControllerConfig *config = controller->getConfigMutable();

                config->position_variance_exp_fit_a = request.position_variance_exp_fit_a();
                config->position_variance_exp_fit_b = request.position_variance_exp_fit_b();
                config->orientation_variance_exp_fit_a = request.orientation_variance_exp_fit_a();
                config->orientation_variance_exp_fit_b = request.orientation_variance_exp_fit_b();
                config->save();

                ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
            {
                PSMoveController *controller = ControllerView->castChecked<PSMoveController>();
                PSMoveControllerConfig *config = controller->getConfigMutable();

                config->position_variance_exp_fit_a = request.position_variance_exp_fit_a();
                config->position_variance_exp_fit_b = request.position_variance_exp_fit_b();
                // No optical variance set for the psmove
                config->save();

                ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::VirtualController)
            {
                VirtualController *controller = ControllerView->castChecked<VirtualController>();
                VirtualControllerConfig *config = controller->getConfigMutable();

                config->position_variance_exp_fit_a = request.position_variance_exp_fit_a();
                config->position_variance_exp_fit_b = request.position_variance_exp_fit_b();
                // No optical variance set for the psmove
                config->save();

                ControllerView->resetPoseFilter();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_orientation_filter(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->request_set_orientation_filter().controller_id();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);
        const PSMoveProtocol::Request_RequestSetOrientationFilter &request =
            context.request->request_set_orientation_filter();

        if (ControllerView && ControllerView->getIsOpen())
        {
            if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4)
            {
                PSDualShock4Controller *controller = ControllerView->castChecked<PSDualShock4Controller>();
                PSDualShock4ControllerConfig *config = controller->getConfigMutable();

                if (config->orientation_filter_type != request.orientation_filter())
                {
                    config->orientation_filter_type = request.orientation_filter();
                    config->save();

                    ControllerView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
            {
                PSMoveController *controller = ControllerView->castChecked<PSMoveController>();
                PSMoveControllerConfig *config = controller->getConfigMutable();

                if (config->orientation_filter_type != request.orientation_filter())
                {
                    config->orientation_filter_type = request.orientation_filter();
                    config->save();

                    ControllerView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_position_filter(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->request_set_position_filter().controller_id();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);
        const PSMoveProtocol::Request_RequestSetPositionFilter &request =
            context.request->request_set_position_filter();

        if (ControllerView && ControllerView->getIsOpen())
        {
            if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4)
            {
                PSDualShock4Controller *controller = ControllerView->castChecked<PSDualShock4Controller>();
                PSDualShock4ControllerConfig *config = controller->getConfigMutable();

                if (config->position_filter_type != request.position_filter())
                {
                    config->position_filter_type = request.position_filter();
                    config->save();

                    ControllerView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
            {
                PSMoveController *controller = ControllerView->castChecked<PSMoveController>();
                PSMoveControllerConfig *config = controller->getConfigMutable();

                if (config->position_filter_type != request.position_filter())
                {
                    config->position_filter_type = request.position_filter();
                    config->save();

                    ControllerView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::VirtualController)
            {
                VirtualController *controller = ControllerView->castChecked<VirtualController>();
                VirtualControllerConfig *config = controller->getConfigMutable();

                if (config->position_filter_type != request.position_filter())
                {
                    config->position_filter_type = request.position_filter();
                    config->save();

                    ControllerView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_controller_prediction_time(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->request_set_controller_prediction_time().controller_id();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);
        const PSMoveProtocol::Request_RequestSetControllerPredictionTime &request =
            context.request->request_set_controller_prediction_time();

        if (ControllerView && ControllerView->getIsOpen())
        {
            if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSDualShock4)
            {
                PSDualShock4Controller *controller = ControllerView->castChecked<PSDualShock4Controller>();
                PSDualShock4ControllerConfig *config = controller->getConfigMutable();

                if (config->prediction_time != request.prediction_time())
                {
                    config->prediction_time = request.prediction_time();
                    config->save();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::PSMove)
            {
                PSMoveController *controller = ControllerView->castChecked<PSMoveController>();
                PSMoveControllerConfig *config = controller->getConfigMutable();

                if (config->prediction_time != request.prediction_time())
                {
                    config->prediction_time = request.prediction_time();
                    config->save();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (ControllerView->getControllerDeviceType() == CommonDeviceState::VirtualController)
            {
                VirtualController *controller = ControllerView->castChecked<VirtualController>();
                VirtualControllerConfig *config = controller->getConfigMutable();

                if (config->prediction_time != request.prediction_time())
                {
                    config->prediction_time = request.prediction_time();
                    config->save();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_attached_controller(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int child_controller_id = context.request->request_set_attached_controller().child_controller_id();
        const int parent_controller_id = context.request->request_set_attached_controller().parent_controller_id();

        ServerControllerViewPtr ChildControllerView = m_device_manager.getControllerViewPtr(child_controller_id);
        ServerControllerViewPtr ParentControllerView = m_device_manager.getControllerViewPtr(parent_controller_id);

        if (ChildControllerView && ChildControllerView->getIsOpen() &&
            ParentControllerView && ParentControllerView->getIsOpen() &&
            ChildControllerView != ParentControllerView)
        {
            if (ParentControllerView->getControllerDeviceType() == CommonDeviceState::PSMove && 
                ChildControllerView->getControllerDeviceType() == CommonDeviceState::PSNavi)
            {
                const PSMoveController *psmove = ParentControllerView->castChecked<PSMoveController>();

                PSNaviController *psnavi = ChildControllerView->castChecked<PSNaviController>();
                PSNaviControllerConfig &psnavi_config = psnavi->getConfigMutable();

                psnavi_config.attached_to_controller = psmove->getSerial();
                psnavi_config.save();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_gamepad_index(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->request_set_gamepad_index().controller_id();
        const int gamepad_index = context.request->request_set_gamepad_index().gamepad_index();

        ServerControllerViewPtr ControllerView = m_device_manager.getControllerViewPtr(controller_id);

        if (ControllerView && 
            ControllerView->getIsOpen() &&
            ControllerView->getControllerDeviceType() == CommonDeviceState::VirtualController)
        {
            const int gamepadCount= DeviceManager::getInstance()->m_controller_manager->getGamepadCount();

            if (gamepad_index >= -1 && gamepad_index < gamepadCount)
            {
                VirtualController *virtual_controller = ControllerView->castChecked<VirtualController>();
                VirtualControllerConfig *config = virtual_controller->getConfigMutable();

                config->gamepad_index = gamepad_index;
                config->save();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_controller_data_stream_tracker_index(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int controller_id = context.request->request_set_controller_data_stream_tracker_index().controller_id();
        const int tracker_id = context.request->request_set_controller_data_stream_tracker_index().tracker_id();

        if (ServerUtility::is_index_valid(controller_id, m_device_manager.getControllerViewMaxCount()) &&
            ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerControllerViewPtr controller_view = m_device_manager.getControllerViewPtr(controller_id);
            ControllerStreamInfo &streamInfo =
                context.connection_state->active_controller_stream_info[controller_id];

            if (controller_view->getIsStreamable())
            {
                SERVER_LOG_INFO("ServerRequestHandler") << "Set controller(" << controller_id << ") stream tracker id: " << tracker_id;

                streamInfo.selected_tracker_index= tracker_id;

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    // -- tracker requests -----
    inline void common_device_pose_to_protocol_pose(
        const CommonDevicePose &pose, 
        PSMoveProtocol::Pose *result)
    {
        PSMoveProtocol::Orientation *orietation = result->mutable_orientation();
        PSMoveProtocol::Position *position = result->mutable_position();

        orietation->set_w(pose.Orientation.w);
        orietation->set_x(pose.Orientation.x);
        orietation->set_y(pose.Orientation.y);
        orietation->set_z(pose.Orientation.z);

        position->set_x(pose.PositionCm.x);
        position->set_y(pose.PositionCm.y);
        position->set_z(pose.PositionCm.z);
    }

    inline void common_distortion_to_protocol_distortion(
        const CommonDistortionCoefficients &d,
        PSMoveProtocol::TrackerIntrinsics_DistortionCoefficients *result)
    {
        result->set_k1(d.k1);
        result->set_k3(d.k2);
        result->set_k3(d.k3);
        result->set_p1(d.p1);
        result->set_p2(d.p2);
    }

    inline void common_vec3_to_protocol_vec3(
        const std::array<double, 3> &v,
        PSMoveProtocol::DoubleVector *result)
    {
        result->set_i(v[0]);
        result->set_j(v[1]);
        result->set_k(v[2]);
    }

    inline void common_mat33_to_protocol_mat33(
        const std::array<double, 9> &m,
        PSMoveProtocol::DoubleMatrix33 *result)
    {
        result->set_m00(m[0]); result->set_m01(m[1]); result->set_m02(m[2]);
        result->set_m10(m[3]); result->set_m11(m[4]); result->set_m12(m[5]);
        result->set_m20(m[6]); result->set_m21(m[7]); result->set_m22(m[8]);
    }

    inline void common_mat34_to_protocol_mat34(
        const std::array<double, 12> &m,
        PSMoveProtocol::DoubleMatrix34 *result)
    {
        result->set_m00(m[0]); result->set_m01(m[1]); result->set_m02(m[2]); result->set_m03(m[3]);
        result->set_m10(m[4]); result->set_m11(m[5]); result->set_m12(m[6]); result->set_m13(m[7]);
        result->set_m20(m[8]); result->set_m21(m[9]); result->set_m22(m[10]); result->set_m23(m[11]);
    }

    inline void common_mat44_to_protocol_mat44(
        const std::array<double, 16> &m,
        PSMoveProtocol::DoubleMatrix44 *result)
    {
        result->set_m00(m[0]); result->set_m01(m[1]); result->set_m02(m[2]); result->set_m03(m[3]);
        result->set_m10(m[4]); result->set_m11(m[5]); result->set_m12(m[6]); result->set_m13(m[7]);
        result->set_m20(m[8]); result->set_m21(m[9]); result->set_m22(m[10]); result->set_m23(m[11]);
        result->set_m30(m[12]); result->set_m31(m[13]); result->set_m32(m[14]); result->set_m33(m[15]);
    }

    void handle_request__get_tracker_list(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        PSMoveProtocol::Response_ResultTrackerList* list = response->mutable_result_tracker_list();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_LIST);

        for (int tracker_id = 0; tracker_id < m_device_manager.getTrackerViewMaxCount(); ++tracker_id)
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);

            if (tracker_view->getIsOpen())
            {
                PSMoveProtocol::Response_ResultTrackerList_TrackerInfo *tracker_info = list->add_trackers();

                switch (tracker_view->getTrackerDeviceType())
                {
                case CommonDeviceState::PS3EYE:
                    tracker_info->set_tracker_type(PSMoveProtocol::PS3EYE);
                    break;
                case CommonDeviceState::VirtualStereoCamera:
                    tracker_info->set_tracker_type(PSMoveProtocol::VirtualStereoCamera);
                    break;
                default:
                    assert(0 && "Unhandled tracker type");
                }

                switch (tracker_view->getTrackerDriverType())
                {
                case ITrackerInterface::Libusb:
                    tracker_info->set_tracker_driver(PSMoveProtocol::LIBUSB);
                    break;
                case ITrackerInterface::CL:
                    tracker_info->set_tracker_driver(PSMoveProtocol::CL_EYE);
                    break;
                case ITrackerInterface::CLMulti:
                    tracker_info->set_tracker_driver(PSMoveProtocol::CL_EYE_MULTICAM);
                    break;
                    // PSMoveProtocol::ISIGHT?
                case ITrackerInterface::Generic_Webcam:
                    tracker_info->set_tracker_driver(PSMoveProtocol::GENERIC_WEBCAM);
                    break;
                default:
                    assert(0 && "Unhandled tracker type");
                }

                tracker_info->set_tracker_id(tracker_id);
                tracker_info->set_device_path(tracker_view->getUSBDevicePath());
                tracker_info->set_shared_memory_name(tracker_view->getSharedMemoryStreamName());

                // Get the intrinsic camera lens properties
                {
                    CommonTrackerIntrinsics tracker_intrinsics;

                    tracker_view->getCameraIntrinsics(tracker_intrinsics);

                    switch (tracker_intrinsics.intrinsics_type)
                    {
                    case CommonTrackerIntrinsics::MONO_TRACKER_INTRINSICS:
                        {                            
                            const CommonMonoTrackerIntrinsics &src_intrinsics= tracker_intrinsics.mono_intrinsics;
                            auto* dest_intrinsics= tracker_info->mutable_tracker_intrinsics()->mutable_mono_intrinsics();

                            dest_intrinsics->mutable_tracker_screen_dimensions()->set_x(src_intrinsics.pixel_width);
                            dest_intrinsics->mutable_tracker_screen_dimensions()->set_y(src_intrinsics.pixel_height);
                            dest_intrinsics->set_hfov(src_intrinsics.hfov);
                            dest_intrinsics->set_vfov(src_intrinsics.vfov);
                            dest_intrinsics->set_znear(src_intrinsics.znear);
                            dest_intrinsics->set_zfar(src_intrinsics.zfar);
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.camera_matrix, 
                                dest_intrinsics->mutable_camera_matrix());
                            common_distortion_to_protocol_distortion(
                                src_intrinsics.distortion_coefficients,
                                dest_intrinsics->mutable_distortion_coefficients());
                        } break;
                    case CommonTrackerIntrinsics::STEREO_TRACKER_INTRINSICS:
                        {
                            const CommonStereoTrackerIntrinsics &src_intrinsics= tracker_intrinsics.stereo_intrinsics;
                            auto* dest_intrinsics= tracker_info->mutable_tracker_intrinsics()->mutable_stereo_intrinsics();

                            dest_intrinsics->mutable_tracker_screen_dimensions()->set_x(src_intrinsics.pixel_width);
                            dest_intrinsics->mutable_tracker_screen_dimensions()->set_y(src_intrinsics.pixel_height);
                            dest_intrinsics->set_hfov(src_intrinsics.hfov);
                            dest_intrinsics->set_vfov(src_intrinsics.vfov);
                            dest_intrinsics->set_znear(src_intrinsics.znear);
                            dest_intrinsics->set_zfar(src_intrinsics.zfar);
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.left_camera_matrix, 
                                dest_intrinsics->mutable_left_camera_matrix());
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.right_camera_matrix, 
                                dest_intrinsics->mutable_right_camera_matrix());
                            common_distortion_to_protocol_distortion(
                                src_intrinsics.left_distortion_coefficients,
                                dest_intrinsics->mutable_left_distortion_coefficients());
                            common_distortion_to_protocol_distortion(
                                src_intrinsics.right_distortion_coefficients,
                                dest_intrinsics->mutable_right_distortion_coefficients());
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.left_rectification_rotation, 
                                dest_intrinsics->mutable_left_rectification_rotation());
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.right_rectification_rotation, 
                                dest_intrinsics->mutable_right_rectification_rotation());
                            common_mat34_to_protocol_mat34(
                                src_intrinsics.left_rectification_projection, 
                                dest_intrinsics->mutable_left_rectification_projection());
                            common_mat34_to_protocol_mat34(
                                src_intrinsics.right_rectification_projection, 
                                dest_intrinsics->mutable_right_rectification_projection());
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.rotation_between_cameras, 
                                dest_intrinsics->mutable_rotation_between_cameras());
                            common_vec3_to_protocol_vec3(
                                src_intrinsics.translation_between_cameras, 
                                dest_intrinsics->mutable_translation_between_cameras());
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.essential_matrix, 
                                dest_intrinsics->mutable_essential_matrix());
                            common_mat33_to_protocol_mat33(
                                src_intrinsics.fundamental_matrix, 
                                dest_intrinsics->mutable_fundamental_matrix());
                            common_mat44_to_protocol_mat44(
                                src_intrinsics.reprojection_matrix,
                                dest_intrinsics->mutable_reprojection_matrix());
                        } break;
                    default:
                        assert(0 && "Unhandled intrinsics type");
                        break;
                    }
                }

                // Get the tracker pose
                {
                    CommonDevicePose pose= tracker_view->getTrackerPose();

                    common_device_pose_to_protocol_pose(pose, tracker_info->mutable_tracker_pose());
                }                
            }
        }

        list->set_global_forward_degrees(m_device_manager.m_tracker_manager->getConfig().global_forward_degrees);
        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }

    void handle_request__start_tracker_data_stream(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const PSMoveProtocol::Request_RequestStartTrackerDataStream& request =
            context.request->request_start_tracker_data_stream();
        int tracker_id = request.tracker_id();

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);

            if (tracker_view->getIsOpen())
            {
                TrackerStreamInfo &streamInfo =
                    context.connection_state->active_tracker_stream_info[tracker_id];

                // The tracker manager will always publish updates regardless of who is listening.
                // All we have to do is keep track of which connections care about the updates.
                context.connection_state->active_tracker_streams.set(tracker_id, true);

                // Set control flags for the stream
                streamInfo.streaming_video_data = true;

                // Increment the number of stream listeners
                tracker_view->startSharedMemoryVideoStream();

                // Return the name of the shared memory block the video frames will be written to
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                // Device not opened
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__stop_tracker_data_stream(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        int tracker_id = context.request->request_stop_tracker_data_stream().tracker_id();

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);

            if (tracker_view->getIsOpen())
            {
                context.connection_state->active_tracker_streams.set(tracker_id, false);
                context.connection_state->active_tracker_stream_info[tracker_id].Clear();

                // Restore any overridden camera settings from the config
                if (context.connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override)
                {
                    tracker_view->loadSettings();
                }

                // Decrement the number of stream listeners
                tracker_view->stopSharedMemoryVideoStream();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                // Device not opened
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__get_tracker_settings(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_get_tracker_settings().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_SETTINGS);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                PSMoveProtocol::Response_ResultTrackerSettings* settings =
                    response->mutable_result_tracker_settings();
                const int device_id = context.request->request_get_tracker_settings().device_id();

                settings->set_frame_width(static_cast<float>(tracker_view->getFrameWidth()));
                settings->set_frame_height(static_cast<float>(tracker_view->getFrameHeight()));
                settings->set_frame_rate(static_cast<float>(tracker_view->getFrameRate()));
                settings->set_exposure(static_cast<float>(tracker_view->getExposure()));
                settings->set_gain(static_cast<float>(tracker_view->getGain()));
                tracker_view->gatherTrackerOptions(settings);

                switch (context.request->request_get_tracker_settings().device_category())
                {
                case PSMoveProtocol::Request_RequestGetTrackerSettings_DeviceCategory_CONTROLLER:
                    {
                        ServerControllerView *controller_view = get_controller_view_or_null(device_id);

                tracker_view->gatherTrackingColorPresets(controller_view, settings);
                    } break;
                case PSMoveProtocol::Request_RequestGetTrackerSettings_DeviceCategory_HMD:
                    {
                        ServerHMDView *hmd_view = get_hmd_view_or_null(device_id);

                        tracker_view->gatherTrackingColorPresets(hmd_view, settings);
                    } break;
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_frame_width(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_frame_width().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_FRAME_WIDTH_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const bool bSaveSetting = context.request->request_set_tracker_frame_width().save_setting();
                const float desired_frame_width = context.request->request_set_tracker_frame_width().value();
                PSMoveProtocol::Response_ResultSetTrackerFrameWidth* result_frame_width =
                    response->mutable_result_set_tracker_frame_width();

                // Set the desired frame width on the tracker
                tracker_view->setFrameWidth(desired_frame_width, bSaveSetting);

                // Only save the setting if requested
                if (bSaveSetting)
                {
                    tracker_view->saveSettings();
                }
                else
                {
                    context.connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override = true;
                }

                // Return back the actual frame width that got set
                result_frame_width->set_new_frame_width(static_cast<float>(tracker_view->getFrameWidth()));

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_frame_height(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_frame_height().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_FRAME_HEIGHT_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const bool bSaveSetting = context.request->request_set_tracker_frame_height().save_setting();
                const float desired_frame_height = context.request->request_set_tracker_frame_height().value();
                PSMoveProtocol::Response_ResultSetTrackerFrameHeight* result_frame_height =
                    response->mutable_result_set_tracker_frame_height();

                // Set the desired frame height on the tracker
                tracker_view->setFrameHeight(desired_frame_height, bSaveSetting);

                // Only save the setting if requested
                if (bSaveSetting)
                {
                    tracker_view->saveSettings();
                }
                else
                {
                    context.connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override = true;
                }

                // Return back the actual frame height that got set
                result_frame_height->set_new_frame_height(static_cast<float>(tracker_view->getFrameHeight()));

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_frame_rate(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_frame_rate().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_FRAME_RATE_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const bool bSaveSetting = context.request->request_set_tracker_frame_rate().save_setting();
                const float desired_frame_rate = context.request->request_set_tracker_frame_rate().value();
                PSMoveProtocol::Response_ResultSetTrackerFrameRate* result_frame_rate =
                    response->mutable_result_set_tracker_frame_rate();

                // Set the desired frame rate on the tracker
                tracker_view->setFrameRate(desired_frame_rate, bSaveSetting);

                // Only save the setting if requested
                if (bSaveSetting)
                {
                    tracker_view->saveSettings();
                }
                else
                {
                    context.connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override = true;
                }

                // Return back the actual frame rate that got set
                result_frame_rate->set_new_frame_rate(static_cast<float>(tracker_view->getFrameRate()));

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_exposure(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_exposure().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_EXPOSURE_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const bool bSaveSetting= context.request->request_set_tracker_exposure().save_setting();
                const float desired_exposure = context.request->request_set_tracker_exposure().value();
                PSMoveProtocol::Response_ResultSetTrackerExposure* result_exposure =
                    response->mutable_result_set_tracker_exposure();

                // Set the desired exposure on the tracker
                tracker_view->setExposure(desired_exposure, bSaveSetting);

                // Only save the setting if requested
                if (bSaveSetting)
                {
                    tracker_view->saveSettings();
                }
                else
                {
                    context.connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override = true;
                }

                // Return back the actual exposure that got set
                result_exposure->set_new_exposure(static_cast<float>(tracker_view->getExposure()));

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_gain(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_gain().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_GAIN_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const bool bSaveSetting = context.request->request_set_tracker_gain().save_setting();
                const double desired_gain = context.request->request_set_tracker_gain().value();
                PSMoveProtocol::Response_ResultSetTrackerGain* result_gain =
                    response->mutable_result_set_tracker_gain();

                // Set the desired gain on the tracker
                tracker_view->setGain(desired_gain, bSaveSetting);

                // Only save the setting if requested
                if (bSaveSetting)
                {
                    tracker_view->saveSettings();
                }
                else
                {
                    context.connection_state->active_tracker_stream_info[tracker_id].has_temp_settings_override = true;
                }

                // Return back the actual gain that got set
                result_gain->set_new_gain(static_cast<float>(tracker_view->getGain()));

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_option(const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_gain().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_OPTION_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const std::string &option_name = context.request->request_set_tracker_option().option_name();
                const int desired_option_index = context.request->request_set_tracker_option().option_index();
                PSMoveProtocol::Response_ResultSetTrackerOption* result_gain =
                    response->mutable_result_set_tracker_option();

                // Set the desired gain on the tracker
                if (tracker_view->setOptionIndex(option_name, desired_option_index))
                {
                    // Return back the actual option index that got set
                    int result_option_index;

                    tracker_view->getOptionIndex(option_name, result_option_index);
                    result_gain->set_option_name(option_name);
                    result_gain->set_new_option_index(result_option_index);

                    tracker_view->saveSettings();

                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
                }
                else
                {
                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
                }
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_tracker_color_preset(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_color_preset().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_PRESET_UPDATED);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const auto device_category= context.request->request_set_tracker_color_preset().device_category();
                const int device_id= context.request->request_set_tracker_color_preset().device_id();

                const PSMoveProtocol::TrackingColorPreset &colorPreset =
                    context.request->request_set_tracker_color_preset().color_preset();
                const eCommonTrackingColorID colorType=
                    static_cast<eCommonTrackingColorID>(colorPreset.color_type());

                // Set the color preset on the tracker
                CommonHSVColorRange inHSVColorRange;
                CommonHSVColorRange outHSVColorRange;
                {
                    inHSVColorRange.hue_range.center= colorPreset.hue_center();
                    inHSVColorRange.hue_range.range = colorPreset.hue_range();
                    inHSVColorRange.saturation_range.center = colorPreset.saturation_center();
                    inHSVColorRange.saturation_range.range = colorPreset.saturation_range();
                    inHSVColorRange.value_range.center = colorPreset.value_center();
                    inHSVColorRange.value_range.range = colorPreset.value_range();

                    switch (device_category)
                    {
                    case PSMoveProtocol::Request_RequestSetTrackerColorPreset_DeviceCategory_CONTROLLER:
                        {
                            ServerControllerView *controller_view = get_controller_view_or_null(device_id);

                            // Assign the color range
                            tracker_view->setControllerTrackingColorPreset(controller_view, colorType, &inHSVColorRange);
                            // Read back what actually got set
                            tracker_view->getControllerTrackingColorPreset(controller_view, colorType, &outHSVColorRange);
                        } break;
                    case PSMoveProtocol::Request_RequestSetTrackerColorPreset_DeviceCategory_HMD:
                        {
                            ServerHMDView *hmd_view = get_hmd_view_or_null(device_id);

                            // Assign the color range
                            tracker_view->setHMDTrackingColorPreset(hmd_view, colorType, &inHSVColorRange);
                            // Read back what actually got set
                            tracker_view->getHMDTrackingColorPreset(hmd_view, colorType, &outHSVColorRange);
                        } break;
                }
//                    tracker_view->saveSettings(); // Carried over from old generic_camera
                }

                // Get the resulting preset from the tracker
                {
                    PSMoveProtocol::Response_ResultSetTrackerColorPreset *result =
                        response->mutable_result_set_tracker_color_preset();
                    result->set_tracker_id(tracker_id);

                    PSMoveProtocol::TrackingColorPreset *presetResult = result->mutable_new_color_preset();
                    presetResult->set_color_type(colorPreset.color_type());
                    presetResult->set_hue_center(outHSVColorRange.hue_range.center);
                    presetResult->set_hue_range(outHSVColorRange.hue_range.range);
                    presetResult->set_saturation_center(outHSVColorRange.saturation_range.center);
                    presetResult->set_saturation_range(outHSVColorRange.saturation_range.range);
                    presetResult->set_value_center(outHSVColorRange.value_range.center);
                    presetResult->set_value_range(outHSVColorRange.value_range.range);
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    inline CommonDevicePose protocol_pose_to_common_device_pose(const PSMoveProtocol::Pose &pose)
    {
        CommonDevicePose result;

        result.Orientation.w = pose.orientation().w();
        result.Orientation.x = pose.orientation().x();
        result.Orientation.y = pose.orientation().y();
        result.Orientation.z = pose.orientation().z();

        result.PositionCm.x = pose.position().x();
        result.PositionCm.y = pose.position().y();
        result.PositionCm.z = pose.position().z();

        return result;
    }

    void handle_request__set_tracker_pose(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_pose().tracker_id();
        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const PSMoveProtocol::Pose &srcPose = 
                    context.request->request_set_tracker_pose().pose();
                CommonDevicePose destPose = protocol_pose_to_common_device_pose(srcPose);

                tracker_view->setTrackerPose(&destPose);
                tracker_view->saveSettings();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    inline void protocol_distortion_to_common_distortion(
        const PSMoveProtocol::TrackerIntrinsics_DistortionCoefficients &d,
        CommonDistortionCoefficients &result)
    {
        result.k1= d.k1();
        result.k2= d.k2();
        result.k3= d.k3();
        result.p1= d.p1();
        result.p2= d.p2();
    }

    inline void protocol_vec3_to_common_vec3(
        const PSMoveProtocol::DoubleVector &v,
        std::array<double, 3> &result)
    {
        result[0]= v.i();
        result[1]= v.j();
        result[2]= v.k();
    }

    inline void protocol_mat33_to_common_mat33(
        const PSMoveProtocol::DoubleMatrix33 &m,
        std::array<double, 9> &result)
    {
        result[0]= m.m00(); result[1]= m.m01(); result[2]= m.m02();
        result[3]= m.m10(); result[4]= m.m11(); result[5]= m.m12();
        result[6]= m.m20(); result[7]= m.m21(); result[8]= m.m22();
    }

    inline void protocol_mat34_to_common_mat34(
        const PSMoveProtocol::DoubleMatrix34 &m,
        std::array<double, 12> &result)
    {
        result[0]= m.m00(); result[1]= m.m01(); result[2]= m.m02(); result[3]= m.m03();
        result[4]= m.m10(); result[5]= m.m11(); result[6]= m.m12(); result[7]= m.m13();
        result[8]= m.m20(); result[9]= m.m21(); result[10]= m.m22(); result[11]= m.m23();
    }
    
    inline void protocol_mat44_to_common_mat44(
        const PSMoveProtocol::DoubleMatrix44 &m,
        std::array<double, 16> &result)
    {
        result[0]= m.m00(); result[1]= m.m01(); result[2]= m.m02(); result[3]= m.m03();
        result[4]= m.m10(); result[5]= m.m11(); result[6]= m.m12(); result[7]= m.m13();
        result[8]= m.m20(); result[9]= m.m21(); result[10]= m.m22(); result[11]= m.m23();
        result[12]= m.m30(); result[13]= m.m31(); result[14]= m.m32(); result[15]= m.m33();
    }

    void handle_request__set_tracker_intrinsics(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_set_tracker_intrinsics().tracker_id();
        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const PSMoveProtocol::TrackerIntrinsics &src_intrinsics= 
                    context.request->request_set_tracker_intrinsics().tracker_intrinsics();
                CommonTrackerIntrinsics dest_intrinsics;
                bool bValidIntrinsics= false;

                if (tracker_view->getIsStereoCamera() && src_intrinsics.has_stereo_intrinsics())
                {
                    const PSMoveProtocol::TrackerIntrinsics_StereoIntrinsics &src_stereo= 
                        src_intrinsics.stereo_intrinsics(); 
                    CommonStereoTrackerIntrinsics &dest_stereo= dest_intrinsics.stereo_intrinsics;

                    dest_intrinsics.intrinsics_type= CommonTrackerIntrinsics::STEREO_TRACKER_INTRINSICS;
                    dest_stereo.pixel_width= src_stereo.tracker_screen_dimensions().x();
                    dest_stereo.pixel_height= src_stereo.tracker_screen_dimensions().y();
                    dest_stereo.hfov= src_stereo.hfov();
                    dest_stereo.vfov= src_stereo.vfov();
                    dest_stereo.znear= src_stereo.znear();
                    dest_stereo.zfar= src_stereo.zfar();
                    protocol_distortion_to_common_distortion(
                        src_stereo.left_distortion_coefficients(), dest_stereo.left_distortion_coefficients);
                    protocol_distortion_to_common_distortion(
                        src_stereo.right_distortion_coefficients(), dest_stereo.right_distortion_coefficients);
                    protocol_mat33_to_common_mat33(
                        src_stereo.left_camera_matrix(), dest_stereo.left_camera_matrix);
                    protocol_mat33_to_common_mat33(
                        src_stereo.right_camera_matrix(), dest_stereo.right_camera_matrix);
                    protocol_mat33_to_common_mat33(
                        src_stereo.left_rectification_rotation(), dest_stereo.left_rectification_rotation);
                    protocol_mat33_to_common_mat33(
                        src_stereo.right_rectification_rotation(), dest_stereo.right_rectification_rotation);
                    protocol_mat34_to_common_mat34(
                        src_stereo.left_rectification_projection(), dest_stereo.left_rectification_projection);
                    protocol_mat34_to_common_mat34(
                        src_stereo.right_rectification_projection(), dest_stereo.right_rectification_projection);
                    protocol_mat33_to_common_mat33(
                        src_stereo.rotation_between_cameras(), dest_stereo.rotation_between_cameras);
                    protocol_vec3_to_common_vec3(
                        src_stereo.translation_between_cameras(), dest_stereo.translation_between_cameras);
                    protocol_mat33_to_common_mat33(
                        src_stereo.essential_matrix(), dest_stereo.essential_matrix);
                    protocol_mat33_to_common_mat33(
                        src_stereo.fundamental_matrix(), dest_stereo.fundamental_matrix);
                    protocol_mat44_to_common_mat44(
                        src_stereo.reprojection_matrix(), dest_stereo.reprojection_matrix);

                    bValidIntrinsics= true;
                }
                else if (!tracker_view->getIsStereoCamera() && src_intrinsics.has_mono_intrinsics())
                {
                    const PSMoveProtocol::TrackerIntrinsics_MonoIntrinics &src_mono= 
                        src_intrinsics.mono_intrinsics(); 
                    CommonMonoTrackerIntrinsics &dest_mono= dest_intrinsics.mono_intrinsics;

                    dest_intrinsics.intrinsics_type= CommonTrackerIntrinsics::MONO_TRACKER_INTRINSICS;
                    dest_mono.pixel_width= src_mono.tracker_screen_dimensions().x();
                    dest_mono.pixel_height= src_mono.tracker_screen_dimensions().y();
                    dest_mono.hfov= src_mono.hfov();
                    dest_mono.vfov= src_mono.vfov();
                    dest_mono.znear= src_mono.znear();
                    dest_mono.zfar= src_mono.zfar();
                    protocol_distortion_to_common_distortion(
                        src_mono.distortion_coefficients(), dest_mono.distortion_coefficients);
                    protocol_mat33_to_common_mat33(
                        src_mono.camera_matrix(), dest_mono.camera_matrix);

                    bValidIntrinsics= true;
                }

                if (bValidIntrinsics)
                {
                    tracker_view->setCameraIntrinsics(dest_intrinsics);
                    tracker_view->saveSettings();

                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
                }
                else
                {
                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
                }
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__save_tracker_profile(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_save_tracker_profile().tracker_id();

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const int controller_id= context.request->request_save_tracker_profile().controller_id();
                ServerControllerView *controller_view= get_controller_view_or_null(controller_id);

                TrackerProfile trackerProfile;

                trackerProfile.clear();
                trackerProfile.frame_width = static_cast<float>(tracker_view->getFrameWidth());
                //trackerProfile.frame_height = static_cast<float>(tracker_view->getFrameHeight());
                trackerProfile.frame_rate = static_cast<float>(tracker_view->getFrameRate());
                trackerProfile.exposure= static_cast<float>(tracker_view->getExposure());
                trackerProfile.gain = static_cast<float>(tracker_view->getGain());

                for (int preset_index = 0; preset_index < eCommonTrackingColorID::MAX_TRACKING_COLOR_TYPES; ++preset_index)
                {
                    tracker_view->getControllerTrackingColorPreset(
                        controller_view,
                        static_cast<eCommonTrackingColorID>(preset_index),
                        &trackerProfile.color_preset_table.color_presets[preset_index]);
                }

                m_device_manager.m_tracker_manager->saveDefaultTrackerProfile(&trackerProfile);

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__apply_tracker_profile(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_apply_tracker_profile().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKER_SETTINGS);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {
                const int controller_id= context.request->request_apply_tracker_profile().controller_id();
                ServerControllerView *controller_view= get_controller_view_or_null(controller_id);

                const TrackerProfile *trackerProfile = 
                    m_device_manager.m_tracker_manager->getDefaultTrackerProfile();
    
                // Apply the profile to the tracker
                tracker_view->setFrameWidth(trackerProfile->frame_width, true);
                //tracker_view->setFrameHeight(trackerProfile->frame_height, true);
                tracker_view->setFrameRate(trackerProfile->frame_rate, true);
                tracker_view->setExposure(trackerProfile->exposure, true);
                tracker_view->setGain(trackerProfile->gain, true);
                for (int preset_index = 0; preset_index < eCommonTrackingColorID::MAX_TRACKING_COLOR_TYPES; ++preset_index)
                {
                    const CommonHSVColorRange *preset= &trackerProfile->color_preset_table.color_presets[preset_index];
                    const eCommonTrackingColorID color_type = static_cast<eCommonTrackingColorID>(preset_index);

                    tracker_view->setControllerTrackingColorPreset(controller_view, color_type, preset);
                }

                // Send the profile application result to the client
                {
                    PSMoveProtocol::Response_ResultTrackerSettings* settings =
                        response->mutable_result_tracker_settings();

                    settings->set_frame_width(static_cast<float>(tracker_view->getFrameWidth()));
                    settings->set_frame_height(static_cast<float>(tracker_view->getFrameHeight()));
                    settings->set_frame_rate(static_cast<float>(tracker_view->getFrameRate()));
                    settings->set_exposure(static_cast<float>(tracker_view->getExposure()));
                    settings->set_gain(static_cast<float>(tracker_view->getGain()));
                    tracker_view->gatherTrackerOptions(settings);
                    tracker_view->gatherTrackingColorPresets(controller_view, settings);
                }

                tracker_view->saveSettings();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__reload_tracker_settings(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int tracker_id = context.request->request_apply_tracker_profile().tracker_id();

        response->set_type(PSMoveProtocol::Response_ResponseType_GENERAL_RESULT);

        if (ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerTrackerViewPtr tracker_view = m_device_manager.getTrackerViewPtr(tracker_id);
            if (tracker_view->getIsOpen())
            {   
                tracker_view->loadSettings();
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__search_for_new_trackers(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        // The video polling threads tend to stall out when we are polling for new devices via libusb.
        // Until we have a better solution, best to just shut down all of the trackers
        // and then wait for them to restart next tracker device refresh.
        m_device_manager.m_tracker_manager->closeAllTrackers();

        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }

    void handle_request__get_tracking_space_settings(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        PSMoveProtocol::Response_ResultTrackingSpaceSettings* settings = response->mutable_result_tracking_space_settings();

        response->set_type(PSMoveProtocol::Response_ResponseType_TRACKING_SPACE_SETTINGS);

        settings->set_global_forward_degrees(m_device_manager.m_tracker_manager->getConfig().global_forward_degrees);
        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }
    // -- hmd requests -----
    inline ServerHMDView *get_hmd_view_or_null(int hmd_id)
    {
        ServerHMDView *hmd_view = nullptr;

        if (ServerUtility::is_index_valid(hmd_id, m_device_manager.getHMDViewMaxCount()))
        {
            ServerHMDViewPtr hmd_view_ptr = m_device_manager.getHMDViewPtr(hmd_id);

            if (hmd_view_ptr->getIsOpen())
            {
                hmd_view = hmd_view_ptr.get();
            }
        }

        return hmd_view;
    }

    void handle_request__get_hmd_list(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        PSMoveProtocol::Response_ResultHMDList* list = response->mutable_result_hmd_list();

        response->set_type(PSMoveProtocol::Response_ResponseType_HMD_LIST);

        for (int hmd_id = 0; hmd_id < m_device_manager.getHMDViewMaxCount(); ++hmd_id)
        {
            ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);

            if (hmd_view->getIsOpen())
            {
                PSMoveProtocol::Response_ResultHMDList_HMDInfo *hmd_info = list->add_hmd_entries();

                switch (hmd_view->getHMDDeviceType())
                {
                case CommonHMDState::Morpheus:
                    {
                        const MorpheusHMD *morpheusHMD= hmd_view->castCheckedConst<MorpheusHMD>();
                        const MorpheusHMDConfig *config= morpheusHMD->getConfig();

                        hmd_info->set_hmd_type(PSMoveProtocol::Morpheus);
                        hmd_info->set_prediction_time(config->prediction_time);
                        hmd_info->set_orientation_filter(config->orientation_filter_type);
                        hmd_info->set_position_filter(config->position_filter_type);
                    }
                    break;
                case CommonHMDState::VirtualHMD:
                    {
                        const VirtualHMD *virtualHMD= hmd_view->castCheckedConst<VirtualHMD>();
                        const VirtualHMDConfig *config= virtualHMD->getConfig();

                        hmd_info->set_hmd_type(PSMoveProtocol::VirtualHMD);
                        hmd_info->set_prediction_time(config->prediction_time);
                        hmd_info->set_position_filter(config->position_filter_type);
                    }
                    break;
                default:
                    assert(0 && "Unhandled tracker type");
                }

                hmd_info->set_hmd_id(hmd_id);
                hmd_info->set_device_path(hmd_view->getUSBDevicePath());
                hmd_info->set_tracking_color_type(static_cast<PSMoveProtocol::TrackingColorType>(hmd_view->getTrackingColorID()));
            }
        }

        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }

    void handle_request__start_hmd_data_stream(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const PSMoveProtocol::Request_RequestStartHmdDataStream& request =
            context.request->request_start_hmd_data_stream();
        int hmd_id = request.hmd_id();

        if (ServerUtility::is_index_valid(hmd_id, m_device_manager.getHMDViewMaxCount()))
        {
            ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);

            if (hmd_view->getIsOpen())
            {
                HMDStreamInfo &streamInfo =
                    context.connection_state->active_hmd_stream_info[hmd_id];

                // The hmd manager will always publish updates regardless of who is listening.
                // All we have to do is keep track of which connections care about the updates.
                context.connection_state->active_hmd_streams.set(hmd_id, true);

                // Set control flags for the stream
                streamInfo.Clear();
                streamInfo.include_position_data = request.include_position_data();
                streamInfo.include_physics_data = request.include_physics_data();
                streamInfo.include_raw_sensor_data = request.include_raw_sensor_data();
                streamInfo.include_calibrated_sensor_data = request.include_calibrated_sensor_data();
                streamInfo.include_raw_tracker_data = request.include_raw_tracker_data();
                streamInfo.disable_roi = request.disable_roi();

                SERVER_LOG_INFO("ServerRequestHandler") << "Start hmd(" << hmd_id << ") stream ("
                    << "pos=" << streamInfo.include_position_data
                    << ",phys=" << streamInfo.include_physics_data
                    << ",raw_sens=" << streamInfo.include_raw_sensor_data
                    << ",cal_sens=" << streamInfo.include_calibrated_sensor_data
                    << ",trkr=" << streamInfo.include_raw_tracker_data
                    << ",roi=" << streamInfo.disable_roi
                    << ")";

                if (streamInfo.disable_roi)
                {
                    ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);

                    hmd_view->pushDisableROI();
                }

                if (streamInfo.include_position_data)
                {
                    ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);

                    hmd_view->startTracking();
                }

                // Return the name of the shared memory block the video frames will be written to
        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }
            else
            {
                // Device not opened
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__stop_hmd_data_stream(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        int hmd_id = context.request->request_stop_hmd_data_stream().hmd_id();

        if (ServerUtility::is_index_valid(hmd_id, m_device_manager.getHMDViewMaxCount()))
        {
            ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);

            if (hmd_view->getIsOpen())
            {
                const HMDStreamInfo &streamInfo = context.connection_state->active_hmd_stream_info[hmd_id];

                if (streamInfo.disable_roi)
                {
                    hmd_view->popDisableROI();
                }

                if (streamInfo.include_position_data)
                {
                    hmd_view->stopTracking();
                }

                context.connection_state->active_hmd_streams.set(hmd_id, false);
                context.connection_state->active_hmd_stream_info[hmd_id].Clear();

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                // Device not opened
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }


    void handle_request__set_hmd_led_tracking_color(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int connection_id = context.connection_state->connection_id;
        const int hmd_id = context.request->set_hmd_led_tracking_color_request().hmd_id();
        const eCommonTrackingColorID newColorID=
            static_cast<eCommonTrackingColorID>(context.request->set_hmd_led_tracking_color_request().color_type());

        ServerHMDViewPtr HmdView = m_device_manager.getHMDViewPtr(hmd_id);

        if (HmdView && 
            HmdView->getHMDDeviceType() == CommonDeviceState::VirtualHMD)
        {
            const eCommonTrackingColorID oldColorID = HmdView->getTrackingColorID();

            if (newColorID != oldColorID)
            {
                // Give up control of our existing tracking color
                if (oldColorID != eCommonTrackingColorID::INVALID_COLOR)
                {
                    m_device_manager.m_tracker_manager->freeTrackingColorID(oldColorID);
                }

                // Take the color from any other controller that might have it
                if (m_device_manager.m_tracker_manager->claimTrackingColorID(HmdView.get(), newColorID))
                {
                    // Assign the new color to ourselves
                    HmdView->setTrackingColorID(newColorID);
                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
                }
                else
                {
                    if (oldColorID != eCommonTrackingColorID::INVALID_COLOR)
                    {
                        m_device_manager.m_tracker_manager->claimTrackingColorID(HmdView.get(), oldColorID);
                    }

                    response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
                }
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_hmd_accelerometer_calibration(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int hmd_id = context.request->set_hmd_accelerometer_calibration_request().hmd_id();

        ServerHMDViewPtr HMDView = m_device_manager.getHMDViewPtr(hmd_id);

        if (HMDView && HMDView->getHMDDeviceType() == CommonDeviceState::Morpheus)
        {
            MorpheusHMD *hmd = HMDView->castChecked<MorpheusHMD>();
            IPoseFilter *poseFilter = HMDView->getPoseFilterMutable();
            MorpheusHMDConfig *config = hmd->getConfigMutable();

            const auto &request = context.request->set_hmd_accelerometer_calibration_request();

            // Compute the bias as 1g subtracted from the measured direction of gravity
            CommonDeviceVector measured_g;
            set_config_vector(request.raw_average_gravity(), measured_g);
            float length = sqrtf(measured_g.i*measured_g.i + measured_g.j*measured_g.j + measured_g.k*measured_g.k);
            if (length > k_real_epsilon)
            {
                config->raw_accelerometer_bias.i = measured_g.i * (1.f - 1.f / (length*config->accelerometer_gain.i));
                config->raw_accelerometer_bias.j = measured_g.j * (1.f - 1.f / (length*config->accelerometer_gain.j));
                config->raw_accelerometer_bias.k = measured_g.k * (1.f - 1.f / (length*config->accelerometer_gain.k));
            }

            config->raw_accelerometer_variance = request.raw_variance();
            config->save();

            // Reset the orientation filter state the calibration changed
            poseFilter->resetState();

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_hmd_gyroscope_calibration(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int hmd_id = context.request->set_hmd_accelerometer_calibration_request().hmd_id();

        ServerHMDViewPtr HMDView = m_device_manager.getHMDViewPtr(hmd_id);

        if (HMDView && HMDView->getHMDDeviceType() == CommonDeviceState::Morpheus)
        {
            MorpheusHMD *hmd = HMDView->castChecked<MorpheusHMD>();
            MorpheusHMDConfig *config = hmd->getConfigMutable();

            const auto &request = context.request->set_hmd_gyroscope_calibration_request();

            set_config_vector(request.raw_bias(), config->raw_gyro_bias);
            config->raw_gyro_variance = request.raw_variance();
            config->raw_gyro_drift = request.raw_drift();
            config->save();

            // Reset the orientation filter state the calibration changed
            HMDView->getPoseFilterMutable()->resetState();

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_hmd_orientation_filter(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int hmd_id = context.request->request_set_hmd_orientation_filter().hmd_id();

        ServerHMDViewPtr HmdView = m_device_manager.getHMDViewPtr(hmd_id);
        const PSMoveProtocol::Request_RequestSetHMDOrientationFilter &request =
            context.request->request_set_hmd_orientation_filter();

        if (HmdView && HmdView->getIsOpen())
        {
            if (HmdView->getHMDDeviceType() == CommonDeviceState::Morpheus)
            {
                MorpheusHMD *hmd = HmdView->castChecked<MorpheusHMD>();
                MorpheusHMDConfig *config = hmd->getConfigMutable();

                if (config->orientation_filter_type != request.orientation_filter())
                {
                    config->orientation_filter_type = request.orientation_filter();
                    config->save();

                    HmdView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_hmd_position_filter(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int hmd_id = context.request->request_set_hmd_position_filter().hmd_id();

        ServerHMDViewPtr HmdView = m_device_manager.getHMDViewPtr(hmd_id);
        const PSMoveProtocol::Request_RequestSetHMDPositionFilter &request =
            context.request->request_set_hmd_position_filter();

        if (HmdView && HmdView->getIsOpen())
        {
            if (HmdView->getHMDDeviceType() == CommonDeviceState::Morpheus)
            {
                MorpheusHMD *hmd = HmdView->castChecked<MorpheusHMD>();
                MorpheusHMDConfig *config = hmd->getConfigMutable();

                if (config->position_filter_type != request.position_filter())
                {
                    config->position_filter_type = request.position_filter();
                    config->save();

                    HmdView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (HmdView->getHMDDeviceType() == CommonDeviceState::VirtualHMD)
            {
                VirtualHMD *hmd = HmdView->castChecked<VirtualHMD>();
                VirtualHMDConfig *config = hmd->getConfigMutable();

                if (config->position_filter_type != request.position_filter())
                {
                    config->position_filter_type = request.position_filter();
                    config->save();

                    HmdView->resetPoseFilter();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_hmd_prediction_time(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int hmd_id = context.request->request_set_hmd_prediction_time().hmd_id();

        ServerHMDViewPtr HmdView = m_device_manager.getHMDViewPtr(hmd_id);
        const PSMoveProtocol::Request_RequestSetHMDPredictionTime &request =
            context.request->request_set_hmd_prediction_time();

        if (HmdView && HmdView->getIsOpen())
        {
            if (HmdView->getHMDDeviceType() == CommonDeviceState::Morpheus)
            {
                MorpheusHMD *hmd = HmdView->castChecked<MorpheusHMD>();
                MorpheusHMDConfig *config = hmd->getConfigMutable();

                if (config->prediction_time != request.prediction_time())
                {
                    config->prediction_time = request.prediction_time();
                    config->save();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else if (HmdView->getHMDDeviceType() == CommonDeviceState::VirtualHMD)
            {
                VirtualHMD *hmd = HmdView->castChecked<VirtualHMD>();
                VirtualHMDConfig *config = hmd->getConfigMutable();

                if (config->prediction_time != request.prediction_time())
                {
                    config->prediction_time = request.prediction_time();
                    config->save();
                }

                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
            }
            else
            {
                response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
            }
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__set_hmd_data_stream_tracker_index(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        const int hmd_id = context.request->request_set_hmd_data_stream_tracker_index().hmd_id();
        const int tracker_id = context.request->request_set_hmd_data_stream_tracker_index().tracker_id();

        if (ServerUtility::is_index_valid(hmd_id, m_device_manager.getHMDViewMaxCount()) &&
            ServerUtility::is_index_valid(tracker_id, m_device_manager.getTrackerViewMaxCount()))
        {
            ServerHMDViewPtr hmd_view = m_device_manager.getHMDViewPtr(hmd_id);
            HMDStreamInfo &streamInfo =
                context.connection_state->active_hmd_stream_info[hmd_id];

            SERVER_LOG_INFO("ServerRequestHandler") << "Set hmd(" << hmd_id << ") stream tracker id: " << tracker_id;

            streamInfo.selected_tracker_index= tracker_id;

            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
        }
        else
        {
            response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_ERROR);
        }
    }

    void handle_request__get_service_version(
        const RequestContext &context,
        PSMoveProtocol::Response *response)
    {
        PSMoveProtocol::Response_ResultServiceVersion* version_info = response->mutable_result_service_version();

        response->set_type(PSMoveProtocol::Response_ResponseType_SERVICE_VERSION);

        // Return the protocol version
        version_info->set_version(PSM_PROTOCOL_VERSION_STRING);
        response->set_result_code(PSMoveProtocol::Response_ResultCode_RESULT_OK);
    }

    // -- Data Frame Updates -----
    void handle_data_frame__controller_packet(
        RequestConnectionStatePtr connection_state,
        DeviceInputDataFramePtr data_frame)
    {
        const auto &controllerDataPacket = data_frame->controller_data_packet();
        const int controller_id = controllerDataPacket.controller_id();

        if (ServerUtility::is_index_valid(controller_id, m_device_manager.getControllerViewMaxCount()))
        {
            ServerControllerViewPtr controller_view = m_device_manager.getControllerViewPtr(controller_id);
            ControllerStreamInfo &streamInfo = connection_state->active_controller_stream_info[controller_id];

            // Don't consider this data frame if the controller isn't in a streamable connection
            // or if the sequence number is old
            if (controller_view->getIsStreamable() && 
                controllerDataPacket.sequence_num() > streamInfo.last_data_input_sequence_number)
            {
                // Remember the last sequence number we received from this connection
                streamInfo.last_data_input_sequence_number = controllerDataPacket.sequence_num();

                switch (controller_view->getControllerDeviceType())
                {
                case CommonDeviceState::eDeviceType::PSMove:
                    {
                        const auto &psmove_state= controllerDataPacket.psmove_state();

                        // Update the rumble
                        const float rumbleValue= static_cast<float>(psmove_state.rumble_value()) / 255.f;
                        controller_view->setControllerRumble(rumbleValue, CommonControllerState::ChannelAll);

                        // Update the override led color
                        {
                            unsigned char r = static_cast<unsigned char>(psmove_state.led_r());
                            unsigned char g = static_cast<unsigned char>(psmove_state.led_g());
                            unsigned char b = static_cast<unsigned char>(psmove_state.led_b());

                            // (0,0,0) is treated as clearing the override
                            if (r == 0 && g == 0 && b == 0)
                            {
                                if (controller_view->getIsLEDOverrideActive())
                                {
                                    // Removes the over led color and restores the tracking color
                                    // of the controller is currently being tracked
                                    controller_view->clearLEDOverride();
                                }
                            }
                            // Otherwise we are setting the override to a new color
                            else
                            {
                                // Sets the bulb LED color to some new override color
                                // If tracking was active this likely will affect controller tracking
                                controller_view->setLEDOverride(r, g, b);
                            }

                            // Flag if the LED override is active
                            // If the stream closes and this flag is active we'll need to clear the led override
                            streamInfo.led_override_active= controller_view->getIsLEDOverrideActive();
                        }
                    } break;
                case CommonDeviceState::eDeviceType::PSNavi:
                    {
                        // Nothing to update...
                    } break;
                case CommonDeviceState::eDeviceType::PSDualShock4:
                    {
                        const auto &psmove_state= controllerDataPacket.psdualshock4_state();

                        // Update the rumble
                        const float bigRumbleValue= static_cast<float>(psmove_state.big_rumble_value()) / 255.f;
                        const float smallRumbleValue= static_cast<float>(psmove_state.small_rumble_value()) / 255.f;
                        controller_view->setControllerRumble(bigRumbleValue, CommonControllerState::ChannelLeft);
                        controller_view->setControllerRumble(smallRumbleValue, CommonControllerState::ChannelRight);

                        // Update the override led color
                        {
                            unsigned char r = static_cast<unsigned char>(psmove_state.led_r());
                            unsigned char g = static_cast<unsigned char>(psmove_state.led_g());
                            unsigned char b = static_cast<unsigned char>(psmove_state.led_b());

                            // (0,0,0) is treated as clearing the override
                            if (r == 0 && g == 0 && b == 0)
                            {
                                if (controller_view->getIsLEDOverrideActive())
                                {
                                    // Removes the over led color and restores the tracking color
                                    // of the controller is currently being tracked
                                    controller_view->clearLEDOverride();
                                }
                            }
                            // Otherwise we are setting the override to a new color
                            else
                            {
                                // Sets the bulb LED color to some new override color
                                // If tracking was active this likely will affect controller tracking
                                controller_view->setLEDOverride(r, g, b);
                            }

                            // Flag if the LED override is active
                            // If the stream closes and this flag is active we'll need to clear the led override
                            streamInfo.led_override_active= controller_view->getIsLEDOverrideActive();
                        }
                    } break;
                }
            }
        }
    }

private:
    DeviceManager &m_device_manager;
    t_connection_state_map m_connection_state_map;
};

//-- public interface -----
ServerRequestHandler *ServerRequestHandler::m_instance = NULL;

ServerRequestHandler::ServerRequestHandler(DeviceManager *deviceManager)
    : m_implementation_ptr(new ServerRequestHandlerImpl(*deviceManager))
{

}

ServerRequestHandler::~ServerRequestHandler()
{
    if (m_instance != NULL)
    {
        SERVER_LOG_ERROR("~ServerRequestHandler") << "Request handler deleted without calling shutdown first!";
    }

    delete m_implementation_ptr;
}

bool ServerRequestHandler::any_active_bluetooth_requests() const
{
    return m_implementation_ptr->any_active_bluetooth_requests();
}

bool ServerRequestHandler::startup()
{
    m_instance= this;
    return true;
}

void ServerRequestHandler::update()
{
    return m_implementation_ptr->update();
}

void ServerRequestHandler::shutdown()
{
    m_instance= NULL;
}

ResponsePtr ServerRequestHandler::handle_request(int connection_id, RequestPtr request)
{
    return m_implementation_ptr->handle_request(connection_id, request);
}

void ServerRequestHandler::handle_input_data_frame(DeviceInputDataFramePtr data_frame)
{
    return m_implementation_ptr->handle_input_data_frame(data_frame);
}

void ServerRequestHandler::handle_client_connection_stopped(int connection_id)
{
    return m_implementation_ptr->handle_client_connection_stopped(connection_id);
}

void ServerRequestHandler::publish_controller_data_frame(
    ServerControllerView *controller_view, 
    t_generate_controller_data_frame_for_stream callback)
{
    return m_implementation_ptr->publish_controller_data_frame(controller_view, callback);
}

void ServerRequestHandler::publish_tracker_data_frame(
    class ServerTrackerView *tracker_view,
    t_generate_tracker_data_frame_for_stream callback)
{
    return m_implementation_ptr->publish_tracker_data_frame(tracker_view, callback);
}

void ServerRequestHandler::publish_hmd_data_frame(
    class ServerHMDView *hmd_view,
    t_generate_hmd_data_frame_for_stream callback)
{
    return m_implementation_ptr->publish_hmd_data_frame(hmd_view, callback);
}
