//-- inludes -----
#include "AppStage_HMDModelCalibration.h"
#include "AppStage_HMDSettings.h"
#include "AppStage_MainMenu.h"
#include "App.h"
#include "AssetManager.h"
#include "Camera.h"
#include "GeometryUtility.h"
#include "Logger.h"
#include "MathEigen.h"
#include "MathUtility.h"
#include "Renderer.h"
#include "UIConstants.h"
#include "PSMoveProtocolInterface.h"
#include "PSMoveProtocol.pb.h"
#include "SharedTrackerState.h"
#include "MathGLM.h"
#include "MathEigen.h"

#include "SDL_keycode.h"
#include "SDL_opengl.h"

#include "ICP.h"

#include "PSMoveClient_CAPI.h"

#include <imgui.h>
#include <sstream>
#include <vector>
#include <set>

//-- typedefs ----
namespace SICP
{
    typedef Eigen::Matrix<double, 3, Eigen::Dynamic> Vertices;
};

namespace cv
{
    typedef Matx<double, 5, 1> Matx51d;
}

//-- statics ----
const char *AppStage_HMDModelCalibration::APP_STAGE_NAME = "HMDModelCalibration";

//-- constants -----
static const int k_max_projection_points = 16;
static const int k_morpheus_led_count = 9;
static const int k_led_position_sample_count = 100;

static float k_cosine_aligned_camera_angle = cosf(60.f *k_degrees_to_radians);

static const float k_default_correspondance_tolerance = 1.0f;

static const float k_icp_point_snap_distance = 3.0; // cm

static const glm::vec3 k_psmove_frustum_color = glm::vec3(0.1f, 0.7f, 0.3f);
static const glm::vec3 k_psmove_frustum_color_no_track = glm::vec3(1.0f, 0.f, 0.f);

//-- private methods -----
static PSMVector2f projectTrackerRelativePositionOnTracker(
    const PSMVector3f &trackerRelativePosition,
    const PSMMatrix3d &camera_matrix,
    const PSMDistortionCoefficients &distortion_coefficients);
static void drawHMD(const PSMHeadMountedDisplay *hmdView, const glm::mat4 &transform);

//-- private structures -----
struct StereoCameraSectionState
{
    // Raw video buffer
    int frameWidth;
    int frameHeight;
    TextureAsset *textureAsset;

    // Stereo Calibration state
    cv::Matx33d intrinsic_matrix;
    cv::Matx33d rectification_rotation;
    cv::Matx34d rectification_projection;
    cv::Matx51d distortion_coeffs;

    // Distortion preview
    cv::Mat *distortionMapX;
    cv::Mat *distortionMapY;
    cv::Mat *bgrUndistortBuffer;

    void init()
    {
        frameWidth= frameHeight = 0;
        textureAsset = nullptr;
        bgrUndistortBuffer = nullptr;
        distortionMapX = nullptr;
        distortionMapY = nullptr;
    }

    void dispose()
    {
        if (textureAsset != nullptr)
        {
            delete textureAsset;
            textureAsset= nullptr;
        }

        if (bgrUndistortBuffer != nullptr)
        {
            delete bgrUndistortBuffer;
            bgrUndistortBuffer= nullptr;
        }

        if (distortionMapX != nullptr)
        {
            delete distortionMapX;
            distortionMapX= nullptr;
        }

        if (distortionMapY != nullptr)
        {
            delete distortionMapY;
            distortionMapY= nullptr;
        }
    }

    void applyIntrinsics(const PSMTrackerIntrinsics &intrinsics, const PSMVideoFrameSection section_index)
    {
        frameWidth= static_cast<int>(intrinsics.intrinsics.stereo.pixel_width);
        frameHeight= static_cast<int>(intrinsics.intrinsics.stereo.pixel_height);

        switch (section_index)
        {
        case PSMVideoFrameSection_Left:
            intrinsic_matrix= psmove_matrix3x3_to_cv_mat33d(intrinsics.intrinsics.stereo.left_camera_matrix);
            rectification_rotation= psmove_matrix3x3_to_cv_mat33d(intrinsics.intrinsics.stereo.left_rectification_rotation);
            rectification_projection= psmove_matrix3x4_to_cv_mat34d(intrinsics.intrinsics.stereo.left_rectification_projection);
            distortion_coeffs= psm_distortion_to_cv_vec5(intrinsics.intrinsics.stereo.left_distortion_coefficients);
            break;
        case PSMVideoFrameSection_Right:
            intrinsic_matrix= psmove_matrix3x3_to_cv_mat33d(intrinsics.intrinsics.stereo.right_camera_matrix);
            rectification_rotation= psmove_matrix3x3_to_cv_mat33d(intrinsics.intrinsics.stereo.right_rectification_rotation);
            rectification_projection= psmove_matrix3x4_to_cv_mat34d(intrinsics.intrinsics.stereo.right_rectification_projection);
            distortion_coeffs= psm_distortion_to_cv_vec5(intrinsics.intrinsics.stereo.right_distortion_coefficients);
            break;
        default:
            break;
        }

        // Clean up any previously allocated buffers
        dispose();

        // Create a texture to render the video frame to
        textureAsset = new TextureAsset();
        textureAsset->init(
            frameWidth, frameHeight,
            GL_RGB, // texture format
            GL_BGR, // buffer format
            nullptr);

        bgrUndistortBuffer = new cv::Mat(frameHeight, frameWidth, CV_8UC3);
        distortionMapX = new cv::Mat(cv::Size(frameWidth, frameHeight), CV_32FC1);
        distortionMapY = new cv::Mat(cv::Size(frameWidth, frameHeight), CV_32FC1);

        cv::initUndistortRectifyMap(
            intrinsic_matrix, distortion_coeffs, 
            rectification_rotation, rectification_projection, 
            cv::Size(frameWidth, frameHeight),
            CV_32FC1, // Distortion map type
            *distortionMapX, *distortionMapY);
    }

    inline cv::Matx51d psm_distortion_to_cv_vec5(const PSMDistortionCoefficients &distortion_coeffs)
    {
        cv::Matx51d cv_distortion_coeffs;
        cv_distortion_coeffs(0, 0)= distortion_coeffs.k1;
        cv_distortion_coeffs(1, 0)= distortion_coeffs.k2;
        cv_distortion_coeffs(2, 0)= distortion_coeffs.p1;
        cv_distortion_coeffs(3, 0)= distortion_coeffs.p2;
        cv_distortion_coeffs(4, 0)= distortion_coeffs.k3;

        return cv_distortion_coeffs;
    }

    void applyVideoFrame(const unsigned char *buffer)
    {
        const cv::Mat bgrSourceBuffer(frameHeight, frameWidth, CV_8UC3, const_cast<unsigned char *>(buffer));

        // Apply the distortion map
        cv::remap(
            bgrSourceBuffer, *bgrUndistortBuffer, 
            *distortionMapX, *distortionMapY, 
            cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // Copy undistorted buffer into texture
        textureAsset->copyBufferIntoTexture(bgrUndistortBuffer->data);
    }
};

struct StereoCameraState
{
    PSMTracker *trackerView;
    StereoCameraSectionState sections[2];

    // Stereo Calibration state
    Eigen::Matrix3f F_ab; // Fundamental matrix from left to right tracker frame
    Eigen::Matrix4d Q; // Reprojection matrix
    float tolerance;

    void init()
    {
        trackerView= nullptr;
        sections[PSMVideoFrameSection_Left].init();
        sections[PSMVideoFrameSection_Right].init();
        memset(this, 0, sizeof(StereoCameraState));
        tolerance = k_default_correspondance_tolerance;
    }

    void applyIntrinsics(const PSMTrackerIntrinsics &intrinsics)
    {
        sections[PSMVideoFrameSection_Left].applyIntrinsics(intrinsics, PSMVideoFrameSection_Left);
        sections[PSMVideoFrameSection_Right].applyIntrinsics(intrinsics, PSMVideoFrameSection_Right);

        F_ab= psm_matrix3d_to_eigen_matrix3f(intrinsics.intrinsics.stereo.fundamental_matrix);
        Q= psm_matrix4d_to_eigen_matrix4d(intrinsics.intrinsics.stereo.reprojection_matrix);

    }

    void dispose()
    {
        sections[PSMVideoFrameSection_Left].dispose();
        sections[PSMVideoFrameSection_Right].dispose();
    }

    bool do_points_correspond(
        const cv::Mat &pointA,
        const cv::Mat &pointB,
        float tolerance) const
    {
        //See if image point A * Fundamental Matrix * image point B <= tolerance
        const Eigen::Vector3f a(pointA.at<float>(0,0), pointA.at<float>(1,0), 1.f);
        const Eigen::Vector3f b(pointB.at<float>(0,0), pointB.at<float>(1,0), 1.f);
        const float epipolar_distance = fabsf(a.transpose() * F_ab * b);

        return epipolar_distance <= tolerance;
    }
};

struct LEDModelSamples
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3f position_samples[k_led_position_sample_count];
    Eigen::Vector3f average_position;
    int position_sample_count;

    void init()
    {
        average_position = Eigen::Vector3f::Zero();
        position_sample_count = 0;
    }

    bool add_position(const Eigen::Vector3f &point)
    {
        bool bAdded = false;

        if (position_sample_count < k_led_position_sample_count)
        {
            position_samples[position_sample_count] = point;
            ++position_sample_count;

            // Recompute the average position
            if (position_sample_count > 1)
            {
                const float N = static_cast<float>(position_sample_count);

                average_position = Eigen::Vector3f::Zero();
                for (int position_index = 0; position_index < position_sample_count; ++position_index)
                {
                    average_position += position_samples[position_index];
                }
                average_position /= N;
            }
            else
            {
                average_position = position_samples[0];
            }

            bAdded = true;
        }

        return bAdded;
    }
};

class HMDModelState
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    HMDModelState(const PSMTrackingShape *hmd_tracking_shape)
        : m_expectedLEDCount(0)
        , m_seenLEDCount(0)
        , m_totalLEDSampleCount(0)
        , m_ledSampleSet(nullptr)
        , m_bIsIcpTransformValid(false)
    {
        assert(hmd_tracking_shape->shape_type == PSMTrackingShape_PointCloud);
        assert(hmd_tracking_shape->shape.pointcloud.point_count > 0);

        m_expectedLEDCount= hmd_tracking_shape->shape.pointcloud.point_count;
        m_ledSampleSet= new LEDModelSamples[hmd_tracking_shape->shape.pointcloud.point_count];
        m_icpModelToSourceTransform = Eigen::Affine3d::Identity();

        for (int led_index = 0; led_index < m_expectedLEDCount; ++led_index)
        {
            m_ledSampleSet[led_index].init();
        }

        m_icpModelVertices.resize(Eigen::NoChange, hmd_tracking_shape->shape.pointcloud.point_count);
        for (int source_index = 0; source_index < hmd_tracking_shape->shape.pointcloud.point_count; ++source_index)
        {
            const PSMVector3f &point= hmd_tracking_shape->shape.pointcloud.points[source_index];

            m_icpModelVertices(0, source_index) = point.x;
            m_icpModelVertices(1, source_index) = point.y;
            m_icpModelVertices(2, source_index) = point.z;
        }

        m_icpModelKDTree= new nanoflann::KDTreeAdaptor<SICP::Vertices, 3, nanoflann::metric_L2_Simple>(m_icpModelVertices);
    }

    ~HMDModelState()
    {
        if (m_icpModelKDTree)
            delete m_icpModelKDTree;

        delete[] m_ledSampleSet;
    }

    bool getIsComplete() const
    {
        const int expectedSampleCount = m_expectedLEDCount * k_led_position_sample_count;

        return m_totalLEDSampleCount >= expectedSampleCount;
    }

    float getProgressFraction() const 
    {
        const float expectedSampleCount = static_cast<float>(m_seenLEDCount * k_led_position_sample_count);
        const float fraction = static_cast<float>(m_totalLEDSampleCount) / expectedSampleCount;

        return fraction;
    }

    bool getHmdTransform(glm::mat4 &out_hmd_transform)
    {
        if (m_bIsIcpTransformValid)
        {
            out_hmd_transform= eigen_matrix4f_to_glm_mat4(m_icpModelToSourceTransform.cast<float>().matrix());
        }

        return m_bIsIcpTransformValid;
    }

    void recordSamples(PSMHeadMountedDisplay *hmd_view, StereoCameraState *stereo_camera_state)
    {
        if (triangulateHMDProjections(hmd_view, stereo_camera_state, m_lastCorrelatedPoints))
        {
            const int source_point_count = static_cast<int>(m_lastCorrelatedPoints.size());

            if (source_point_count >= 3)
            {
                if (m_seenLEDCount > 0)
                {
                    // Copy the triangulated vertices into a 3xN matric the SICP algorithms can use
                    SICP::Vertices icpSourceVertices;
                    icpSourceVertices.resize(Eigen::NoChange, source_point_count);
                    for (int source_index = 0; source_index < source_point_count; ++source_index)
                    {
                        const Eigen::Vector3f &point = m_lastCorrelatedPoints[source_index].triangulated_point_cm;

                        icpSourceVertices(0, source_index) = point.x();
                        icpSourceVertices(1, source_index) = point.y();
                        icpSourceVertices(2, source_index) = point.z();
                    }

                    // Build kd-tree of the current set of target vertices
                    nanoflann::KDTreeAdaptor<SICP::Vertices, 3, nanoflann::metric_L2_Simple> kdtree(m_icpModelVertices);

                    // Attempt to align the new triangulated points with the previously found LED locations
                    // using the ICP algorithm
                    SICP::Parameters params;
                    params.p = .5;
                    params.max_icp = 15;
                    params.print_icpn = true;
                    SICP::point_to_point(icpSourceVertices, m_icpModelVertices, params);

                    // Update the LED models based on the alignment
                    bool bUpdateTargetVertices = false;
                    for (int source_index = 0; source_index < icpSourceVertices.cols(); ++source_index)
                    {
                        const Eigen::Vector3d source_vertex = icpSourceVertices.col(source_index).cast<double>();
                        const int closest_led_index = kdtree.closest(source_vertex.data());
                        const Eigen::Vector3d closest_led_position = m_ledSampleSet[closest_led_index].average_position.cast<double>();
                        const double cloest_distance_sqrd = (closest_led_position - source_vertex).squaredNorm();

                        // Add the points to the their respective bucket...
                        if (cloest_distance_sqrd <= k_icp_point_snap_distance)
                        {
                            bUpdateTargetVertices |= addPointToLedModel(closest_led_index, source_vertex.cast<float>());
                        }
                        // ... or make a new bucket if no point at that location
                        else
                        {
                            bUpdateTargetVertices |= addLedModel(source_vertex.cast<float>());
                        }
                    }

                    if (bUpdateTargetVertices)
                    {
                        rebuildModelVertices();
                    }
                }
                else
                {
                    for (auto it = m_lastCorrelatedPoints.begin(); it != m_lastCorrelatedPoints.end(); ++it)
                    {
                        addLedModel(it->triangulated_point_cm);
                    }

                    rebuildModelVertices();
                }
            }

            //TODO:
            /*
            // Create a mesh from the average of the best N buckets
            // where N is the expected tracking light count from HMD properties
            */
        }
    }

    void computeHmdTransform(PSMHeadMountedDisplay *hmd_view, StereoCameraState *stereo_camera_state)
    {
        if (triangulateHMDProjections(hmd_view, stereo_camera_state, m_lastCorrelatedPoints))
        {
            const int source_point_count = static_cast<int>(m_lastCorrelatedPoints.size());

            if (source_point_count >= 3)
            {
                // Copy the triangulated vertices into a 3xN matrix the SICP algorithms can use
                SICP::Vertices icpSourceVertices, icpAlignedVertices;
                icpSourceVertices.resize(Eigen::NoChange, source_point_count);
                icpAlignedVertices.resize(Eigen::NoChange, source_point_count);
                for (int aligned_index = 0; aligned_index < source_point_count; ++aligned_index)
                {
                    const Eigen::Vector3f &point = m_lastCorrelatedPoints[aligned_index].triangulated_point_cm;

                    icpSourceVertices(0, aligned_index) = point.x();
                    icpSourceVertices(1, aligned_index) = point.y();
                    icpSourceVertices(2, aligned_index) = point.z();
                    icpAlignedVertices(0, aligned_index) = point.x();
                    icpAlignedVertices(1, aligned_index) = point.y();
                    icpAlignedVertices(2, aligned_index) = point.z();
                }

                if (m_bIsIcpTransformValid)
                {
                    const Eigen::Affine3d sourceToModelTransform= m_icpModelToSourceTransform.inverse();

                    icpAlignedVertices= sourceToModelTransform * icpSourceVertices;
                }
                else
                {
                    // Subtract off the centroid of the source points
                    // to get the source points in close proximity to the target points
                    // which are centered at the origin
                    const Eigen::Vector3f centroid(
                        icpAlignedVertices.row(0).mean(), 
                        icpAlignedVertices.row(1).mean(), 
                        icpAlignedVertices.row(2).mean());

                    icpAlignedVertices.row(0).array() -= centroid(0);
                    icpAlignedVertices.row(1).array() -= centroid(1); 
                    icpAlignedVertices.row(2).array() -= centroid(2);
                }

                // Attempt to align the new triangulated points with the Model
                // using the ICP algorithm
                SICP::Parameters params;
                params.p = .5;
                params.max_icp = 15;
                params.print_icpn = false;
                SICP::point_to_point(icpAlignedVertices, m_icpModelVertices, params);

                // Find the closest vertices on the model to aligned source vertices
                SICP::Vertices icpCorrespondingModelVertices;
                icpCorrespondingModelVertices.resize(Eigen::NoChange, source_point_count);
                for (int aligned_index = 0; aligned_index < icpAlignedVertices.cols(); ++aligned_index)
                {
                    const Eigen::Vector3d aligned_vertex = 
                        icpAlignedVertices.col(aligned_index);
                    const int corresponding_model_point_index = 
                        m_icpModelKDTree->closest(aligned_vertex.data());

                    icpCorrespondingModelVertices.col(aligned_index)=
                        m_icpModelVertices.col(corresponding_model_point_index);
                }

                // Compute the rigid transform from model vertices to source vertices
                m_icpModelToSourceTransform= 
                    RigidMotionEstimator::point_to_point(icpCorrespondingModelVertices, icpSourceVertices);
                m_bIsIcpTransformValid= true;
            }
            else if (source_point_count < 1)
            {
                m_bIsIcpTransformValid= false;
            }
        }
        else
        {
            m_bIsIcpTransformValid= false;
        }
    }

    void render2DState(const PSMTracker *trackerView, const float gl_top_y, const float gl_bottom_y) const
    {
        PSMVector2f tracker_size;
        PSM_GetTrackerScreenSize(trackerView->tracker_info.tracker_id, &tracker_size);

        PSMTrackerIntrinsics intrinsics;
        PSM_GetTrackerIntrinsics(trackerView->tracker_info.tracker_id, &intrinsics);
        assert(intrinsics.intrinsics_type == PSMTrackerIntrinsics::PSM_STEREO_TRACKER_INTRINSICS);
        const PSMStereoTrackerIntrinsics &stereo_intrinsics= intrinsics.intrinsics.stereo;

        const float top_y_fraction= 1.f - ((gl_top_y + 1.f) / 2.f); // [1,-1] -> [0, 1]
        const float bottom_y_fraction= 1.f - ((gl_bottom_y + 1.f) / 2.f); // [1,-1] -> [0, 1]

        const float midX= 0.5f*tracker_size.x;
        const float rightX= tracker_size.x-1.f;
        const float topY= top_y_fraction*(tracker_size.y-1.f);
        const float bottomY= bottom_y_fraction*(tracker_size.y-1.f);

        const float leftX0= 0.f, leftY0= topY;
        const float leftX1= midX, leftY1= bottomY;

        const float rightX0= midX, rightY0= topY;
        const float rightX1= rightX, rightY1= bottomY;

        PSMVector2f left_projections[k_max_projection_points];
        PSMVector2f right_projections[k_max_projection_points];
        PSMVector2f correlation_lines[2*k_max_projection_points];
        int point_count = static_cast<int>(m_lastCorrelatedPoints.size());

        // Draw the label for the correlation line showing disparity and z-values
        for (int point_index = 0; point_index < point_count; ++point_index)
        {
            const CorrelatedPixelPair &pair= m_lastCorrelatedPoints[point_index];
            const PSMVector3f worldPosition= eigen_vector3f_to_psm_vector3f(pair.triangulated_point_cm);

            left_projections[point_index] = 
                projectTrackerRelativePositionOnTracker(
                    worldPosition, 
                    stereo_intrinsics.left_camera_matrix,
                    stereo_intrinsics.left_distortion_coefficients);
            right_projections[point_index] = 
                projectTrackerRelativePositionOnTracker(
                    worldPosition, 
                    stereo_intrinsics.right_camera_matrix,
                    stereo_intrinsics.right_distortion_coefficients);
            
            const PSMVector2f remappedLeftPixel= remapPointIntoSubWindow(
                tracker_size.x, tracker_size.y,
                leftX0, leftY0,
                leftX1, leftY1,
                pair.left_pixel);
            const PSMVector2f remappedRightPixel= remapPointIntoSubWindow(
                tracker_size.x, tracker_size.y,
                rightX0, rightY0,
                rightX1, rightY1,
                pair.right_pixel);

            correlation_lines[2*point_index]= remappedLeftPixel;
            correlation_lines[2*point_index+1]= remappedRightPixel;

            drawTextAtScreenPosition(
                glm::vec3(remappedLeftPixel.x, remappedLeftPixel.y, 0.5f),
                "d:%.1f, z:%.1f",
                pair.disparity, pair.triangulated_point_cm.z());
        }

        // Draw the correlation lines between the left and right pixels
        drawLineList2d(
            tracker_size.x, tracker_size.y, 
            glm::vec3(1.f, 1.f, 0.f), 
            reinterpret_cast<float *>(correlation_lines), point_count*2);

        //TODO:
        // Draw the projections of the triangulated points back onto the sub windows
        //drawPointCloudProjectionInSubWindow(
        //    tracker_size.x, tracker_size.y, 
        //    leftX0, leftY0,
        //    leftX1, leftY1,
        //    glm::vec3(0.f, 1.f, 0.f), 
        //    left_projections, point_count, 6.f);
        //drawPointCloudProjectionInSubWindow(
        //    tracker_size.x, tracker_size.y, 
        //    rightX0, rightY0,
        //    rightX1, rightY1,
        //    glm::vec3(0.f, 1.f, 0.f), 
        //    right_projections, point_count, 6.f);
    }

    void render3DState(const PSMTracker *trackerView, const PSMHeadMountedDisplay *hmdView)
    {
        // Draw the origin axes
        drawTransformedAxes(glm::mat4(1.0f), 100.f);

        // Draw the last triangulated point cloud
        const int correlated_point_count = static_cast<int>(m_lastCorrelatedPoints.size());
        if (correlated_point_count > 0)
        {
            PSMVector3f triangulated_points[k_max_projection_points];

            for (int point_index = 0; point_index < correlated_point_count; ++point_index)
            {
                const CorrelatedPixelPair &pair= m_lastCorrelatedPoints[point_index];
            
                triangulated_points[point_index]= eigen_vector3f_to_psm_vector3f(pair.triangulated_point_cm);
            }

            drawPointCloud(glm::mat4(1.f), glm::vec3(1.f, 1.f, 0.f), (float *)triangulated_points, correlated_point_count);
        }

        // Draw the frustum for each tracking camera.
        // The frustums are defined in PSMove tracking space.
        // We need to transform them into chaperone space to display them along side the HMD.
        if (trackerView != nullptr)
        {
            const PSMPosef psm_pose = trackerView->tracker_info.tracker_pose;
            const glm::mat4 glm_pose = psm_posef_to_glm_mat4(psm_pose);

            PSMFrustum frustum;
            PSM_GetTrackerFrustum(trackerView->tracker_info.tracker_id, &frustum);

            // use color depending on tracking status
            glm::vec3 color;
            bool bIsTracking;
            if (PSM_GetIsHmdTracking(hmdView->HmdID, &bIsTracking) == PSMResult_Success)
            {
                if (bIsTracking && 
                    PSM_GetHmdPixelLocationOnTracker(hmdView->HmdID, LEFT_PROJECTION_INDEX, nullptr, nullptr) == PSMResult_Success &&
                    PSM_GetHmdPixelLocationOnTracker(hmdView->HmdID, RIGHT_PROJECTION_INDEX, nullptr, nullptr) == PSMResult_Success)
                {
                    color = k_psmove_frustum_color;
                }
                else
                {
                    color = k_psmove_frustum_color_no_track;
                }
            }
            else
            {
                color = k_psmove_frustum_color_no_track;
            }
            drawTransformedFrustum(glm::mat4(1.f), &frustum, color);

            drawTransformedAxes(glm_pose, 20.f);
        }

        glm::mat4 hmdTransform;
        if (getHmdTransform(hmdTransform))
        {
            // Draw the model points at the HMD transform
            const int model_point_count= m_icpModelVertices.cols();
            PSMVector3f model_points[k_max_projection_points];
            for (int point_index = 0; point_index < model_point_count; ++point_index)
            {
                const Eigen::Vector3f model_point= m_icpModelVertices.col(point_index).cast<float>();

                model_points[point_index]= eigen_vector3f_to_psm_vector3f(model_point);
            }
            drawPointCloud(hmdTransform, glm::vec3(1.f, 0.f, 0.f), (float *)model_points, model_point_count);

            // Draw the Morpheus model
            drawHMD(hmdView, hmdTransform);
            drawTransformedAxes(hmdTransform, 10.f);
        }
    }

protected:
    struct CorrelatedPixelPair
    {
        PSMVector2f left_pixel;
        PSMVector2f right_pixel;
        float disparity;
        Eigen::Vector3f triangulated_point_cm;
    };

    bool addPointToLedModel(const int led_index, const Eigen::Vector3f &point)
    {
        bool bAddedPoint = false;
        LEDModelSamples &ledModel = m_ledSampleSet[led_index];

        if (ledModel.add_position(point))
        {
            ++m_totalLEDSampleCount;
            bAddedPoint = true;
        }

        return bAddedPoint;
    }

    bool addLedModel(const Eigen::Vector3f &initial_point)
    {
        bool bAddedLed = false;

        if (m_seenLEDCount < m_expectedLEDCount)
        {
            if (addPointToLedModel(m_seenLEDCount, initial_point))
            {
                ++m_seenLEDCount;
                bAddedLed = true;
            }
        }

        return bAddedLed;
    }

    void rebuildModelVertices()
    {
        m_icpModelVertices.resize(Eigen::NoChange, m_seenLEDCount);

        for (int led_index = 0; led_index < m_seenLEDCount; ++led_index)
        {
            const Eigen::Vector3f &ledSample= m_ledSampleSet[led_index].average_position;

            m_icpModelVertices(0, led_index) = ledSample.x();
            m_icpModelVertices(1, led_index) = ledSample.y();
            m_icpModelVertices(2, led_index) = ledSample.z();
        }
    }

    inline PSMVector2f computeCentroid2d(const PSMVector2f *points, const int point_count)
    {
        PSMVector2f center= {0.f, 0.f};

        for (int point_index = 0; point_index < point_count; ++point_index)
        {
            center= PSM_Vector2fAdd(&center, &points[point_index]);
        }
        
        center= PSM_Vector2fSafeScalarDivide(&center, (float)point_count, k_psm_float_vector2_zero);

        return center;
    }

    inline void translatePoints2d(
        const PSMVector2f *in_points,
        const int point_count,
        const PSMVector2f *direction, 
        const float scale, 
        PSMVector2f *out_points)
    {
        for (int point_index = 0; point_index < point_count; ++point_index)
        {
            out_points[point_index]= PSM_Vector2fScaleAndAdd(&in_points[point_index], scale, direction);
        }
    }

    inline int findClosestPoint2d(
        const PSMVector2f *test_point, 
        const PSMVector2f *points, 
        const int point_count)
    {
        int best_index= -1;
        float best_sqrd_dist= k_real_max;

        for (int test_index = 0; test_index < point_count; ++test_index)
        {
            const float sqrd_dist= PSM_Vector2fDistanceSquared(test_point, &points[test_index]);

            if (sqrd_dist < best_sqrd_dist)
            {
                best_index= test_index;
                best_sqrd_dist= sqrd_dist;
            }
        }

        return best_index;
    }

    typedef std::pair<int, int> t_point_index_pair;

    void findAllEpipolarCorrelations(
        const StereoCameraState *stereo_tracker_state,
        const PSMTrackingProjection *trackingProjection,
        std::vector<t_point_index_pair> &out_epipolar_pairs)
    {
        // Undistorted/rectified points
        const PSMVector2f *leftPoints = trackingProjection->projections[LEFT_PROJECTION_INDEX].shape.pointcloud.points;
        const PSMVector2f *rightPoints = trackingProjection->projections[RIGHT_PROJECTION_INDEX].shape.pointcloud.points;
        const int leftPointCount = trackingProjection->projections[LEFT_PROJECTION_INDEX].shape.pointcloud.point_count;
        const int rightPointCount = trackingProjection->projections[RIGHT_PROJECTION_INDEX].shape.pointcloud.point_count;

        // For each point in one tracking projection A, 
        // try and find the corresponding point in the projection B
        for (int leftPointIndex = 0; leftPointIndex < leftPointCount; ++leftPointIndex)
        {
            const PSMVector2f &leftPoint = leftPoints[leftPointIndex];
            const cv::Mat cvLeftPoint = cv::Mat(cv::Point2f(leftPoint.x, leftPoint.y));

            for (int rightPointIndex = 0; rightPointIndex < rightPointCount; ++rightPointIndex)
            {
                const PSMVector2f &rightPoint = rightPoints[rightPointIndex];
                cv::Mat cvRightPoint = cv::Mat(cv::Point2f(rightPoint.x, rightPoint.y));

                if (stereo_tracker_state->do_points_correspond(cvLeftPoint, cvRightPoint, stereo_tracker_state->tolerance))
                {
                    out_epipolar_pairs.push_back(t_point_index_pair(leftPointIndex, rightPointIndex));
                }
            }
        }
    }

    int align2DPointCloudsUsingCorrelation(
        const StereoCameraState *stereo_tracker_state,
        const PSMTrackingProjection *trackingProjection,
        const std::vector<t_point_index_pair> &epipolar_pairs,
        const t_point_index_pair &test_correlation,
        std::vector<int> &out_left_to_right_point_correspondence,
        float &out_matching_error)
    {
        // Undistorted/rectified points
        const PSMVector2f *leftPoints = trackingProjection->projections[LEFT_PROJECTION_INDEX].shape.pointcloud.points;
        const PSMVector2f *rightPoints = trackingProjection->projections[RIGHT_PROJECTION_INDEX].shape.pointcloud.points;
        const int leftPointCount = trackingProjection->projections[LEFT_PROJECTION_INDEX].shape.pointcloud.point_count;
        const int rightPointCount = trackingProjection->projections[RIGHT_PROJECTION_INDEX].shape.pointcloud.point_count;

        // Treat the correlation points to the origin
        const PSMVector2f leftOrigin= leftPoints[test_correlation.first];
        const PSMVector2f rightOrigin= rightPoints[test_correlation.second];

        // Recenter the projection points about the origin
        PSMVector2f recenteredLeftPoints[MAX_POINT_CLOUD_POINT_COUNT];
        PSMVector2f recenteredRightPoints[MAX_POINT_CLOUD_POINT_COUNT];
        translatePoints2d(leftPoints, leftPointCount, &leftOrigin, -1.f, recenteredLeftPoints);
        translatePoints2d(rightPoints, rightPointCount, &rightOrigin, -1.f, recenteredRightPoints);

        // Initialize the correspondence tables
        std::vector<int> left_to_right_point_correspondence(leftPointCount);
        std::vector<int> right_to_left_point_correspondence(rightPointCount);
        std::vector<float> point_distance(leftPointCount);
        std::fill(left_to_right_point_correspondence.begin(), left_to_right_point_correspondence.end(), -1);
        std::fill(right_to_left_point_correspondence.begin(), right_to_left_point_correspondence.end(), -1);
        std::fill(point_distance.begin(), point_distance.end(), k_real_max);

        // For each point in the left tracking projection, 
        // try and find the corresponding point in the right tracking projection
        for (int leftPointIndex = 0; leftPointIndex < leftPointCount; ++leftPointIndex)
        {
            const PSMVector2f recenteredLeftPoint = recenteredLeftPoints[leftPointIndex];

            // Find the closest point on the same epipolar line
            int bestRightPointIndex= -1;
            PSMVector2f bestRightPoint;
            float bestSqrdDist= k_real_max;
            for (const t_point_index_pair &epipolar_pair : epipolar_pairs)
            {
                if (epipolar_pair.first == leftPointIndex)
                {
                    const int rightPointIndex= epipolar_pair.second;
                    const PSMVector2f recenteredRightPoint = recenteredRightPoints[rightPointIndex];
                    const float sqrdDist= PSM_Vector2fDistanceSquared(&recenteredLeftPoint, &recenteredRightPoint);

                    if (sqrdDist < bestSqrdDist)
                    {
                        bestRightPointIndex= rightPointIndex;
                        bestRightPoint= recenteredRightPoint;
                        bestSqrdDist= sqrdDist;
                    }
                }
            }

            if (bestRightPointIndex != -1)
            {
                // Associate the left point with the right point
                left_to_right_point_correspondence[leftPointIndex]= bestRightPointIndex;
                point_distance[leftPointIndex]= bestSqrdDist;
            }
        }

        // For each point in the right tracking projection, 
        // try and find the corresponding point in the left tracking projection
        for (int rightPointIndex = 0; rightPointIndex < rightPointCount; ++rightPointIndex)
        {
            const PSMVector2f recenteredRightPoint = recenteredRightPoints[rightPointIndex];

            // Find the closest point on the same epipolar line
            int bestLeftPointIndex= -1;
            float bestSqrdDist= k_real_max;
            for (const t_point_index_pair &epipolar_pair : epipolar_pairs)
            {
                if (epipolar_pair.second == rightPointIndex)
                {
                    const int leftPointIndex= epipolar_pair.first;
                    const PSMVector2f recenteredLeftPoint = recenteredLeftPoints[leftPointIndex];
                    const float sqrdDist= PSM_Vector2fDistanceSquared(&recenteredLeftPoint, &recenteredRightPoint);

                    if (sqrdDist < bestSqrdDist)
                    {
                        bestLeftPointIndex= leftPointIndex;
                        bestSqrdDist= sqrdDist;
                    }
                }
            }

            if (bestLeftPointIndex != -1)
            {
                // Associate the left point with the right point
                right_to_left_point_correspondence[rightPointIndex]= bestLeftPointIndex;
            }
        }

        // Only consider left and right points in correspondance
        // if they agree they correspond with each other
        int matched_points= 0;
        out_matching_error= 0.f;
        out_left_to_right_point_correspondence.resize(leftPointCount);
        for (int leftPointIndex = 0; leftPointIndex < leftPointCount; ++leftPointIndex)
        {
            const int rightPointIndex= left_to_right_point_correspondence[leftPointIndex];
            
            if (rightPointIndex != -1 && 
                right_to_left_point_correspondence[rightPointIndex] == leftPointIndex)
            {
                out_left_to_right_point_correspondence[leftPointIndex]= rightPointIndex;
                out_matching_error+= point_distance[leftPointIndex];
                matched_points++;
            }
            else
            {
                out_left_to_right_point_correspondence[leftPointIndex]= -1;
            }
        }

        return matched_points;
    }

    bool findBest2DPointCloudCorrelation(
        const StereoCameraState *stereo_tracker_state,
        const PSMTrackingProjection *trackingProjection,
        std::vector<int> &bestLeftToRightPointCorrespondence)
    {
        std::vector<t_point_index_pair> epipolar_pairs;
        findAllEpipolarCorrelations(stereo_tracker_state, trackingProjection, epipolar_pairs);

        bool bFoundCorrelation= false;
        float bestCorrelationError= k_real_max;
        int bestMatchCount= 0;
        for (const t_point_index_pair &test_epipolar_pair : epipolar_pairs)
        {
            std::vector<int> testLeftToRightPointCorrespondence;
            float correlationError= 0.f;
            int matchCount = align2DPointCloudsUsingCorrelation(
                stereo_tracker_state,
                trackingProjection,
                epipolar_pairs,
                test_epipolar_pair,
                testLeftToRightPointCorrespondence,
                correlationError);

            if (matchCount > bestMatchCount ||
                (matchCount == bestMatchCount && correlationError < bestCorrelationError))
            {
                bestLeftToRightPointCorrespondence= testLeftToRightPointCorrespondence;
                bestCorrelationError= correlationError;
                bestMatchCount= matchCount;
                bFoundCorrelation= true;
            }
        }

        return bFoundCorrelation;
    }

    bool triangulateHMDProjections(
        PSMHeadMountedDisplay *hmd_view, 
        StereoCameraState *stereo_tracker_state,
        std::vector<CorrelatedPixelPair> &out_triangulated_points)
    {
        const PSMTracker *TrackerView = stereo_tracker_state->trackerView;
        PSMTrackingProjection trackingProjection;
        
        out_triangulated_points.clear();
        
        // Triangulate tracking LEDs that both cameras can see
        bool bIsTracking= false;
        if (PSM_GetIsHmdTracking(hmd_view->HmdID, &bIsTracking) == PSMResult_Success)
        {
            PSMTrackerID selected_tracker_id= -1;

            if (bIsTracking &&
                PSM_GetHmdProjectionOnTracker(hmd_view->HmdID, &selected_tracker_id, &trackingProjection) == PSMResult_Success)
            {
                assert(trackingProjection.shape_type == PSMShape_PointCloud);
                assert(trackingProjection.projection_count == STEREO_PROJECTION_COUNT);

                // Undistorted/rectified points
                const PSMVector2f *leftPoints = trackingProjection.projections[LEFT_PROJECTION_INDEX].shape.pointcloud.points;
                const PSMVector2f *rightPoints = trackingProjection.projections[RIGHT_PROJECTION_INDEX].shape.pointcloud.points;
                const int leftPointCount = trackingProjection.projections[LEFT_PROJECTION_INDEX].shape.pointcloud.point_count;
                const int rightPointCount = trackingProjection.projections[RIGHT_PROJECTION_INDEX].shape.pointcloud.point_count;

                // Find the best fit correlation between the left and right point clouds
                std::vector<int> bestLeftToRightPointCorrespondence;
                if (findBest2DPointCloudCorrelation(
                        stereo_tracker_state,
                        &trackingProjection,
                        bestLeftToRightPointCorrespondence))
                {
                    // Use the correspondence table to make correlated-pixel-pairs
                    // for every pair that satisfies the epipolar distance constraint
                    for (int leftPointIndex = 0; leftPointIndex < leftPointCount; ++leftPointIndex)
                    {
                        const int rightPointIndex= bestLeftToRightPointCorrespondence[leftPointIndex];
                        if (rightPointIndex == -1)
                            continue;
                    
                        const PSMVector2f &leftPoint = leftPoints[leftPointIndex];
                        const cv::Mat cvLeftPoint = cv::Mat(cv::Point2f(leftPoint.x, leftPoint.y));

                        const PSMVector2f &rightPoint = rightPoints[rightPointIndex];
                        cv::Mat cvRightPoint = cv::Mat(cv::Point2f(rightPoint.x, rightPoint.y));

                        // Compute the horizontal pixel disparity between the left and right corresponding pixels
                        const double disparity= (double)(leftPoint.x - rightPoint.x);

                        if (fabs(disparity) > k_real64_epsilon)
                        {
                            // Project the left pixel + disparity into the world using
                            // the projection matrix 'Q' computed during stereo calibration
                            Eigen::Vector4d pixel((double)leftPoint.x, (double)leftPoint.y, disparity, 1.0);
                            Eigen::Vector4d homogeneus_point= stereo_tracker_state->Q * pixel;

                            // Get the triangulated 3d position
                            const double w = homogeneus_point.w();

                            if (fabs(w) > k_real64_epsilon)
                            {
                                CorrelatedPixelPair pair;
                                pair.left_pixel= leftPoint;
                                pair.right_pixel= rightPoint;
                                pair.disparity= (float)disparity;

                                // Q matrix transforms pixels to tracker relative positions in millimeters
                                Eigen::Vector3f triangulated_point_mm= Eigen::Vector3f(
                                    (float)(homogeneus_point.x() / w),
                                    (float)(-homogeneus_point.y() / w), // Q matrix has flipped Y-axis
                                    (float)(homogeneus_point.z() / w));
                                pair.triangulated_point_cm= triangulated_point_mm * PSM_MILLIMETERS_TO_CENTIMETERS;

                                // Add to the list of world space points we saw this frame
                                out_triangulated_points.push_back(pair);
                            }
                        }
                    }
                }
            } 
        }

        return out_triangulated_points.size() >= 3;
    }

private:
    PSMTrackingShape m_currentTrackingShape;

    std::vector<CorrelatedPixelPair> m_lastCorrelatedPoints;
    int m_expectedLEDCount;
    int m_seenLEDCount;
    int m_totalLEDSampleCount;

    LEDModelSamples *m_ledSampleSet;

    SICP::Vertices m_icpModelVertices;
    nanoflann::KDTreeAdaptor<SICP::Vertices, 3, nanoflann::metric_L2_Simple> *m_icpModelKDTree;
    Eigen::Affine3d m_icpModelToSourceTransform;
    bool m_bIsIcpTransformValid;
};

//-- public methods -----
AppStage_HMDModelCalibration::AppStage_HMDModelCalibration(App *app)
    : AppStage(app)
    , m_menuState(AppStage_HMDModelCalibration::inactive)
    , m_bBypassCalibration(false)
    , m_stereoTrackerState(new StereoCameraState)
    , m_hmdModelState(nullptr)
    , m_hmdView(nullptr)
{
    m_stereoTrackerState->init();
}

AppStage_HMDModelCalibration::~AppStage_HMDModelCalibration()
{
    delete m_stereoTrackerState;

    if (m_hmdModelState != nullptr)
    {
        delete m_hmdModelState;
    }
}

void AppStage_HMDModelCalibration::enterStageAndCalibrate(App *app, int requested_hmd_id)
{
    app->getAppStage<AppStage_HMDModelCalibration>()->m_bBypassCalibration = false;
    app->getAppStage<AppStage_HMDModelCalibration>()->m_overrideHmdId = requested_hmd_id;
    app->setAppStage(AppStage_HMDModelCalibration::APP_STAGE_NAME);
}

void AppStage_HMDModelCalibration::enterStageAndSkipCalibration(App *app, int requested_hmd_id)
{
    app->getAppStage<AppStage_HMDModelCalibration>()->m_bBypassCalibration = true;
    app->getAppStage<AppStage_HMDModelCalibration>()->m_overrideHmdId = requested_hmd_id;
    app->setAppStage(AppStage_HMDModelCalibration::APP_STAGE_NAME);
}

void AppStage_HMDModelCalibration::enter()
{
    // Kick off this async request chain 
    // hmd list request
    // -> get hmd tracking shape 
    // -> hmd start request
    // -> tracker list request
    // -> tracker start request
    request_hmd_list();

    m_app->setCameraType(_cameraFixed);
}

void AppStage_HMDModelCalibration::exit()
{
    release_devices();

    setState(eMenuState::inactive);
}

void AppStage_HMDModelCalibration::update()
{
    switch (m_menuState)
    {
    case eMenuState::inactive:
        break;
    case eMenuState::pendingHmdListRequest:
    case eMenuState::pendingHmdShapeRequest:
    case eMenuState::pendingHmdStartRequest:
    case eMenuState::pendingTrackerListRequest:
    case eMenuState::pendingTrackerStartRequest:
        break;
    case eMenuState::failedHmdListRequest:
    case eMenuState::failedHmdShapeRequest:
    case eMenuState::failedHmdStartRequest:
    case eMenuState::failedTrackerListRequest:
    case eMenuState::failedTrackerStartRequest:
        break;
    case eMenuState::verifyTrackers:
        update_tracker_video();
        break;
    case eMenuState::calibrate:
        {
            update_tracker_video();

            if (!m_hmdModelState->getIsComplete())
            {
                m_hmdModelState->recordSamples(m_hmdView, m_stereoTrackerState);
                if (m_hmdModelState->getIsComplete())
                {
                    setState(eMenuState::test);
                }
            }
            else
            {
                request_set_hmd_led_model_calibration();
                setState(eMenuState::test);
            }
        }
        break;
    case eMenuState::test:
        {
            update_tracker_video();
            m_hmdModelState->computeHmdTransform(m_hmdView, m_stereoTrackerState);
        } break;
    default:
        assert(0 && "unreachable");
    }
}

void AppStage_HMDModelCalibration::render()
{
    switch (m_menuState)
    {
    case eMenuState::inactive:
        break;
    case eMenuState::pendingHmdListRequest:
    case eMenuState::pendingHmdShapeRequest:
    case eMenuState::pendingHmdStartRequest:
    case eMenuState::pendingTrackerListRequest:
    case eMenuState::pendingTrackerStartRequest:
        break;
    case eMenuState::failedHmdListRequest:
    case eMenuState::failedHmdShapeRequest:
    case eMenuState::failedHmdStartRequest:
    case eMenuState::failedTrackerListRequest:
    case eMenuState::failedTrackerStartRequest:
        break;
    case eMenuState::verifyTrackers:
        {
            render_tracker_video(0.5f, -0.5f);
        } break;
    case eMenuState::calibrate:
        {
            const PSMTracker *TrackerView = m_stereoTrackerState->trackerView;

            // Draw the video from the PoV of the current tracker
            render_tracker_video(0.5f, -0.5f);

            // Draw the LED tracking centroids and correlation lines
            m_hmdModelState->render2DState(TrackerView, 0.5f, -0.5f);

        } break;
    case eMenuState::test:
        {
            const PSMTracker *trackerView = m_stereoTrackerState->trackerView;

            switch (m_testRenderMode)
            {
            case eTrackingTestRenderMode::renderMode2d:
                {
                    // Draw the video from the PoV of the current tracker
                    render_tracker_video(0.5f, -0.5f);

                    // Draw the LED tracking centroids and correlation lines
                    m_hmdModelState->render2DState(trackerView, 0.5f, -0.5f);
                } break;
            case eTrackingTestRenderMode::renderMode3d:
                {
                    // Draw the triangulated LED locations and the estimated HMD pose
                    m_hmdModelState->render3DState(trackerView, m_hmdView);
                } break;
            }
        } break;
    default:
        assert(0 && "unreachable");
    }
}

void AppStage_HMDModelCalibration::renderUI()
{
    const float k_panel_width = 300.f;
    const char *k_window_title = "Compute HMD Model";
    const ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_ShowBorders |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse;

    switch (m_menuState)
    {
    case eMenuState::inactive:
        break;

    case eMenuState::pendingHmdListRequest:
    case eMenuState::pendingHmdShapeRequest:
    case eMenuState::pendingHmdStartRequest:
    case eMenuState::pendingTrackerListRequest:
    case eMenuState::pendingTrackerStartRequest:
    {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_panel_width / 2.f, 20.f));
        ImGui::SetNextWindowSize(ImVec2(k_panel_width, 80));
        ImGui::Begin(k_window_title, nullptr, window_flags);

        ImGui::Text("Pending device initialization...");

        if (ImGui::Button("Return to HMD Settings"))
        {
            request_exit_to_app_stage(AppStage_HMDModelCalibration::APP_STAGE_NAME);
        }

        ImGui::End();
    } break;

    case eMenuState::failedHmdListRequest:
    case eMenuState::failedHmdShapeRequest:
    case eMenuState::failedHmdStartRequest:
    case eMenuState::failedTrackerListRequest:
    case eMenuState::failedTrackerStartRequest:
    {
        ImGui::SetNextWindowPosCenter();
        ImGui::SetNextWindowSize(ImVec2(k_panel_width, 180));
        ImGui::Begin(k_window_title, nullptr, window_flags);

        switch (m_menuState)
        {
        case eMenuState::failedHmdListRequest:
            ImGui::Text("Failed hmd list retrieval!");
            break;
        case eMenuState::failedHmdShapeRequest:
            ImGui::Text("Failed hmd shape retrieval!");
            break;
        case eMenuState::failedHmdStartRequest:
            ImGui::Text("Failed hmd stream start!");
            break;
        case eMenuState::failedTrackerListRequest:
            {
                const char * szFailure = m_failureDetails.c_str();
                ImGui::Text("Failed tracker list retrieval:");
                ImGui::Text(szFailure);
            }
            break;
        case eMenuState::failedTrackerStartRequest:
            ImGui::Text("Failed tracker stream start!");
            break;
        }

        if (ImGui::Button("Ok"))
        {
            request_exit_to_app_stage(AppStage_HMDSettings::APP_STAGE_NAME);
        }

        if (ImGui::Button("Return to Main Menu"))
        {
            request_exit_to_app_stage(AppStage_MainMenu::APP_STAGE_NAME);
        }

        ImGui::End();
    } break;

    case eMenuState::verifyTrackers:
    {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - 500.f / 2.f, 20.f));
        ImGui::SetNextWindowSize(ImVec2(500.f, 100.f));
        ImGui::Begin(k_window_title, nullptr, window_flags);

        ImGui::Text("Verify that your stereo camera can see your HMD");
        ImGui::Separator();

        if (ImGui::Button("Looks Good!"))
        {
            if (!m_bBypassCalibration)
            {
                setState(eMenuState::calibrate);
            }
            else
            {
                setState(eMenuState::test);
            }
        }

        if (ImGui::Button("Hmm... Something is wrong."))
        {
            request_exit_to_app_stage(AppStage_HMDSettings::APP_STAGE_NAME);
        }

        ImGui::End();
    } break;

    case eMenuState::calibrate:
    {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_panel_width / 2.f, 20.f));
        ImGui::SetNextWindowSize(ImVec2(k_panel_width, 180));
        ImGui::Begin(k_window_title, nullptr, window_flags);

        // TODO: Show calibration progress
        ImGui::ProgressBar(m_hmdModelState->getProgressFraction(), ImVec2(250, 20));

        // display tracking quality
        if (m_stereoTrackerState != nullptr)
        {
            const PSMTracker *trackerView = m_stereoTrackerState->trackerView;

            bool bIsTracking;
            if (PSM_GetIsHmdTracking(m_hmdView->HmdID, &bIsTracking) == PSMResult_Success)
            {
                if (bIsTracking && 
                    PSM_GetHmdPixelLocationOnTracker(m_hmdView->HmdID, LEFT_PROJECTION_INDEX, nullptr, nullptr) == PSMResult_Success &&
                    PSM_GetHmdPixelLocationOnTracker(m_hmdView->HmdID, RIGHT_PROJECTION_INDEX, nullptr, nullptr) == PSMResult_Success)
                {
                    ImGui::Text("Tracking OK");
                }
                else
                {
                    ImGui::Text("Tracking FAIL");
                }
            }
            else
            {
                ImGui::Text("Tracking FAIL");
            }
        }

        ImGui::SliderFloat("Tolerance", &m_stereoTrackerState->tolerance, 0.f, 1.f);

        ImGui::Separator();

        if (ImGui::Button("Restart Calibration"))
        {
            setState(eMenuState::verifyTrackers);
        }
        ImGui::SameLine();
        if (ImGui::Button("Exit"))
        {
            m_app->setAppStage(AppStage_HMDSettings::APP_STAGE_NAME);
        }

        ImGui::End();
    } break;

    case eMenuState::test:
    {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_panel_width / 2.f, 20.f));
        ImGui::SetNextWindowSize(ImVec2(k_panel_width, 130));
        ImGui::Begin("Test HMD Model", nullptr, window_flags);

        if (!m_bBypassCalibration)
        {
            ImGui::Text("Calibration Complete");

            if (ImGui::Button("Redo Calibration"))
            {
                setState(eMenuState::verifyTrackers);
            }
        }

        switch (m_testRenderMode)
        {
        case eTrackingTestRenderMode::renderMode2d:
            if (ImGui::Button("Switch to 3d View"))
            {
                m_testRenderMode= eTrackingTestRenderMode::renderMode3d;
            }
            break;
        case eTrackingTestRenderMode::renderMode3d:
            if (ImGui::Button("Switch to 2d View"))
            {
                m_testRenderMode= eTrackingTestRenderMode::renderMode2d;
            }
            break;
        default:
            break;
        }
        // display tracking quality
        if (m_stereoTrackerState != nullptr)
        {
            const PSMTracker *trackerView = m_stereoTrackerState->trackerView;

            bool bIsTracking;
            if (PSM_GetIsHmdTracking(m_hmdView->HmdID, &bIsTracking) == PSMResult_Success)
            {
                if (bIsTracking && 
                    PSM_GetHmdPixelLocationOnTracker(m_hmdView->HmdID, LEFT_PROJECTION_INDEX, nullptr, nullptr) == PSMResult_Success &&
                    PSM_GetHmdPixelLocationOnTracker(m_hmdView->HmdID, RIGHT_PROJECTION_INDEX, nullptr, nullptr) == PSMResult_Success)
                {
                    ImGui::Text("Tracking OK");
                }
                else
                {
                    ImGui::Text("Tracking FAIL");
                }
            }
            else
            {
                ImGui::Text("Tracking FAIL");
            }
        }
        ImGui::Text("");

        if (ImGui::Button("Exit"))
        {
            m_app->setAppStage(AppStage_HMDModelCalibration::APP_STAGE_NAME);
        }

        ImGui::End();
    }
    break;

    default:
        assert(0 && "unreachable");
    }
}

void AppStage_HMDModelCalibration::setState(AppStage_HMDModelCalibration::eMenuState newState)
{
    if (newState != m_menuState)
    {
        onExitState(m_menuState);
        onEnterState(newState);

        m_menuState = newState;
    }
}

void AppStage_HMDModelCalibration::onExitState(AppStage_HMDModelCalibration::eMenuState newState)
{
    switch (m_menuState)
    {
    case eMenuState::inactive:
        break;
    case eMenuState::pendingHmdListRequest:
    case eMenuState::pendingHmdShapeRequest:
    case eMenuState::pendingHmdStartRequest:
    case eMenuState::pendingTrackerListRequest:
    case eMenuState::pendingTrackerStartRequest:
        break;
    case eMenuState::failedHmdListRequest:
    case eMenuState::failedHmdShapeRequest:
    case eMenuState::failedHmdStartRequest:
    case eMenuState::failedTrackerListRequest:
    case eMenuState::failedTrackerStartRequest:
        break;
    case eMenuState::verifyTrackers:
        break;
    case eMenuState::calibrate:
        break;
    case eMenuState::test:
        m_app->setCameraType(_cameraFixed);
        break;
    default:
        assert(0 && "unreachable");
    }
}

void AppStage_HMDModelCalibration::onEnterState(AppStage_HMDModelCalibration::eMenuState newState)
{
    switch (newState)
    {
    case eMenuState::inactive:
        break;
    case eMenuState::pendingHmdListRequest:
    case eMenuState::pendingHmdShapeRequest:
    case eMenuState::pendingHmdStartRequest:
    case eMenuState::pendingTrackerListRequest:
        m_stereoTrackerState->init();
        m_failureDetails = "";
        break;
    case eMenuState::pendingTrackerStartRequest:
        break;
    case eMenuState::failedHmdListRequest:
    case eMenuState::failedHmdShapeRequest:
    case eMenuState::failedHmdStartRequest:
    case eMenuState::failedTrackerListRequest:
    case eMenuState::failedTrackerStartRequest:
        break;
    case eMenuState::verifyTrackers:
        break;
    case eMenuState::calibrate:
        m_app->setCameraType(_cameraFixed);
        break;
    case eMenuState::test:
        m_testRenderMode= eTrackingTestRenderMode::renderMode2d;
        m_app->setCameraType(_cameraOrbit);
        m_app->getOrbitCamera()->reset();
        m_app->getOrbitCamera()->setCameraOrbitRadius(500.f);
        break;
    default:
        assert(0 && "unreachable");
    }
}

void AppStage_HMDModelCalibration::update_tracker_video()
{
    // Render the latest from the currently active tracker
    if (m_stereoTrackerState->trackerView != nullptr &&
        PSM_PollTrackerVideoStream(m_stereoTrackerState->trackerView->tracker_info.tracker_id))
    {
        const unsigned char *buffer= nullptr;

        if (PSM_GetTrackerVideoFrameBuffer(
                m_stereoTrackerState->trackerView->tracker_info.tracker_id, 
                PSMVideoFrameSection_Left, 
                &buffer) == PSMResult_Success)
        {
            m_stereoTrackerState->sections[PSMVideoFrameSection_Left].applyVideoFrame(buffer);
        }

        if (PSM_GetTrackerVideoFrameBuffer(
                m_stereoTrackerState->trackerView->tracker_info.tracker_id, 
                PSMVideoFrameSection_Right, 
                &buffer) == PSMResult_Success)
        {
            m_stereoTrackerState->sections[PSMVideoFrameSection_Right].applyVideoFrame(buffer);
        }
    }
}

void AppStage_HMDModelCalibration::render_tracker_video(const float top_y, const float bottom_y)
{
    if (m_stereoTrackerState->sections[PSMVideoFrameSection_Left].textureAsset != nullptr &&
        m_stereoTrackerState->sections[PSMVideoFrameSection_Left].textureAsset != nullptr)
    {
        // Draw the two video feeds on the top half of the screen
        drawFullscreenStereoTexture(
            m_stereoTrackerState->sections[PSMVideoFrameSection_Left].textureAsset->texture_id, 
            m_stereoTrackerState->sections[PSMVideoFrameSection_Right].textureAsset->texture_id,
            top_y, bottom_y);
    }
}

void AppStage_HMDModelCalibration::release_devices()
{
    //###HipsterSloth $REVIEW Do we care about canceling in-flight requests?

    if (m_hmdModelState != nullptr)
    {
        delete m_hmdModelState;
        m_hmdModelState = nullptr;
    }

    if (m_hmdView != nullptr)
    {
        PSMRequestID request_id;
        PSM_StopHmdDataStreamAsync(m_hmdView->HmdID, &request_id);
        PSM_EatResponse(request_id);

        PSM_FreeHmdListener(m_hmdView->HmdID);
        m_hmdView = nullptr;
    }

    if (m_stereoTrackerState != nullptr)
    {
        m_stereoTrackerState->dispose();

        if (m_stereoTrackerState->trackerView != nullptr)
        {
            PSM_CloseTrackerVideoStream(m_stereoTrackerState->trackerView->tracker_info.tracker_id);

            PSMRequestID request_id;
            PSM_StopTrackerDataStreamAsync(m_stereoTrackerState->trackerView->tracker_info.tracker_id, &request_id);
            PSM_EatResponse(request_id);

            PSM_FreeTrackerListener(m_stereoTrackerState->trackerView->tracker_info.tracker_id);
        }
    }

    m_stereoTrackerState->init();
}

void AppStage_HMDModelCalibration::request_exit_to_app_stage(const char *app_stage_name)
{
    release_devices();

    m_app->setAppStage(app_stage_name);
}

void AppStage_HMDModelCalibration::request_hmd_list()
{
    if (m_menuState != AppStage_HMDModelCalibration::pendingHmdListRequest)
    {
        m_menuState = AppStage_HMDModelCalibration::pendingHmdListRequest;

        // Request a list of controllers back from the server
        PSMRequestID requestId;
        PSM_GetHmdListAsync(&requestId);
        PSM_RegisterCallback(requestId, AppStage_HMDModelCalibration::handle_hmd_list_response, this);
    }
}

void AppStage_HMDModelCalibration::handle_hmd_list_response(
    const PSMResponseMessage *response_message,
    void *userdata)
{
    const PSMoveProtocol::Response *response = GET_PSMOVEPROTOCOL_RESPONSE(response_message->opaque_response_handle);
    const PSMoveProtocol::Request *request = GET_PSMOVEPROTOCOL_REQUEST(response_message->opaque_request_handle);

    AppStage_HMDModelCalibration *thisPtr = static_cast<AppStage_HMDModelCalibration *>(userdata);

    const PSMResult ResultCode = response_message->result_code;


    switch (ResultCode)
    {
    case PSMResult_Success:
    {
        assert(response_message->payload_type == PSMResponseMessage::_responsePayloadType_HmdList);
        const PSMHmdList *hmd_list = &response_message->payload.hmd_list;

        PSMHmdID trackedHmdId = thisPtr->m_overrideHmdId;

        if (trackedHmdId == -1)
        {
            for (int list_index = 0; list_index < hmd_list->count; ++list_index)
            {
                if (hmd_list->hmd_type[list_index] == PSMHmd_Morpheus)
                {
                    trackedHmdId = hmd_list->hmd_id[list_index];
                    break;
                }
            }
        }

        if (trackedHmdId != -1)
        {
            // Allocate an HMD view to track HMD state
            assert(thisPtr->m_hmdView == nullptr);
            PSM_AllocateHmdListener(trackedHmdId);
            thisPtr->m_hmdView = PSM_GetHmd(trackedHmdId);

            // Request the tracking shape so that we know how many LEDs we are looking for
            thisPtr->request_hmd_tracking_shape(trackedHmdId);
        }
        else
        {
            thisPtr->setState(AppStage_HMDModelCalibration::failedHmdListRequest);
        }
    } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
    case PSMResult_Timeout:
        {
            thisPtr->setState(AppStage_HMDModelCalibration::failedHmdListRequest);
        } break;
    }
}

void AppStage_HMDModelCalibration::request_hmd_tracking_shape(PSMHmdID HmdID)
{
    if (m_menuState != AppStage_HMDModelCalibration::pendingHmdShapeRequest)
    {
        m_menuState = AppStage_HMDModelCalibration::pendingHmdShapeRequest;

        PSMRequestID requestId;
        PSM_GetHmdTrackingShapeAsync(HmdID, &requestId);
        PSM_RegisterCallback(requestId, AppStage_HMDModelCalibration::handle_hmd_tracking_shape_response, this);
    }
}

void AppStage_HMDModelCalibration::handle_hmd_tracking_shape_response(
    const PSMResponseMessage *response_message,
    void *userdata)
{
    const PSMoveProtocol::Response *response = GET_PSMOVEPROTOCOL_RESPONSE(response_message->opaque_response_handle);
    const PSMoveProtocol::Request *request = GET_PSMOVEPROTOCOL_REQUEST(response_message->opaque_request_handle);

    AppStage_HMDModelCalibration *thisPtr = static_cast<AppStage_HMDModelCalibration *>(userdata);

    const PSMResult ResultCode = response_message->result_code;

    switch (ResultCode)
    {
    case PSMResult_Success:
    {
        assert(response_message->payload_type == PSMResponseMessage::_responsePayloadType_HmdTrackingShape);
        const PSMTrackingShape *hmd_tracking_shape = &response_message->payload.hmd_tracking_shape;

        if (hmd_tracking_shape->shape_type == PSMTrackingShape_PointCloud)
        {
            // Create a model for the HMD that corresponds to the tracking geometry we are looking for
            assert(thisPtr->m_hmdModelState == nullptr);
            thisPtr->m_hmdModelState = new HMDModelState(hmd_tracking_shape);

            // Start streaming data for the HMD
            thisPtr->request_start_hmd_stream(thisPtr->m_hmdView->HmdID);
        }
        else
        {
            // we only work with point cloud calibration
            thisPtr->setState(AppStage_HMDModelCalibration::failedHmdShapeRequest);
        }
    } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
    case PSMResult_Timeout:
        {
            thisPtr->setState(AppStage_HMDModelCalibration::failedHmdListRequest);
        } break;
    }
}

void AppStage_HMDModelCalibration::request_start_hmd_stream(int HmdID)
{
    // Start receiving data from the controller
    setState(AppStage_HMDModelCalibration::pendingHmdStartRequest);

    PSMRequestID requestId;
    PSM_StartHmdDataStreamAsync(
        HmdID, 
        PSMStreamFlags_includePositionData |
        PSMStreamFlags_includeRawTrackerData |
        PSMStreamFlags_disableROI, 
        &requestId);
    PSM_RegisterCallback(requestId, &AppStage_HMDModelCalibration::handle_start_hmd_response, this);

}

void AppStage_HMDModelCalibration::handle_start_hmd_response(
    const PSMResponseMessage *response_message,
    void *userdata)
{
    AppStage_HMDModelCalibration *thisPtr = static_cast<AppStage_HMDModelCalibration *>(userdata);

    const PSMResult ResultCode = response_message->result_code;

    switch (ResultCode)
    {
    case PSMResult_Success:
    {
        thisPtr->request_tracker_list();
    } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
    case PSMResult_Timeout:
    {
        thisPtr->setState(AppStage_HMDModelCalibration::failedHmdStartRequest);
    } break;
    }
}

void AppStage_HMDModelCalibration::request_tracker_list()
{
    if (m_menuState != eMenuState::pendingTrackerListRequest)
    {
        setState(eMenuState::pendingTrackerListRequest);

        // Tell the psmove service that we we want a list of trackers connected to this machine
        PSMRequestID requestId;
        PSM_GetTrackerListAsync(&requestId);
        PSM_RegisterCallback(requestId, AppStage_HMDModelCalibration::handle_tracker_list_response, this);
    }
}

void AppStage_HMDModelCalibration::handle_tracker_list_response(
    const PSMResponseMessage *response_message,
    void *userdata)
{
    AppStage_HMDModelCalibration *thisPtr = static_cast<AppStage_HMDModelCalibration *>(userdata);

    switch (response_message->result_code)
    {
    case PSMResult_Success:
    {
        assert(response_message->payload_type == PSMResponseMessage::_responsePayloadType_TrackerList);
        const PSMTrackerList &tracker_list = response_message->payload.tracker_list;
        
        if (thisPtr->setup_stereo_tracker(tracker_list))
        {
            thisPtr->setState(eMenuState::pendingTrackerStartRequest);
        }
        else
        {
            thisPtr->setState(eMenuState::failedTrackerListRequest);
        }
    } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
    case PSMResult_Timeout:
        {
            thisPtr->m_failureDetails = "Server Failure";
            thisPtr->setState(eMenuState::failedTrackerListRequest);
        } break;
    }
}

bool AppStage_HMDModelCalibration::setup_stereo_tracker(const PSMTrackerList &tracker_list)
{
    bool bSuccess = true;

    // Find the first stereo camera
    if (bSuccess)
    {
        int stereo_tracker_index= -1;
        for (int tracker_index = 0; tracker_index < tracker_list.count; ++tracker_index)
        {
            const PSMClientTrackerInfo *tracker_info = &tracker_list.trackers[tracker_index];

            if (tracker_info->tracker_intrinsics.intrinsics_type == PSMTrackerIntrinsics::PSM_STEREO_TRACKER_INTRINSICS)
            {
                stereo_tracker_index= tracker_index;
                break;
            }
        }

        if (stereo_tracker_index != -1)
        {
            const PSMClientTrackerInfo *tracker_info = &tracker_list.trackers[stereo_tracker_index];
            const PSMTrackerIntrinsics &intrinsics= tracker_info->tracker_intrinsics;

            // Allocate tracker view for the stereo camera
            PSM_AllocateTrackerListener(tracker_info->tracker_id, tracker_info);

            m_stereoTrackerState->init();
            m_stereoTrackerState->applyIntrinsics(intrinsics);
            m_stereoTrackerState->trackerView= PSM_GetTracker(tracker_info->tracker_id);

            // Start streaming tracker video
            request_tracker_start_stream(m_stereoTrackerState->trackerView);
        }
        else
        {
            m_failureDetails = "Can't find stereo camera!";
            bSuccess = false;
        }
    }

    return bSuccess;
}

void AppStage_HMDModelCalibration::request_tracker_start_stream(
    PSMTracker *tracker_view)
{
    setState(eMenuState::pendingTrackerStartRequest);

    // Request data to start streaming to the tracker
    PSMRequestID requestID;
    PSM_StartTrackerDataStreamAsync(
        tracker_view->tracker_info.tracker_id, 
        &requestID);
    PSM_RegisterCallback(requestID, AppStage_HMDModelCalibration::handle_tracker_start_stream_response, this);
}

void AppStage_HMDModelCalibration::handle_tracker_start_stream_response(
    const PSMResponseMessage *response_message,
    void *userdata)
{
    AppStage_HMDModelCalibration *thisPtr = static_cast<AppStage_HMDModelCalibration *>(userdata);

    switch (response_message->result_code)
    {
    case PSMResult_Success:
    {
        // Get the tracker ID this request was for
        const PSMoveProtocol::Request *request = GET_PSMOVEPROTOCOL_REQUEST(response_message->opaque_request_handle);
        const int tracker_id = request->request_start_tracker_data_stream().tracker_id();

        // Open the shared memory that the video stream is being written to
        if (PSM_OpenTrackerVideoStream(tracker_id) == PSMResult_Success)
        {   
            thisPtr->handle_all_devices_ready();
        }
        else
        {
            thisPtr->setState(eMenuState::failedTrackerStartRequest);
        }
    } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
    case PSMResult_Timeout:
        {
            thisPtr->setState(eMenuState::failedTrackerStartRequest);
        } break;
    }
}

void AppStage_HMDModelCalibration::request_set_hmd_led_model_calibration()
{

}

void AppStage_HMDModelCalibration::handle_all_devices_ready()
{
    setState(eMenuState::verifyTrackers);
}

//-- private methods -----
static PSMVector2f projectTrackerRelativePositionOnTracker(
    const PSMVector3f &trackerRelativePosition,
    const PSMMatrix3d &camera_matrix,
    const PSMDistortionCoefficients &distortion_coefficients)
{
    cv::Mat cvDistCoeffs(5, 1, cv::DataType<double>::type);
    cvDistCoeffs.at<double>(0) = distortion_coefficients.k1;
    cvDistCoeffs.at<double>(1) = distortion_coefficients.k2;
    cvDistCoeffs.at<double>(2) = distortion_coefficients.p1;
    cvDistCoeffs.at<double>(3) = distortion_coefficients.p2;
    cvDistCoeffs.at<double>(4) = distortion_coefficients.k3;
    
    // Use the identity transform for tracker relative positions
    cv::Mat rvec(3, 1, cv::DataType<double>::type, double(0));
    cv::Mat tvec(3, 1, cv::DataType<double>::type, double(0));

    // Only one point to project
    std::vector<cv::Point3d> cvObjectPoints;
    cvObjectPoints.push_back(
        cv::Point3d(
            (double)trackerRelativePosition.x,
            (double)trackerRelativePosition.y,
            (double)trackerRelativePosition.z));

    // Compute the camera intrinsic matrix in opencv format
    cv::Matx33d cvCameraMatrix = psmove_matrix3x3_to_cv_mat33d(camera_matrix);

    // Projected point 
    std::vector<cv::Point2d> projectedPoints;
    cv::projectPoints(cvObjectPoints, rvec, tvec, cvCameraMatrix, cvDistCoeffs, projectedPoints);

    PSMVector2f screenLocation = {(float)projectedPoints[0].x, (float)projectedPoints[0].y};

    return screenLocation;
}

static void drawHMD(const PSMHeadMountedDisplay *hmdView, const glm::mat4 &transform)
{
    switch (hmdView->HmdType)
    {
    case PSMHmd_Morpheus:
        drawMorpheusModel(transform);
        break;
    case PSMHmd_Virtual:
        const glm::mat4 offset_transform = glm::translate(transform, glm::vec3(0.f, 0.f, 10.f));
        drawMorpheusModel(offset_transform);
        //drawVirtualHMDModel(transform, glm::vec3(0.f, 0.f, 1.f));
        break;
    }
}
