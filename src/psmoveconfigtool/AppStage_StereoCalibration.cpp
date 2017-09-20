// Derived From example 11-1 of "Learning OpenCV: Computer Vision with the OpenCV Library" by Gary Bradski

//-- includes -----
#include "AppStage_StereoCalibration.h"
#include "AppStage_TrackerSettings.h"
#include "AppStage_MainMenu.h"
#include "AssetManager.h"
#include "App.h"
#include "Camera.h"
#include "ClientLog.h"
#include "GeometryUtility.h"
#include "MathUtility.h"
#include "Renderer.h"
#include "UIConstants.h"
#include "PSMoveProtocolInterface.h"
#include "PSMoveProtocol.pb.h"
#include "SharedTrackerState.h"

#include "SDL_keycode.h"
#include "SDL_opengl.h"

#include <imgui.h>

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>

#ifdef _MSC_VER
#pragma warning (disable: 4996) // 'This function or variable may be unsafe': snprintf
#define snprintf _snprintf
#endif

//-- statics ----
const char *AppStage_StereoCalibration::APP_STAGE_NAME = "StereoCalibration";

//-- constants -----
enum eVideoDisplayMode
{
    mode_bgr,
    mode_grayscale,
    mode_undistored,

    MAX_VIDEO_DISPLAY_MODES
};

static const char *k_video_display_mode_names[] = {
    "BGR",
    "Grayscale",
    "Undistorted"
};

#define PATTERN_W 9 // Internal corners
#define PATTERN_H 6
#define CORNER_COUNT (PATTERN_W*PATTERN_H)
#define DEFAULT_SQUARE_LEN_MM 24
#define DESIRED_CAPTURE_BOARD_COUNT 12

#define BOARD_MOVED_PIXEL_DIST 5
#define BOARD_MOVED_ERROR_SUM BOARD_MOVED_PIXEL_DIST*CORNER_COUNT

#define BOARD_NEW_LOCATION_PIXEL_DIST 100 
#define BOARD_NEW_LOCATION_ERROR_SUM BOARD_NEW_LOCATION_PIXEL_DIST*CORNER_COUNT

#define STRAIGHT_LINE_TOLERANCE 5 // error tolerance in pixels

//-- typedefs -----
namespace cv
{
    typedef Matx<double, 5, 1> Matx51d;
}

//-- private definitions -----
class OpenCVBufferState
{
public:
    OpenCVBufferState(const PSMClientTrackerInfo &_trackerInfo, PSMVideoFrameSection _section)
        : trackerInfo(_trackerInfo)
        , section(_section)
        , frameWidth(static_cast<int>(_trackerInfo.tracker_intrinsics.intrinsics.mono.pixel_width))
        , frameHeight(static_cast<int>(_trackerInfo.tracker_intrinsics.intrinsics.mono.pixel_height))
        , capturedBoardCount(0)
    {
        // Video Frame data
        bgrSourceBuffer = new cv::Mat(frameHeight, frameWidth, CV_8UC3);
        gsBuffer = new cv::Mat(frameHeight, frameWidth, CV_8UC1);
        gsBGRBuffer = new cv::Mat(frameHeight, frameWidth, CV_8UC3);
        bgrUndistortBuffer = new cv::Mat(frameHeight, frameWidth, CV_8UC3);

        // Distortion state
        distortionMapX = new cv::Mat(cv::Size(frameWidth, frameHeight), CV_32FC1);
        distortionMapY = new cv::Mat(cv::Size(frameWidth, frameHeight), CV_32FC1);

        resetCaptureState();
        resetCalibrationState();
    }

    virtual ~OpenCVBufferState()
    {
        // Video Frame data
        delete bgrSourceBuffer;
        delete gsBuffer;
        delete gsBGRBuffer;
        delete bgrUndistortBuffer;

        // Distortion state
        delete distortionMapX;
        delete distortionMapY;
    }

    void resetCaptureState()
    {
        capturedBoardCount= 0;
        bCurrentImagePointsValid= false;
        currentImagePoints.clear();
        lastValidImagePoints.clear();
        quadList.clear();
        imagePointsList.clear();
    }

    void resetCalibrationState()
    {
        PSMMatrix3d psm_camera_matrix;
        PSMDistortionCoefficients psm_distortion_coeffs;

        switch (section)
        {
        case PSMVideoFrameSection_Left:
            psm_camera_matrix= trackerInfo.tracker_intrinsics.intrinsics.stereo.left_camera_matrix;
            psm_distortion_coeffs= trackerInfo.tracker_intrinsics.intrinsics.stereo.left_distortion_coefficients;
            break;
        case PSMVideoFrameSection_Right:
            psm_camera_matrix= trackerInfo.tracker_intrinsics.intrinsics.stereo.right_camera_matrix;
            psm_distortion_coeffs= trackerInfo.tracker_intrinsics.intrinsics.stereo.right_distortion_coefficients;
            break;
        }

        intrinsic_matrix= psmove_matrix3x3_to_cv_mat33d(psm_camera_matrix);
        
        // Fill in the distortion coefficients
        distortion_coeffs(0, 0)= psm_distortion_coeffs.k1; 
        distortion_coeffs(1, 0)= psm_distortion_coeffs.k2;
        distortion_coeffs(2, 0)= psm_distortion_coeffs.p1;
        distortion_coeffs(3, 0)= psm_distortion_coeffs.p2;
        distortion_coeffs(4, 0)= psm_distortion_coeffs.k3;

        // Generate the distortion map that corresponds to the tracker's camera settings
        rebuildDistortionMap();
    }

    void applyVideoFrame(const unsigned char *video_buffer)
    {
        const cv::Mat videoBufferMat(frameHeight, frameWidth, CV_8UC3, const_cast<unsigned char *>(video_buffer));

        // Copy and Flip image about the x-axis
        videoBufferMat.copyTo(*bgrSourceBuffer);

        // Convert the video buffer to a grayscale image
        cv::cvtColor(*bgrSourceBuffer, *gsBuffer, cv::COLOR_BGR2GRAY);
        cv::cvtColor(*gsBuffer, *gsBGRBuffer, cv::COLOR_GRAY2BGR);

        // Apply the distortion map
        cv::remap(
            *bgrSourceBuffer, *bgrUndistortBuffer, 
            *distortionMapX, *distortionMapY, 
            cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    }

    bool findAndAppendNewChessBoard(bool appWantsAppend)
    {        
        bool bAppendNewChessBoard= false;

        if (capturedBoardCount < DESIRED_CAPTURE_BOARD_COUNT)
        {
            std::vector<cv::Point2f> new_image_points;

            // Find chessboard corners:
            if (cv::findChessboardCorners(
                    *gsBuffer, 
                    cv::Size(PATTERN_W, PATTERN_H), 
                    new_image_points, // output corners
                    cv::CALIB_CB_ADAPTIVE_THRESH 
                    + cv::CALIB_CB_FILTER_QUADS 
                    // + cv::CALIB_CB_NORMALIZE_IMAGE is suuuper slow
                    + cv::CALIB_CB_FAST_CHECK))
            {
                // Get subpixel accuracy on those corners
                cv::cornerSubPix(
                    *gsBuffer, 
                    new_image_points, // corners to refine
                    cv::Size(11, 11), // winSize- Half of the side length of the search window
                    cv::Size(-1, -1), // zeroZone- (-1,-1) means no dead zone in search
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

                // Append the new chessboard corner pixels into the image_points matrix
                // Append the corresponding 3d chessboard corners into the object_points matrix
                if (new_image_points.size() == CORNER_COUNT) 
                {
                    bCurrentImagePointsValid= false;
                    // See if the board is stationary (didn't move much since last frame)
                    if (currentImagePoints.size() > 0)
                    {
                        float error_sum= 0.f;

                        for (int corner_index= 0; corner_index < CORNER_COUNT; ++corner_index)
                        {
                            float squared_error= static_cast<float>(cv::norm(new_image_points[corner_index] - currentImagePoints[corner_index]));

                            error_sum+= squared_error;
                        }

                        bCurrentImagePointsValid= error_sum <= BOARD_MOVED_ERROR_SUM;
                    }
                    else
                    {
                        // We don't have previous capture.
                        bCurrentImagePointsValid= true;
                    }

                    // See if the board moved far enough from the last valid location
                    if (bCurrentImagePointsValid)
                    {
                        if (lastValidImagePoints.size() > 0)
                        {
                            float error_sum= 0.f;

                            for (int corner_index= 0; corner_index < CORNER_COUNT; ++corner_index)
                            {
                                float squared_error= static_cast<float>(cv::norm(new_image_points[corner_index] - lastValidImagePoints[corner_index]));

                                error_sum+= squared_error;
                            }

                            bCurrentImagePointsValid= error_sum >= BOARD_NEW_LOCATION_ERROR_SUM;
                        }
                    }

                    if (bCurrentImagePointsValid)
                    {
                        bCurrentImagePointsValid= areGridLinesStraight(new_image_points);
                    }

                    // If it's a valid new location, append it to the board list
                    if (bCurrentImagePointsValid && appWantsAppend)
                    {
                        // Keep track of the corners of all of the chessboards we sample
                        quadList.push_back(new_image_points[0]);
                        quadList.push_back(new_image_points[PATTERN_W - 1]);
                        quadList.push_back(new_image_points[CORNER_COUNT-1]);
                        quadList.push_back(new_image_points[CORNER_COUNT-PATTERN_W]);                        

                        // Append the new images points and object points
                        imagePointsList.push_back(new_image_points);

                        // Remember the last valid captured points
                        lastValidImagePoints= currentImagePoints;

                        // Keep track of how many boards have been captured so far
                        capturedBoardCount++;

                        bAppendNewChessBoard= true;
                    }

                    // Remember the last set of valid corners
                    currentImagePoints= new_image_points;
                }
            }
        }

        return bAppendNewChessBoard;
    }

    void popLastChessBoard()
    {
        if (capturedBoardCount > 0)
        {
            assert(quadList.size() > 0);
            assert(quadList.size() % 4 == 0);
            quadList.pop_back();
            quadList.pop_back();
            quadList.pop_back();
            quadList.pop_back();

            assert(imagePointsList.size() > 0);
            imagePointsList.pop_back();

            capturedBoardCount--;
        }
    }

    static bool areGridLinesStraight(const std::vector<cv::Point2f> &corners)
    {
        assert(corners.size() == CORNER_COUNT);
        bool bAllLinesStraight= true;

        for (int line_index= 0; bAllLinesStraight && line_index < PATTERN_H; ++line_index)
        {
            int start_index= line_index*PATTERN_W;
            int end_index= start_index + PATTERN_W - 1;

            cv::Point2f line_start= corners[start_index];
            cv::Point2f line_end= corners[end_index];

            for (int point_index= start_index + 1; bAllLinesStraight && point_index < end_index; ++point_index)
            {
                cv::Point2f point= corners[point_index];

                if (distanceToLine(line_start, line_end, point) > STRAIGHT_LINE_TOLERANCE)
                {
                    bAllLinesStraight= false;
                }
            }
        }

        return bAllLinesStraight;
    }

    static float distanceToLine(cv::Point2f line_start, cv::Point2f line_end, cv::Point2f point)
    {
        const auto start_to_end= line_end - line_start;
        const auto start_to_point= point - line_start;

        float area = static_cast<float>(start_to_point.cross(start_to_end));
        float line_length= static_cast<float>(cv::norm(start_to_end));
        return fabsf(safe_divide_with_default(area, line_length, 0.f));
    }

    void rebuildDistortionMap()
    {
        cv::initUndistortRectifyMap(
            intrinsic_matrix, distortion_coeffs, 
            rectification_rotation, rectification_projection, 
            cv::Size(frameWidth, frameHeight),
            CV_32FC1, // Distortion map type
            *distortionMapX, *distortionMapY);
    }

    const PSMClientTrackerInfo &trackerInfo;
    PSMVideoFrameSection section;
    int frameWidth;
    int frameHeight;

    // Video frame buffers
    cv::Mat *bgrSourceBuffer;
    cv::Mat *gsBuffer;
    cv::Mat *gsBGRBuffer;
    cv::Mat *bgrUndistortBuffer;

    // Chess board computed state
    int capturedBoardCount;
    std::vector<cv::Point2f> lastValidImagePoints;
    std::vector<cv::Point2f> currentImagePoints;
    bool bCurrentImagePointsValid;
    std::vector<cv::Point2f> quadList;
    std::vector<std::vector<cv::Point2f>> imagePointsList;

    // Calibration state
    cv::Matx33d intrinsic_matrix;
    cv::Matx33d rectification_rotation;
    cv::Matx34d rectification_projection;
    cv::Matx51d distortion_coeffs;

    // Distortion preview
    cv::Mat *distortionMapX;
    cv::Mat *distortionMapY;
};

class OpenCVStereoState
{
public:
    OpenCVStereoState(const PSMClientTrackerInfo &_trackerInfo)
        : m_videoDisplayMode(eVideoDisplayMode::mode_bgr)
        , trackerInfo(_trackerInfo)
        , frameWidth(_trackerInfo.tracker_intrinsics.intrinsics.mono.pixel_width)
        , frameHeight(_trackerInfo.tracker_intrinsics.intrinsics.mono.pixel_height)
    {
        m_opencv_state[PSMVideoFrameSection_Left]= nullptr;
        m_opencv_state[PSMVideoFrameSection_Right]= nullptr;
        m_video_texture[PSMVideoFrameSection_Left]= nullptr;
        m_video_texture[PSMVideoFrameSection_Right]= nullptr;
        reprojectionError= 0.f;
    }

    virtual ~OpenCVStereoState()
    {
        if (m_opencv_state[PSMVideoFrameSection_Left] != nullptr)
        {
            delete m_opencv_state[PSMVideoFrameSection_Left];
            m_opencv_state[PSMVideoFrameSection_Left]= nullptr;
        }

        if (m_opencv_state[PSMVideoFrameSection_Right] != nullptr)
        {
            delete m_opencv_state[PSMVideoFrameSection_Right];
            m_opencv_state[PSMVideoFrameSection_Right]= nullptr;
        }

        disposeVideoTextures();
    }

    void allocateVideoTextures()
    {
        const PSMStereoTrackerIntrinsics &trackerIntrinsics= trackerInfo.tracker_intrinsics.intrinsics.stereo;

        const int width= static_cast<int>(trackerIntrinsics.pixel_width);
        const int height= static_cast<int>(trackerIntrinsics.pixel_height);

        // Create a texture to render the video frame to
        m_video_texture[PSMVideoFrameSection_Left] = new TextureAsset();
        m_video_texture[PSMVideoFrameSection_Left]->init(
            width, 
            height,
            GL_RGB, // texture format
            GL_BGR, // buffer format
            nullptr);
        m_video_texture[PSMVideoFrameSection_Right] = new TextureAsset();
        m_video_texture[PSMVideoFrameSection_Right]->init(
            width, 
            height,
            GL_RGB, // texture format
            GL_BGR, // buffer format
            nullptr);

        // Allocate an opencv buffer 
        m_opencv_state[PSMVideoFrameSection_Left] = 
            new OpenCVBufferState(trackerInfo, PSMVideoFrameSection_Left);
        m_opencv_state[PSMVideoFrameSection_Right] = 
            new OpenCVBufferState(trackerInfo, PSMVideoFrameSection_Right);
    }

    void disposeVideoTextures()
    {
        // Free the texture we were rendering to
        if (m_video_texture[PSMVideoFrameSection_Left] != nullptr)
        {
            delete m_video_texture[PSMVideoFrameSection_Left];
            m_video_texture[PSMVideoFrameSection_Left] = nullptr;
        }

        if (m_video_texture[PSMVideoFrameSection_Right] != nullptr)
        {
            delete m_video_texture[PSMVideoFrameSection_Right];
            m_video_texture[PSMVideoFrameSection_Right] = nullptr;
        }
    }

    void applyStereoVideoFrame(
        const unsigned char *left_video_frame_buffer, 
        const unsigned char *right_video_frame_buffer)
    {
        m_opencv_state[PSMVideoFrameSection_Left]->applyVideoFrame(left_video_frame_buffer);
        m_opencv_state[PSMVideoFrameSection_Right]->applyVideoFrame(right_video_frame_buffer);

		// Update the video frame display texture
		switch (m_videoDisplayMode)
		{
		case mode_bgr:
			m_video_texture[PSMVideoFrameSection_Left]->copyBufferIntoTexture(m_opencv_state[PSMVideoFrameSection_Left]->bgrSourceBuffer->data);
            m_video_texture[PSMVideoFrameSection_Right]->copyBufferIntoTexture(m_opencv_state[PSMVideoFrameSection_Right]->bgrSourceBuffer->data);
			break;
		case mode_grayscale:
			m_video_texture[PSMVideoFrameSection_Left]->copyBufferIntoTexture(m_opencv_state[PSMVideoFrameSection_Left]->gsBGRBuffer->data);
            m_video_texture[PSMVideoFrameSection_Right]->copyBufferIntoTexture(m_opencv_state[PSMVideoFrameSection_Right]->gsBGRBuffer->data);
			break;
		case mode_undistored:
			m_video_texture[PSMVideoFrameSection_Left]->copyBufferIntoTexture(m_opencv_state[PSMVideoFrameSection_Left]->bgrUndistortBuffer->data);
            m_video_texture[PSMVideoFrameSection_Right]->copyBufferIntoTexture(m_opencv_state[PSMVideoFrameSection_Right]->bgrUndistortBuffer->data);
			break;
		default:
			assert(0 && "unreachable");
			break;
		}
    }

    void findNewChessBoards()
    {
        ImGuiIO io_state = ImGui::GetIO();
        const bool bWantsAppend= io_state.KeysDown[32];

        bool bLeftAppend= m_opencv_state[PSMVideoFrameSection_Left]->findAndAppendNewChessBoard(bWantsAppend);
        bool bRightAppend= m_opencv_state[PSMVideoFrameSection_Right]->findAndAppendNewChessBoard(bWantsAppend);

        if (bWantsAppend)
        {
            if (bLeftAppend && !bRightAppend)
            {
                m_opencv_state[PSMVideoFrameSection_Left]->popLastChessBoard();
            }
            if (!bLeftAppend && bRightAppend)
            {
                m_opencv_state[PSMVideoFrameSection_Right]->popLastChessBoard();
            }
        }
    }

    bool hasSampledAllChessBoards() const
    {
        return
            m_opencv_state[PSMVideoFrameSection_Left]->capturedBoardCount >= DESIRED_CAPTURE_BOARD_COUNT &&
            m_opencv_state[PSMVideoFrameSection_Right]->capturedBoardCount >= DESIRED_CAPTURE_BOARD_COUNT;
    }

    bool areCurrentImagePointsValid() const
    {
        return 
            m_opencv_state[PSMVideoFrameSection_Left]->bCurrentImagePointsValid &&
            m_opencv_state[PSMVideoFrameSection_Right]->bCurrentImagePointsValid;
    }

    float computeCalibrationProgress() const 
    {
        const float leftSamplePercentage= 
            static_cast<float>(m_opencv_state[PSMVideoFrameSection_Left]->capturedBoardCount) / static_cast<float>(DESIRED_CAPTURE_BOARD_COUNT);
        const float rightSamplePercentage= 
            static_cast<float>(m_opencv_state[PSMVideoFrameSection_Right]->capturedBoardCount) / static_cast<float>(DESIRED_CAPTURE_BOARD_COUNT);

        return std::min(leftSamplePercentage, rightSamplePercentage);
    }

    void resetCalibration()
    {
        m_opencv_state[PSMVideoFrameSection_Left]->resetCaptureState();
        m_opencv_state[PSMVideoFrameSection_Right]->resetCaptureState();
        m_opencv_state[PSMVideoFrameSection_Left]->resetCalibrationState();
        m_opencv_state[PSMVideoFrameSection_Right]->resetCalibrationState();

        m_videoDisplayMode= mode_bgr;
    }

    inline void calcBoardCornerPositions(const float square_length_mm, std::vector<cv::Point3f>& corners)
    {
        corners.clear();
        
        for( int i = 0; i < PATTERN_H; ++i )
        {
            for( int j = 0; j < PATTERN_W; ++j )
            {
                corners.push_back(cv::Point3f(float(j*square_length_mm), float(i*square_length_mm), 0.f));
            }
        }
    }

    inline PSMDistortionCoefficients cv_vec5_to_psm_distortion(const cv::Matx51d &cv_distortion_coeffs)
    {
        PSMDistortionCoefficients distortion_coeffs;
        distortion_coeffs.k1= cv_distortion_coeffs(0, 0);
        distortion_coeffs.k2= cv_distortion_coeffs(1, 0);
        distortion_coeffs.p1= cv_distortion_coeffs(2, 0);
        distortion_coeffs.p2= cv_distortion_coeffs(3, 0);
        distortion_coeffs.k3= cv_distortion_coeffs(4, 0);

        return distortion_coeffs;
    }

    void computeCameraCalibration(
        const float square_length_mm, 
        PSMStereoTrackerIntrinsics &out_stereo_intrinsics)
    {
        // Copy over the pre-existing tracker intrinsics
        out_stereo_intrinsics= trackerInfo.tracker_intrinsics.intrinsics.stereo;

        // Only need to calculate objectPointsList once,
        // then resize for each set of image points.
        std::vector<std::vector<cv::Point3f> > objectPointsList(1);
        calcBoardCornerPositions(square_length_mm, objectPointsList[0]);
        objectPointsList.resize(m_opencv_state[PSMVideoFrameSection_Left]->imagePointsList.size(), objectPointsList[0]);
            
        // Compute the stereo camera calibration using epipolar geometry
        cv::stereoCalibrate(
            objectPointsList, 
            m_opencv_state[PSMVideoFrameSection_Left]->imagePointsList,
            m_opencv_state[PSMVideoFrameSection_Right]->imagePointsList,
            m_opencv_state[PSMVideoFrameSection_Left]->intrinsic_matrix,
            m_opencv_state[PSMVideoFrameSection_Left]->distortion_coeffs,
            m_opencv_state[PSMVideoFrameSection_Right]->intrinsic_matrix,
            m_opencv_state[PSMVideoFrameSection_Right]->distortion_coeffs,
            cv::Size(static_cast<int>(frameWidth), static_cast<int>(frameHeight)), 
            rotation_between_cameras,
            translation_between_cameras,
            essential_matrix,
            fundamental_matrix, 
            cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_SAME_FOCAL_LENGTH,
            cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 1e-5));

        // Calculate the calibration quality
        reprojectionError= computeStereoCalibrationError();

        // Compute the stereo rotation and projection rectification transforms
        cv::stereoRectify(
            m_opencv_state[PSMVideoFrameSection_Left]->intrinsic_matrix,
            m_opencv_state[PSMVideoFrameSection_Left]->distortion_coeffs,
            m_opencv_state[PSMVideoFrameSection_Right]->intrinsic_matrix,
            m_opencv_state[PSMVideoFrameSection_Right]->distortion_coeffs,
            cv::Size(static_cast<int>(frameWidth), static_cast<int>(frameHeight)),
            rotation_between_cameras, translation_between_cameras,
            m_opencv_state[PSMVideoFrameSection_Left]->rectification_rotation,
            m_opencv_state[PSMVideoFrameSection_Right]->rectification_rotation,
            m_opencv_state[PSMVideoFrameSection_Left]->rectification_projection,
            m_opencv_state[PSMVideoFrameSection_Right]->rectification_projection,
            reprojection_matrix);

        // Recompute the distortion mapping (for debug display only)
        m_opencv_state[PSMVideoFrameSection_Left]->rebuildDistortionMap();
        m_opencv_state[PSMVideoFrameSection_Right]->rebuildDistortionMap();

        // Pack the calibration results into the output stereo intrinsics structure
        out_stereo_intrinsics.left_camera_matrix= 
            cv_mat33d_to_psm_matrix3x3(m_opencv_state[PSMVideoFrameSection_Left]->intrinsic_matrix);
        out_stereo_intrinsics.right_camera_matrix= 
            cv_mat33d_to_psm_matrix3x3(m_opencv_state[PSMVideoFrameSection_Right]->intrinsic_matrix);
        out_stereo_intrinsics.left_distortion_coefficients= 
            cv_vec5_to_psm_distortion(m_opencv_state[PSMVideoFrameSection_Left]->distortion_coeffs);
        out_stereo_intrinsics.right_distortion_coefficients= 
            cv_vec5_to_psm_distortion(m_opencv_state[PSMVideoFrameSection_Right]->distortion_coeffs);
        out_stereo_intrinsics.left_rectification_rotation=
            cv_mat33d_to_psm_matrix3x3(m_opencv_state[PSMVideoFrameSection_Left]->rectification_rotation);
        out_stereo_intrinsics.right_rectification_rotation=
            cv_mat33d_to_psm_matrix3x3(m_opencv_state[PSMVideoFrameSection_Right]->rectification_rotation);
        out_stereo_intrinsics.left_rectification_projection= 
            cv_mat34d_to_psm_matrix3x4(m_opencv_state[PSMVideoFrameSection_Left]->rectification_projection);
        out_stereo_intrinsics.right_rectification_projection=
            cv_mat34d_to_psm_matrix3x4(m_opencv_state[PSMVideoFrameSection_Right]->rectification_projection);
        out_stereo_intrinsics.rotation_between_cameras= cv_mat33d_to_psm_matrix3x3(rotation_between_cameras);
        out_stereo_intrinsics.translation_between_cameras= cv_vec3d_to_psm_vector3d(translation_between_cameras);
        out_stereo_intrinsics.essential_matrix= cv_mat33d_to_psm_matrix3x3(essential_matrix);
        out_stereo_intrinsics.fundamental_matrix= cv_mat33d_to_psm_matrix3x3(fundamental_matrix);
        out_stereo_intrinsics.reprojection_matrix= cv_mat44d_to_psm_matrix4x4(reprojection_matrix);
    }

    double computeStereoCalibrationError()
    {
        double totalError= 0.0;
        double errSampleCount= 0.0;

        assert(m_opencv_state[PSMVideoFrameSection_Left]->imagePointsList.size() == DESIRED_CAPTURE_BOARD_COUNT);
        assert(m_opencv_state[PSMVideoFrameSection_Right]->imagePointsList.size() == DESIRED_CAPTURE_BOARD_COUNT);

        for (int boardSampleIndex = 0; boardSampleIndex < DESIRED_CAPTURE_BOARD_COUNT; ++boardSampleIndex)
        {
            const std::vector<cv::Point2f> &leftImagePoints=
                m_opencv_state[PSMVideoFrameSection_Left]->imagePointsList[boardSampleIndex];
            const std::vector<cv::Point2f> &rightImagePoints=
                m_opencv_state[PSMVideoFrameSection_Right]->imagePointsList[boardSampleIndex];
            
            std::vector<cv::Point2f> undistortedLeftImagePoints;
            std::vector<cv::Point2f> undistortedRightImagePoints;
            cv::undistortPoints(
                leftImagePoints, undistortedLeftImagePoints,
                m_opencv_state[PSMVideoFrameSection_Left]->intrinsic_matrix,
                m_opencv_state[PSMVideoFrameSection_Left]->distortion_coeffs);
            cv::undistortPoints(
                rightImagePoints, undistortedRightImagePoints,
                m_opencv_state[PSMVideoFrameSection_Right]->intrinsic_matrix,
                m_opencv_state[PSMVideoFrameSection_Right]->distortion_coeffs);

            std::vector<cv::Vec3f> leftLines;
            std::vector<cv::Vec3f> rightLines;
            cv::computeCorrespondEpilines(undistortedLeftImagePoints, 1, fundamental_matrix, leftLines);
            cv::computeCorrespondEpilines(undistortedRightImagePoints, 2, fundamental_matrix, rightLines);

            assert(undistortedLeftImagePoints.size() == CORNER_COUNT);
            assert(undistortedRightImagePoints.size() == CORNER_COUNT);

            for (int corner_index = 0; corner_index < CORNER_COUNT; ++corner_index)
            {
                const cv::Point2f &leftImagePoint= undistortedLeftImagePoints[corner_index];
                const cv::Point2f &rightImagePoint= undistortedRightImagePoints[corner_index];
                const cv::Vec3f &leftLine= leftLines[corner_index];
                const cv::Vec3f &rightLine= rightLines[corner_index];

                // Compute abs dist of point on one image to the epipolar line of its corresponding
                // point in the other image
                const double left_error= 
                    fabs(rightLine(0)*leftImagePoint.x + rightLine(1)*leftImagePoint.y + rightLine(2));
                const double right_error= 
                    fabs(leftLine(0)*rightImagePoint.x + leftLine(1)*rightImagePoint.y + leftLine(2));

                totalError+= (left_error+right_error);
                errSampleCount+= 1.0;
            }
        }

        return (errSampleCount > 0.0) ? totalError / errSampleCount : 0.0;
    }

    void renderStereoVideoBuffers()
    {
        assert(m_video_texture[PSMVideoFrameSection_Left] != nullptr);
        assert(m_video_texture[PSMVideoFrameSection_Right] != nullptr);
        unsigned int left_texture_id = m_video_texture[PSMVideoFrameSection_Left]->texture_id;
        unsigned int right_texture_id = m_video_texture[PSMVideoFrameSection_Right]->texture_id;

        if (left_texture_id != 0 && right_texture_id != 0)
        {
            drawFullscreenStereoTexture(left_texture_id, right_texture_id);
        }
    }

    void renderStereoChessboard()
    {
        const float midX= 0.5f*frameWidth;
        const float rightX= frameWidth-1.f;
        const float topY= 0.25f*(frameHeight-1.f);
        const float bottomY= 0.75f*(frameHeight-1.f);

        const float leftX0= 0.f, leftY0= topY;
        const float leftX1= midX, leftY1= bottomY;

        const float rightX0= midX, rightY0= topY;
        const float rightX1= rightX, rightY1= bottomY;

        // Draw the last valid capture chessboard
        if (m_opencv_state[PSMVideoFrameSection_Left]->lastValidImagePoints.size() > 0)
        {
            drawOpenCVChessBoardInSubWindow(
                frameWidth, frameHeight,
                leftX0, leftY0,
                leftX1, leftY1,
                reinterpret_cast<float *>(m_opencv_state[PSMVideoFrameSection_Left]->lastValidImagePoints.data()), // cv::point2f is just two floats 
                static_cast<int>(m_opencv_state[PSMVideoFrameSection_Left]->lastValidImagePoints.size()),
                true);
        }
        if (m_opencv_state[PSMVideoFrameSection_Right]->lastValidImagePoints.size() > 0)
        {
            drawOpenCVChessBoardInSubWindow(
                frameWidth, frameHeight,
                rightX0, rightY0,
                rightX1, rightY1,
                reinterpret_cast<float *>(m_opencv_state[PSMVideoFrameSection_Right]->lastValidImagePoints.data()), // cv::point2f is just two floats 
                static_cast<int>(m_opencv_state[PSMVideoFrameSection_Right]->lastValidImagePoints.size()),
                true);
        } 


        // Draw the most recently capture chessboard
        if (m_opencv_state[PSMVideoFrameSection_Left]->currentImagePoints.size() > 0)
        {
            drawOpenCVChessBoardInSubWindow(
                frameWidth, frameHeight,
                leftX0, leftY0,
                leftX1, leftY1,
                reinterpret_cast<float *>(m_opencv_state[PSMVideoFrameSection_Left]->currentImagePoints.data()), // cv::point2f is just two floats 
                static_cast<int>(m_opencv_state[PSMVideoFrameSection_Left]->currentImagePoints.size()),
                m_opencv_state[PSMVideoFrameSection_Left]->bCurrentImagePointsValid);
        }
        if (m_opencv_state[PSMVideoFrameSection_Right]->currentImagePoints.size() > 0)
        {
            drawOpenCVChessBoardInSubWindow(
                frameWidth, frameHeight,
                rightX0, rightY0,
                rightX1, rightY1,
                reinterpret_cast<float *>(m_opencv_state[PSMVideoFrameSection_Right]->currentImagePoints.data()), // cv::point2f is just two floats 
                static_cast<int>(m_opencv_state[PSMVideoFrameSection_Right]->currentImagePoints.size()),
                m_opencv_state[PSMVideoFrameSection_Right]->bCurrentImagePointsValid);
        }

        // Draw the outlines of all of the chess boards 
        if (m_opencv_state[PSMVideoFrameSection_Left]->quadList.size() > 0)
        {
            drawQuadList2dInSubWindow(
                frameWidth, frameHeight, 
                leftX0, leftY0,
                leftX1, leftY1,
                glm::vec3(1.f, 1.f, 0.f), 
                    reinterpret_cast<float *>(m_opencv_state[PSMVideoFrameSection_Left]->quadList.data()), // cv::point2f is just two floats 
                    static_cast<int>(m_opencv_state[PSMVideoFrameSection_Left]->quadList.size()));
        }
        if (m_opencv_state[PSMVideoFrameSection_Right]->quadList.size() > 0)
        {
            drawQuadList2dInSubWindow(
                frameWidth, frameHeight, 
                rightX0, rightY0,
                rightX1, rightY1,
                glm::vec3(1.f, 1.f, 0.f), 
                    reinterpret_cast<float *>(m_opencv_state[PSMVideoFrameSection_Right]->quadList.data()), // cv::point2f is just two floats 
                    static_cast<int>(m_opencv_state[PSMVideoFrameSection_Right]->quadList.size()));
        }
    }

    // Menu state
    eVideoDisplayMode m_videoDisplayMode;

    const PSMClientTrackerInfo &trackerInfo;
    float frameWidth;
    float frameHeight;

    // Calibration State
    double reprojectionError;
    cv::Matx33d rotation_between_cameras;
    cv::Vec3d translation_between_cameras;
    cv::Matx33d essential_matrix;
    cv::Matx33d fundamental_matrix;
    cv::Matx44d reprojection_matrix;

    OpenCVBufferState *m_opencv_state[2];
    TextureAsset *m_video_texture[2];
};

//-- public methods -----
AppStage_StereoCalibration::AppStage_StereoCalibration(App *app)
    : AppStage(app)
    , m_menuState(AppStage_StereoCalibration::inactive)
	, m_square_length_mm(DEFAULT_SQUARE_LEN_MM)
    , m_trackerExposure(0.0)
    , m_trackerGain(0.0)
    , m_bStreamIsActive(false)
    , m_tracker_view(nullptr)
{ 
    m_opencv_stereo_state= nullptr;
}

void AppStage_StereoCalibration::enter()
{
    const AppStage_TrackerSettings *trackerSettings =
        m_app->getAppStage<AppStage_TrackerSettings>();
    const PSMClientTrackerInfo *trackerInfo = trackerSettings->getSelectedTrackerInfo();
    assert(trackerInfo->tracker_id != -1);

    m_app->setCameraType(_cameraFixed);

    assert(m_tracker_view == nullptr);
	PSM_AllocateTrackerListener(trackerInfo->tracker_id, trackerInfo);
	m_tracker_view = PSM_GetTracker(trackerInfo->tracker_id);

	m_square_length_mm = DEFAULT_SQUARE_LEN_MM;

	assert(!m_bStreamIsActive);
	request_tracker_start_stream();
}

void AppStage_StereoCalibration::exit()
{
    m_menuState = AppStage_StereoCalibration::inactive;

    if (m_opencv_stereo_state != nullptr)
    {
        delete m_opencv_stereo_state;
        m_opencv_stereo_state= nullptr;
    }

    // Revert unsaved modifications to the tracker settings
    request_tracker_reload_settings();

    PSM_FreeTrackerListener(m_tracker_view->tracker_info.tracker_id);
    m_tracker_view = nullptr;
    m_bStreamIsActive= false;
}

void AppStage_StereoCalibration::update()
{
    if (m_menuState == AppStage_StereoCalibration::capture ||
        m_menuState == AppStage_StereoCalibration::complete)
    {
        // Try and read the next video frame from shared memory
        if (PSM_PollTrackerVideoStream(m_tracker_view->tracker_info.tracker_id) == PSMResult_Success)
        {
            const unsigned char *left_video_frame_buffer= nullptr;
            const unsigned char *right_video_frame_buffer= nullptr;
			if (PSM_GetTrackerVideoFrameBuffer(m_tracker_view->tracker_info.tracker_id, PSMVideoFrameSection_Left, &left_video_frame_buffer) == PSMResult_Success &&
                PSM_GetTrackerVideoFrameBuffer(m_tracker_view->tracker_info.tracker_id, PSMVideoFrameSection_Right, &right_video_frame_buffer) == PSMResult_Success)
			{
				// Update the video frame buffers
                m_opencv_stereo_state->applyStereoVideoFrame(left_video_frame_buffer, right_video_frame_buffer);
			}

            if (m_menuState == AppStage_StereoCalibration::capture)
            {
                // Update the chess board capture state
                m_opencv_stereo_state->findNewChessBoards();

                if (m_opencv_stereo_state->hasSampledAllChessBoards())
                {
                    PSMStereoTrackerIntrinsics new_stereo_intrinsics;

                    m_opencv_stereo_state->computeCameraCalibration(m_square_length_mm, new_stereo_intrinsics);

                    // Update the camera intrinsics for this camera
                    request_tracker_set_intrinsic(new_stereo_intrinsics);

                    m_opencv_stereo_state->m_videoDisplayMode= mode_undistored;
                    m_menuState= AppStage_StereoCalibration::complete;
                }
            }
        }
    }
}

void AppStage_StereoCalibration::render()
{
    if (m_menuState == AppStage_StereoCalibration::capture ||
        m_menuState == AppStage_StereoCalibration::complete)
    {
        m_opencv_stereo_state->renderStereoVideoBuffers();

        if (m_menuState == AppStage_StereoCalibration::capture)
        {
            m_opencv_stereo_state->renderStereoChessboard();
        }
    }
}

void AppStage_StereoCalibration::renderUI()
{
    const float k_panel_width = 200.f;
    const char *k_window_title = "Distortion Calibration";
    const ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_ShowBorders |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoCollapse;

    switch (m_menuState)
    {
	case eMenuState::showWarning:
		{
			const float k_wide_panel_width = 350.f;
			ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_wide_panel_width / 2.f, 20.f));
			ImGui::SetNextWindowSize(ImVec2(k_wide_panel_width, 130));

			ImGui::Begin("WARNING", nullptr, window_flags);

			ImGui::TextWrapped(
				"The tracker you want to calibrate already has pre-computed distortion and focal lengths." \
				"If you proceed you will be overriding these defaults.");

			ImGui::Spacing();

			if (ImGui::Button("Continue"))
			{
				m_menuState = eMenuState::enterBoardSettings;
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel"))
			{
				request_exit();
			}

			ImGui::End();
		} break;
	case eMenuState::enterBoardSettings:
		{
			const float k_wide_panel_width = 350.f;
			ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_wide_panel_width / 2.f, 20.f));
			ImGui::SetNextWindowSize(ImVec2(k_wide_panel_width, 100));

			ImGui::Begin("Enter Calibration Settings", nullptr, window_flags);

			ImGui::PushItemWidth(100.f);
			if (ImGui::InputFloat("Square Length (mm)", &m_square_length_mm, 0.5f, 1.f, 1))
			{
				if (m_square_length_mm < 1.f)
				{
					m_square_length_mm = 1.f;
				}

				if (m_square_length_mm > 100.f)
				{
					m_square_length_mm = 100.f;
				}
			}
			ImGui::PopItemWidth();

			ImGui::Spacing();

			if (ImGui::Button("Ok"))
			{
				// Crank up the exposure and gain so that we can see the chessboard
				// These overrides will get rolled back once tracker gets closed
				request_tracker_set_temp_exposure(128.f);
				request_tracker_set_temp_gain(128.f);

				m_menuState = eMenuState::capture;
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel"))
			{
				request_exit();
			}

			ImGui::End();
		} break;
    case eMenuState::capture:
        {
            assert (m_opencv_stereo_state != nullptr);

            {
                ImGui::SetNextWindowPos(ImVec2(10.f, 10.f));
                ImGui::SetNextWindowSize(ImVec2(275, 150));
                ImGui::Begin("Video Controls", nullptr, window_flags);

                if (ImGui::Button("<##Filter"))
                {
                    m_opencv_stereo_state->m_videoDisplayMode =
                        static_cast<eVideoDisplayMode>(
                        (m_opencv_stereo_state->m_videoDisplayMode + eVideoDisplayMode::MAX_VIDEO_DISPLAY_MODES - 1)
                        % eVideoDisplayMode::MAX_VIDEO_DISPLAY_MODES);
                }
                ImGui::SameLine();
                if (ImGui::Button(">##Filter"))
                {
                    m_opencv_stereo_state->m_videoDisplayMode =
                        static_cast<eVideoDisplayMode>(
                        (m_opencv_stereo_state->m_videoDisplayMode + 1) % eVideoDisplayMode::MAX_VIDEO_DISPLAY_MODES);
                }
                ImGui::SameLine();
                ImGui::Text("Video Filter Mode: %s", k_video_display_mode_names[m_opencv_stereo_state->m_videoDisplayMode]);

                if (ImGui::Button("-##Exposure"))
                {
                    request_tracker_set_temp_exposure(m_trackerExposure - 8);
                }
                ImGui::SameLine();
                if (ImGui::Button("+##Exposure"))
                {
                    request_tracker_set_temp_exposure(m_trackerExposure + 8);
                }
                ImGui::SameLine();
                ImGui::Text("Exposure: %f", m_trackerExposure);

                if (ImGui::Button("-##Gain"))
                {
                    request_tracker_set_temp_gain(m_trackerGain - 8);
                }
                ImGui::SameLine();
                if (ImGui::Button("+##Gain"))
                {
                    request_tracker_set_temp_gain(m_trackerGain + 8);
                }
                ImGui::SameLine();
                ImGui::Text("Gain: %f", m_trackerGain);

                ImGui::End();
            }

            {
                ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_panel_width / 2.f, 20.f));
                ImGui::SetNextWindowSize(ImVec2(k_panel_width, 110));
                ImGui::Begin(k_window_title, nullptr, window_flags);

                ImGui::ProgressBar(m_opencv_stereo_state->computeCalibrationProgress(), ImVec2(k_panel_width - 20, 20));

                if (ImGui::Button("Restart"))
                {
                    m_opencv_stereo_state->resetCalibration();
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel"))
                {
                    request_exit();
                }
                if (m_opencv_stereo_state->areCurrentImagePointsValid())
                {
                    ImGui::Text("Press spacebar to capture");
                }

                ImGui::End();
            }
        } break;

    case eMenuState::complete:
        {
            ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x / 2.f - k_panel_width / 2.f, 10.f));
            ImGui::SetNextWindowSize(ImVec2(k_panel_width, 110));
            ImGui::Begin(k_window_title, nullptr, window_flags);

            ImGui::Text("Calibration complete!");
            ImGui::Text("Avg Error: %f", m_opencv_stereo_state->reprojectionError);

            if (ImGui::Button("Ok"))
            {
                request_exit();
            }

            if (ImGui::Button("Redo Calibration"))
            {
                m_opencv_stereo_state->resetCalibration();

                m_menuState= eMenuState::capture;
            }

            ImGui::End();
        } break;

    case eMenuState::pendingTrackerStartStreamRequest:
        {
            ImGui::SetNextWindowPosCenter();
            ImGui::SetNextWindowSize(ImVec2(k_panel_width, 50));
            ImGui::Begin(k_window_title, nullptr, window_flags);

            ImGui::Text("Waiting for tracker stream to start...");

            ImGui::End();
        } break;

    case eMenuState::failedTrackerStartStreamRequest:
    case eMenuState::failedTrackerOpenStreamRequest:
        {
            ImGui::SetNextWindowPosCenter();
            ImGui::SetNextWindowSize(ImVec2(k_panel_width, 130));
            ImGui::Begin(k_window_title, nullptr, window_flags);

            if (m_menuState == eMenuState::failedTrackerStartStreamRequest)
                ImGui::Text("Failed to start tracker stream!");
            else
                ImGui::Text("Failed to open tracker stream!");

            if (ImGui::Button("Ok"))
            {
                m_app->setAppStage(AppStage_TrackerSettings::APP_STAGE_NAME);
            }

            if (ImGui::Button("Return to Main Menu"))
            {
                m_app->setAppStage(AppStage_MainMenu::APP_STAGE_NAME);
            }

            ImGui::End();
        } break;

    case eMenuState::pendingTrackerStopStreamRequest:
        {
            ImGui::SetNextWindowPosCenter();
            ImGui::SetNextWindowSize(ImVec2(k_panel_width, 50));
            ImGui::Begin(k_window_title, nullptr, window_flags);

            ImGui::Text("Waiting for tracker stream to stop...");

            ImGui::End();
        } break;

    case eMenuState::failedTrackerStopStreamRequest:
        {
            ImGui::SetNextWindowPosCenter();
            ImGui::SetNextWindowSize(ImVec2(k_panel_width, 130));
            ImGui::Begin(k_window_title, nullptr, window_flags);

            ImGui::Text("Failed to stop tracker stream!");

            if (ImGui::Button("Ok"))
            {
                m_app->setAppStage(AppStage_TrackerSettings::APP_STAGE_NAME);
            }

            if (ImGui::Button("Return to Main Menu"))
            {
                m_app->setAppStage(AppStage_MainMenu::APP_STAGE_NAME);
            }

            ImGui::End();
        } break;

    default:
        assert(0 && "unreachable");
    }
}

void AppStage_StereoCalibration::request_tracker_start_stream()
{
    if (m_menuState != AppStage_StereoCalibration::pendingTrackerStartStreamRequest)
    {
        m_menuState = AppStage_StereoCalibration::pendingTrackerStartStreamRequest;

        // Tell the psmove service that we want to start streaming data from the tracker
		PSMRequestID requestID;
		PSM_StartTrackerDataStreamAsync(
			m_tracker_view->tracker_info.tracker_id, 
			&requestID);
		PSM_RegisterCallback(requestID, AppStage_StereoCalibration::handle_tracker_start_stream_response, this);
    }
}

void AppStage_StereoCalibration::handle_tracker_start_stream_response(
    const PSMResponseMessage *response,
    void *userdata)
{
    AppStage_StereoCalibration *thisPtr = static_cast<AppStage_StereoCalibration *>(userdata);

    switch (response->result_code)
    {
    case PSMResult_Success:
        {
            PSMTracker *trackerView= thisPtr->m_tracker_view;

            thisPtr->m_bStreamIsActive = true;

            // Open the shared memory that the video stream is being written to
            if (PSM_OpenTrackerVideoStream(trackerView->tracker_info.tracker_id) == PSMResult_Success)
            {
                thisPtr->m_opencv_stereo_state->allocateVideoTextures();

				// Warn the user if they are about to change the distortion calibration settings for the PS4Camera
				if (trackerView->tracker_info.tracker_type == PSMTrackerType::PSMTracker_PS4Camera)
				{
					thisPtr->m_menuState = AppStage_StereoCalibration::showWarning;
				}
				else
				{
					// Start capturing chess boards
					thisPtr->m_menuState = AppStage_StereoCalibration::enterBoardSettings;
				}
            }
            else
            {
                thisPtr->m_menuState = AppStage_StereoCalibration::failedTrackerOpenStreamRequest;
            }
        } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
	case PSMResult_Timeout:
        {
            thisPtr->m_menuState = AppStage_StereoCalibration::failedTrackerStartStreamRequest;
        } break;
    }
}

void AppStage_StereoCalibration::request_tracker_stop_stream()
{
    if (m_bStreamIsActive && m_menuState != AppStage_StereoCalibration::pendingTrackerStopStreamRequest)
    {
        m_menuState = AppStage_StereoCalibration::pendingTrackerStopStreamRequest;

        // Tell the psmove service that we want to stop streaming data from the tracker
		PSMRequestID request_id;
		PSM_StopTrackerDataStreamAsync(m_tracker_view->tracker_info.tracker_id, &request_id);
		PSM_RegisterCallback(request_id, AppStage_StereoCalibration::handle_tracker_stop_stream_response, this);
    }
}

void AppStage_StereoCalibration::handle_tracker_stop_stream_response(
    const PSMResponseMessage *response,
    void *userdata)
{
    AppStage_StereoCalibration *thisPtr = static_cast<AppStage_StereoCalibration *>(userdata);

    // In either case consider the stream as now inactive
    thisPtr->m_bStreamIsActive = false;

    switch (response->result_code)
    {
    case PSMResult_Success:
        {
            thisPtr->m_menuState = AppStage_StereoCalibration::inactive;

            // Close the shared memory buffer
			PSM_CloseTrackerVideoStream(thisPtr->m_tracker_view->tracker_info.tracker_id);

            // Free memory allocated for the video frame textures
            thisPtr->m_opencv_stereo_state->disposeVideoTextures();

            // After closing the stream, we should go back to the tracker settings
            thisPtr->m_app->setAppStage(AppStage_TrackerSettings::APP_STAGE_NAME);
        } break;

    case PSMResult_Error:
    case PSMResult_Canceled:
	case PSMResult_Timeout:
        {
            thisPtr->m_menuState = AppStage_StereoCalibration::failedTrackerStopStreamRequest;
        } break;
    }
}

void AppStage_StereoCalibration::request_tracker_set_temp_gain(float gain)
{
    m_trackerGain= gain;

    // Tell the psmove service that we want to change gain, but not save the change
    RequestPtr request(new PSMoveProtocol::Request());
    request->set_type(PSMoveProtocol::Request_RequestType_SET_TRACKER_GAIN);
    request->mutable_request_set_tracker_gain()->set_tracker_id(m_tracker_view->tracker_info.tracker_id);
    request->mutable_request_set_tracker_gain()->set_value(gain);
    request->mutable_request_set_tracker_gain()->set_save_setting(false);

    PSM_SendOpaqueRequest(&request, nullptr);
}

void AppStage_StereoCalibration::request_tracker_set_temp_exposure(float exposure)
{
    m_trackerExposure= exposure;

    // Tell the psmove service that we want to change exposure, but not save the change.
    RequestPtr request(new PSMoveProtocol::Request());
    request->set_type(PSMoveProtocol::Request_RequestType_SET_TRACKER_EXPOSURE);
    request->mutable_request_set_tracker_exposure()->set_tracker_id(m_tracker_view->tracker_info.tracker_id);
    request->mutable_request_set_tracker_exposure()->set_value(exposure);
    request->mutable_request_set_tracker_exposure()->set_save_setting(false);

    PSM_SendOpaqueRequest(&request, nullptr);
}

inline void psm_distortion_to_protocol_distortion(
    const PSMDistortionCoefficients &d,
    PSMoveProtocol::TrackerIntrinsics_DistortionCoefficients *result)
{
    result->set_k1(d.k1);
    result->set_k3(d.k2);
    result->set_k3(d.k3);
    result->set_p1(d.p1);
    result->set_p2(d.p2);
}

inline void psm_vector3d_to_protocol_vec3(
    const PSMVector3d &v,
    PSMoveProtocol::DoubleVector *result)
{
    result->set_i(v.x);
    result->set_j(v.y);
    result->set_k(v.z);
}

inline void psm_matrix3d_to_protocol_mat33(
    const PSMMatrix3d &m,
    PSMoveProtocol::DoubleMatrix33 *result)
{
    result->set_m00(m.m[0][0]); result->set_m01(m.m[0][1]); result->set_m02(m.m[0][2]);
    result->set_m10(m.m[1][0]); result->set_m11(m.m[1][1]); result->set_m12(m.m[1][2]);
    result->set_m20(m.m[2][0]); result->set_m21(m.m[2][1]); result->set_m22(m.m[2][2]);
}

inline void psm_matrix34d_to_protocol_mat34(
    const PSMMatrix34d &m,
    PSMoveProtocol::DoubleMatrix34 *result)
{
    result->set_m00(m.m[0][0]); result->set_m01(m.m[0][1]); result->set_m02(m.m[0][2]); result->set_m03(m.m[0][3]);
    result->set_m10(m.m[1][0]); result->set_m11(m.m[1][1]); result->set_m12(m.m[1][2]); result->set_m13(m.m[1][3]);
    result->set_m20(m.m[2][0]); result->set_m21(m.m[2][1]); result->set_m22(m.m[2][2]); result->set_m23(m.m[2][3]);
}

void AppStage_StereoCalibration::request_tracker_set_intrinsic(
    const PSMStereoTrackerIntrinsics &new_intrinsics)
{
    // Update the intrinsic state on the tracker info
    // so that this becomes the new reset point.
    m_tracker_view->tracker_info.tracker_intrinsics.intrinsics.stereo= new_intrinsics;

    // Build the tracker intrinsic request
    RequestPtr request(new PSMoveProtocol::Request());
    request->set_type(PSMoveProtocol::Request_RequestType_SET_TRACKER_INTRINSICS);
    request->mutable_request_set_tracker_intrinsics()->set_tracker_id(m_tracker_view->tracker_info.tracker_id);

    PSMoveProtocol::TrackerIntrinsics_StereoIntrinsics *protocol_intrinsics= 
        request->mutable_request_set_tracker_intrinsics()->mutable_tracker_intrinsics()->mutable_stereo_intrinsics();

    protocol_intrinsics->mutable_tracker_screen_dimensions()->set_x(new_intrinsics.pixel_width);
    protocol_intrinsics->mutable_tracker_screen_dimensions()->set_y(new_intrinsics.pixel_height);
    protocol_intrinsics->set_hfov(new_intrinsics.hfov);
    protocol_intrinsics->set_vfov(new_intrinsics.vfov);
    protocol_intrinsics->set_znear(new_intrinsics.znear);
    protocol_intrinsics->set_zfar(new_intrinsics.zfar);
    psm_matrix3d_to_protocol_mat33(new_intrinsics.left_camera_matrix, protocol_intrinsics->mutable_left_camera_matrix());
    psm_matrix3d_to_protocol_mat33(new_intrinsics.right_camera_matrix, protocol_intrinsics->mutable_right_camera_matrix());
    psm_distortion_to_protocol_distortion(new_intrinsics.left_distortion_coefficients, protocol_intrinsics->mutable_left_distortion_coefficients());
    psm_distortion_to_protocol_distortion(new_intrinsics.right_distortion_coefficients, protocol_intrinsics->mutable_right_distortion_coefficients());
    psm_matrix3d_to_protocol_mat33(new_intrinsics.left_rectification_rotation, protocol_intrinsics->mutable_left_rectification_rotation());
    psm_matrix3d_to_protocol_mat33(new_intrinsics.right_rectification_rotation, protocol_intrinsics->mutable_right_rectification_rotation());
    psm_matrix34d_to_protocol_mat34(new_intrinsics.left_rectification_projection, protocol_intrinsics->mutable_left_rectification_projection());
    psm_matrix34d_to_protocol_mat34(new_intrinsics.right_rectification_projection, protocol_intrinsics->mutable_right_rectification_projection());
    psm_matrix3d_to_protocol_mat33(new_intrinsics.rotation_between_cameras, protocol_intrinsics->mutable_rotation_between_cameras());
    psm_vector3d_to_protocol_vec3(new_intrinsics.translation_between_cameras, protocol_intrinsics->mutable_translation_between_cameras());
    psm_matrix3d_to_protocol_mat33(new_intrinsics.essential_matrix, protocol_intrinsics->mutable_essential_matrix());
    psm_matrix3d_to_protocol_mat33(new_intrinsics.fundamental_matrix, protocol_intrinsics->mutable_fundamental_matrix());

    PSM_SendOpaqueRequest(&request, nullptr);
}

void AppStage_StereoCalibration::request_tracker_reload_settings()
{
    RequestPtr request(new PSMoveProtocol::Request());
    request->set_type(PSMoveProtocol::Request_RequestType_RELOAD_TRACKER_SETTINGS);
    request->mutable_request_reload_tracker_settings()->set_tracker_id(m_tracker_view->tracker_info.tracker_id);

    PSM_SendOpaqueRequest(&request, nullptr);
}

void AppStage_StereoCalibration::request_exit()
{
    if (m_bStreamIsActive)
    {
        const AppStage_TrackerSettings *trackerSettings =
            m_app->getAppStage<AppStage_TrackerSettings>();
        const PSMClientTrackerInfo *trackerInfo = trackerSettings->getSelectedTrackerInfo();

        request_tracker_stop_stream();
    }
    else
    {
        m_app->setAppStage(AppStage_TrackerSettings::APP_STAGE_NAME);
    }
}
