#ifndef SHARED_TRACKER_STATE_H
#define SHARED_TRACKER_STATE_H

#ifdef WIN32
#define BOOST_INTERPROCESS_SHARED_DIR_PATH "shared_mem"
#endif // WIN32

#include <boost/interprocess/sync/interprocess_mutex.hpp>

class SharedVideoFrameHeader
{
public:
    SharedVideoFrameHeader()
        : mutex()
        , width(0)
        , height(0)
        , stride(0)
        , section_count(1)
        , frame_index(0)
    {
    }
    
    //Mutex to protect access to the shared memory
    boost::interprocess::interprocess_mutex mutex;

    int width;
    int height;
    int stride;
    int section_count;
    int frame_index;
    // Buffer stored past the end of the header

    const unsigned char *getBuffer(int section_index) const
    {
        return reinterpret_cast<const unsigned char *>(this) 
            + sizeof(SharedVideoFrameHeader)
            + computeVideoBufferSize(section_index, stride, height);
    }

    unsigned char *getBufferMutable(int section_index)
    {
        return const_cast<unsigned char *>(getBuffer(section_index));
    }

    static size_t computeVideoSectionSize(int stride, int height)
    {
        return stride*height;
    }

    static size_t computeVideoBufferSize(int section_count, int stride, int height)
    {
        return section_count*computeVideoSectionSize(stride, height);
    }

    static size_t computeTotalSize(int section_count, int stride, int height)
    {
        return sizeof(SharedVideoFrameHeader) + computeVideoBufferSize(section_count, stride, height);
    }
};

#endif // SHARED_TRACKER_STATE_H