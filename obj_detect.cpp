#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <errno.h>
#include <stdint.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <netdb.h>
#include <dirent.h>
#include <fnmatch.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MyDetector.h"

#include <map>
#include "CAxcThread.h"
#include "config_file.h"
#include "obj_detect.h"

#ifndef MAX_PATH
#define MAX_PATH (256)
#endif

typedef int SOCKET;
#define SOCKET_ERROR (-1)
#define INVALID_SOCKET (-1)
#define closesocket close
#ifndef SD_BOTH
#define SD_BOTH 2
#endif

typedef struct _OBJ_BOX
{
    int type;
    float score;
    int x;
    int y;
    int w;
    int h;
    int track_id;
    char name[64];
} OBJ_BOX;

extern bool g_bAppRunning;

#define MAX_CLASSES 9
static char g_classes_list[MAX_CLASSES][64] = {
    {
        0,
    },
};
static int g_classes_num = 0;
static char g_classes[1024];

static int g_iBatch = 1;
static float g_fThreshold = 0.3;

#define MAX_OBJ_NUM 100
static OBJ_BOX g_boxs[MAX_OBJ_NUM];
static int g_box_num = 0;
static bool g_isLoaded = false;
static MyDetector g_detector;

static int g_fd = -1;
static int g_width = 0;
static int g_height = 0;
static bool g_enable_capture = false;
static bool g_enable_detect = false;
static CAxcThread g_threadCap;
static pthread_t g_ddwCapThreadId = 0;
static uint32_t Thread_ObjDetect_Cap(void *pContext);
static CAxcThread g_threadDetect;
static pthread_t g_ddwDetectThreadId = 0;
static uint32_t Thread_ObjDetect_Detect(void *pContext);

typedef enum
{
    IMAGE_IDLE,
    IMAGE_WRITING,
    IMAGE_WRITED,
    IMAGE_READING
} E_IMAGE_STATE;
static E_IMAGE_STATE g_eRgb24ImageState = IMAGE_IDLE;
static cv::Mat g_matRgb32; // read from fifo
static cv::Mat g_matRgb24; // convert from rgb32

static double GetCurrentUTCTimeSec()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_sec) + (((double)tv.tv_usec) / 1000000.0);
}
static uint64_t GetCurrentUTCTimeMs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, NULL);
    return ((uint64_t)tv.tv_sec) * 1000 + (tv.tv_usec / 1000);
}
static inline void ThreadSleepMs(uint32_t ms)
{
    usleep(ms * 1000);
}
static bool AccessCheck(const char* szFileName, int iMode)
{
	if (NULL == szFileName || 0 == szFileName[0] || iMode < 0)
	{
		errno = EINVAL;
		return false;
	}
	return (0 == access(szFileName, iMode));
}
static inline bool AccessCheck_IsExisted(const char* szFileName) { return AccessCheck(szFileName, 0); }
static inline bool AccessCheck_CanExecute(const char* szFileName) { return AccessCheck(szFileName, 1); }
static inline bool AccessCheck_CanWrite(const char* szFileName) { return AccessCheck(szFileName, 2); }
static inline bool AccessCheck_CanRead(const char* szFileName) { return AccessCheck(szFileName, 4); }
static inline bool AccessCheck_CanReadWrite(const char* szFileName) { return AccessCheck(szFileName, 2 | 4); }


bool obj_detect_load()
{
    if (!g_isLoaded)
    {
        // auto run 'jetson_clocks' when app startup
        const char JetsonClocksCmd[] = {"/usr/bin/jetson_clocks"};
        if (AccessCheck_IsExisted(JetsonClocksCmd))
        {
            int ret = system(JetsonClocksCmd);
            LOG("[%s] run command '%s' ret '%d'\n", __func__, JetsonClocksCmd, ret);
        }

        // if the previous load failed, no more attempts will be made
        char szModelConfigFile[MAX_PATH + 32] = {0};
        sprintf(szModelConfigFile, "%s/model.conf", OBJ_DETECT_MODEL_FOLDER_DEFAULT);
        if (!AccessCheck_IsExisted(szModelConfigFile))
        {
            LOGE("[%s] config file NOT existed: '%s'\n", __func__, szModelConfigFile);
            return false;
        }
        const int iPreviousLoadOk = config_file_get_int("load_ok", -1, szModelConfigFile);
        if (iPreviousLoadOk == 0 || iPreviousLoadOk < -4)
        {
            if (iPreviousLoadOk != 0)
            {
                config_file_set("load_ok", "0", szModelConfigFile);
            }
            LOGE("[%s] previous load failed (load_ok=%d), file: '%s'\n", __func__, iPreviousLoadOk, szModelConfigFile);
            return false;
        }
        // model filename
        char szModelName[256] = {0};
        config_file_get_str("name", szModelName, sizeof(szModelName), "", szModelConfigFile);
        if (szModelName[0] == 0)
        {
            LOGE("[%s] empty model name, file: '%s'\n", __func__, szModelConfigFile);
            return false;
        }
        // model full path filename
        char szRtFile[MAX_PATH + 32 + 256];
        sprintf(szRtFile, "%s/%s", OBJ_DETECT_MODEL_FOLDER_DEFAULT, szModelName);
        if (!AccessCheck_IsExisted(szRtFile))
        {
            LOGE("[%s] model file NOT existed: '%s'\n", __func__, szRtFile);
            return false;
        }

        // model batch
        const int iBatch = config_file_get_int("batch", 1, szModelConfigFile);
        if (iBatch <= 0 || iBatch > 256)
        {
            LOGE("[%s] invalid batch: %d\n", __func__, iBatch);
            return false;
        }
        g_iBatch = iBatch;
        // model threshold
        const int iThresholdPercent = config_file_get_int("threshold", 30, szModelConfigFile);
        if (iThresholdPercent <= 0 || iThresholdPercent > 100)
        {
            LOGE("[%s] invalid threshold: %d%%\n", __func__, iThresholdPercent);
            return false;
        }
        g_fThreshold = iThresholdPercent * 0.01f;
        // classes name list
        char szAllNames[1024] = {0};
        config_file_get_str("classes", szAllNames, sizeof(szAllNames), "", szModelConfigFile);
        strcpy(g_classes, szAllNames);
        char *pszClassName = strtok(szAllNames, ",");
        g_classes_num = 0;
        memset(g_classes_list, 0, sizeof(g_classes_list));
        while (pszClassName != NULL && g_classes_num < MAX_CLASSES)
        {
            if (pszClassName[0] != 0)
            {
                strncpy(g_classes_list[g_classes_num], pszClassName, sizeof(g_classes_list[g_classes_num]));
                g_classes_num++;
            }
            pszClassName = strtok(NULL, ",");
        }
        if (g_classes_num <= 0)
        {
            LOGE("[%s] invalid classes number: %d\n", __func__, g_classes_num);
            return false;
        }

        // 1. set 'load_ok' to 0
        // 2. load model
        // 3. if load model succeeded, set 'load_ok' to 1
        char szLoadOk[32] = {0};
        sprintf(szLoadOk, "%d", iPreviousLoadOk < 0 ? (iPreviousLoadOk - 1) : -1);
        config_file_set("load_ok", szLoadOk, szModelConfigFile);

        // load model
        // try
        //{
        if (!g_detector.init(szRtFile)) //, g_classes_num, g_iBatch, g_fThreshold))
        {
            LOGE("[%s] failed to create AI detector, error: %d, file: '%s', retry '%s'\n", __func__, errno, szRtFile, szLoadOk);
            g_isLoaded = false;
            return false;
        }
        else
        {
            LOG("[%s] load AI detector OK: batch %d, threshold %.2f, classes %d, model '%s', retry '%s'\n", __func__, g_iBatch, g_fThreshold, g_classes_num, szRtFile, szLoadOk);
            config_file_set("load_ok", "1", szModelConfigFile);
            g_isLoaded = true;
        }
        //}
        // catch(...)
        //{
        //    LOGE("[%s] fault when load AI detector, error: %d, file: %s\n", __func__, errno, szRtFile);
        //    g_isLoaded = false;
        //}
    }
    return g_isLoaded;
}

void obj_detect_unload()
{
    if (g_isLoaded)
    {
        obj_detect_stop();
        obj_detect_capture_stop();
        g_isLoaded = false;
    }
}

bool obj_detect_is_loaded()
{
    return g_isLoaded;
}

int obj_detect_on_timer()
{
    return 0;
}

int obj_detect_get_classes_number()
{
    return g_classes_num;
}
const char *obj_detect_get_classes_name(int index)
{
    if (index < 0 || index >= g_classes_num)
    {
        static char s_unknown[] = {"unknown"};
        return s_unknown;
    }
    return g_classes_list[index];
}

bool obj_detect_start()
{
    g_enable_detect = true;
    return true;
}
void obj_detect_stop()
{
    g_enable_detect = false;
}
bool obj_detect_is_enable()
{
    return g_enable_detect;
}

bool obj_detect_capture_start(int width, int height)
{
    if (!obj_detect_is_loaded())
    {
        LOGE("[%s] NOT ready !\n", __func__);
        return false;
    }
    obj_detect_stop();

    if (width > 0 && height > 0)
    {
        // make fifo
        unlink(OBJ_DETECT_FIFO_FILE);
        if (mkfifo(OBJ_DETECT_FIFO_FILE, 0777) < 0)
        {
            LOGE("[%s] failed to mkfifo, error = %d\n", __func__, errno);
            return false;
        }

        // create image buffer
        g_width = width;
        g_height = height;
        g_matRgb32.create(g_height, g_width, CV_8UC4);
        g_matRgb24.create(g_height, g_width, CV_8UC3);
        if (g_matRgb32.empty() || g_matRgb24.empty())
        {
            LOG("[%s] failed to create image frame, error: %d\n", __func__, errno);
            return false;
        }
        g_eRgb24ImageState = IMAGE_IDLE;

        // create threads
        g_enable_capture = true;
        g_ddwCapThreadId = 0;
        g_ddwDetectThreadId = 0;
        if (!g_threadCap.Create(Thread_ObjDetect_Cap, NULL) ||
            !g_threadDetect.Create(Thread_ObjDetect_Detect, NULL))
        {
            LOG("[%s] failed to create thread, error: %d\n", __func__, errno);
            g_enable_capture = false;
            for (int i = 0; i < 2; i++)
            {
                if (g_fd >= 0)
                {
                    close(g_fd);
                    g_fd = -1;
                }
            }
            return false;
        }
    }
    else
    {
        g_enable_capture = false;
        LOGE("[%s] invalidate parameters !\n", __func__);
        return false;
    }

    return true;
}

void obj_detect_capture_stop()
{
    g_enable_capture = false;
    if (g_threadCap.IsValid() && g_ddwCapThreadId != pthread_self())
    {
        ThreadSleepMs(100);
        g_threadCap.Destroy(1000);
        g_ddwCapThreadId = 0;
    }
    if (g_threadDetect.IsValid() && g_ddwDetectThreadId != pthread_self())
    {
        ThreadSleepMs(100);
        g_threadDetect.Destroy(1000);
        g_ddwDetectThreadId = 0;
    }
    if (g_fd >= 0)
    {
        ThreadSleepMs(100);
        close(g_fd);
        g_fd = -1;
        ThreadSleepMs(100);
    }
    if (!g_matRgb32.empty())
    {
        g_matRgb32.release();
    }
    if (!g_matRgb24.empty())
    {
        g_matRgb24.release();
    }
    g_eRgb24ImageState = IMAGE_IDLE;
}

static uint32_t Thread_ObjDetect_Cap(void *pContext)
{
    LOG("[%s] begin\n", __func__);

    g_ddwCapThreadId = pthread_self();

    const int BITCOUNT = 32;
    const int IMAGESIZE = g_width * g_height * BITCOUNT / 8;

    time_t statBeginTime = time(NULL);
    int statGoodFrames = 0;
    int statBadFrames = 0;
    int statUsedFrames = 0;

    time_t tLastPacketTime = 0;
    char RecvBuffer[64 * 1024]; // MUST BE 64KB
    int iRecvLen = 0;
    const size_t BUFFERSIZE = sizeof(RecvBuffer);
    int iFrameLen = 0;

    while (g_bAppRunning && g_enable_capture)
    {
        if ((time(NULL) - statBeginTime) >= 10)
        {
            const int sec = (int)(time(NULL) - statBeginTime);
            LOG("[%s] capture farmes stat %d sec: %d bad, %d good (%d used)\n", __func__, sec, statBadFrames, statGoodFrames, statUsedFrames);
            statBeginTime = time(NULL);
            statGoodFrames = 0;
            statBadFrames = 0;
            statUsedFrames = 0;
        }

        if (g_fd < 0)
        {
            if ((time(NULL) - tLastPacketTime) >= 3)
            {
                g_fd = open(OBJ_DETECT_FIFO_FILE, O_RDONLY | O_NONBLOCK);
                tLastPacketTime = time(NULL);
            }
            if (g_fd < 0)
            {
                ThreadSleepMs(100);
                continue;
            }
            LOG("[%s] open obj-detect fifo '%s' fd '%d' \n", __func__, OBJ_DETECT_FIFO_FILE, g_fd);
        }

        iRecvLen = read(g_fd, RecvBuffer, BUFFERSIZE);
        if (iRecvLen <= 0)
        {
            ThreadSleepMs(1);
            if ((time(NULL) - tLastPacketTime) >= 3)
            {
                // restart capture !!!
                printf("[%s] close obj-detect fifo file '%d' because NO data \n", __func__, g_fd);
                close(g_fd);
                g_fd = -1;
                tLastPacketTime = time(NULL);
            }
            continue;
        }
        tLastPacketTime = time(NULL);
        if ((iFrameLen + iRecvLen) > IMAGESIZE)
        {
            LOGE("[%s] invalid frame length: too large\n", __func__);
        }
        else
        {
            memcpy(g_matRgb32.data + iFrameLen, RecvBuffer, iRecvLen);
            iFrameLen += iRecvLen;
        }

        if (iRecvLen != BUFFERSIZE)
        {
            if (iFrameLen == IMAGESIZE)
            {
                // copy to rgb24 image for obj_detect and obj_tracker
                if (g_eRgb24ImageState == IMAGE_IDLE)
                {
                    g_eRgb24ImageState = IMAGE_WRITING;
                    cvtColor(g_matRgb32, g_matRgb24, cv::COLOR_RGBA2RGB);
                    g_eRgb24ImageState = IMAGE_WRITED;
                    statUsedFrames++;
                }
                statGoodFrames++;
            }
            else
            {
                // LOGE("[%s] invalid frame length: recv-packet-len %d, total-frame-len %d\n", __func__, iRecvLen, iFrameLen);
                statBadFrames++;
            }
            iFrameLen = 0;
        }
    }

    if (g_fd >= 0)
    {
        close(g_fd);
        g_fd = -1;
    }

    LOG("[%s] end: OK\n", __func__);
    return 0;
}

static uint32_t Thread_ObjDetect_Detect(void *pContext)
{
    LOG("[%s] begin\n", __func__);

    g_ddwDetectThreadId = pthread_self();

    time_t statBeginTime = time(NULL);
    int statUsedFrames = 0;
    double statUsedTimeDet = 0;
    double statUsedTimeCb = 0;

    memset(g_boxs, 0, sizeof(g_boxs));
    g_box_num = 0;

    // send result to local udp port
    SOCKET localUdp = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	struct sockaddr_in local_sin;
	local_sin.sin_family = AF_INET;
	local_sin.sin_addr.s_addr = 0;
	local_sin.sin_port = 0;
	int ret = bind(localUdp, (const sockaddr *)&local_sin, sizeof(local_sin));
    LOG("[%s] bind udp socket(#%d) to local ip(0) result(%d)\n", __func__, localUdp, ret);
    //
    char szSendBuffer[1024];

    while (g_bAppRunning)
    {
        if ((time(NULL) - statBeginTime) >= 10)
        {
            const int sec = (int)(time(NULL) - statBeginTime);
            LOG("[%s] detect stat %d sec: %d frames, detect %.1f ms (%.1f ms/f), callback %.1f ms (%.1f ms/f)\n",
                __func__, sec, statUsedFrames,
                statUsedTimeDet, statUsedFrames > 0 ? (statUsedTimeDet / statUsedFrames) : 0.0,
                statUsedTimeCb, statUsedFrames > 0 ? (statUsedTimeCb / statUsedFrames) : 0.0);
            statBeginTime = time(NULL);
            statUsedFrames = 0;
            statUsedTimeDet = 0;
            statUsedTimeCb = 0;
        }

        ThreadSleepMs(3);

        if (g_eRgb24ImageState == IMAGE_WRITED)
        {
            g_eRgb24ImageState = IMAGE_READING;

            // detector
            double t1 = GetCurrentUTCTimeSec();
            int num = 0;
            if (g_enable_detect)
            {
                const float x_ratio = g_detector.modelInputWidth() / ((float)g_matRgb24.cols);
                const float y_ratio = g_detector.modelInputHeight() / ((float)g_matRgb24.rows);

                // detect
                g_detector.update(g_matRgb24, g_fThreshold);
                for (size_t i = 0; i < g_detector.detected.size() && num < MAX_OBJ_NUM; i++)
                {
                    const uint8_t class_id = g_detector.detected[i].class_id;
                    if (class_id >= 0 && class_id < g_classes_num)
                    {
                        g_boxs[num].type = class_id;
                        g_boxs[num].score = g_detector.detected[i].score;
                        g_boxs[num].track_id = -1;
                        g_boxs[num].x = g_detector.detected[i].x / x_ratio;
                        g_boxs[num].y = g_detector.detected[i].y / y_ratio;
                        g_boxs[num].w = g_detector.detected[i].width / x_ratio;
                        g_boxs[num].h = g_detector.detected[i].height / y_ratio;
                        strncpy(g_boxs[num].name, g_classes_list[class_id], sizeof(g_boxs[num].name));
                        num++;
                    }
                }
            }
            g_box_num = num;
            double t2 = GetCurrentUTCTimeSec();
            statUsedTimeDet += (t2 - t1) * 1000;

            g_eRgb24ImageState = IMAGE_IDLE;

            // send detect result to local udp port
            if (g_enable_detect)
            {
                char* psz = szSendBuffer;
                psz += sprintf(psz, "time=%.3f\r\n", GetCurrentUTCTimeSec());
                psz += sprintf(psz, "image_width=%d\r\n", g_width);
                psz += sprintf(psz, "image_height=%d\r\n", g_height);
                psz += sprintf(psz, "box_num=%d\r\n", num);
                for (int i = 0; i < num; i++)
                {
                    psz += sprintf(psz, "box%d=%d,%.2f,%d,%d,%d,%d,%s,%d\r\n", i, g_boxs[i].type, g_boxs[i].score, g_boxs[i].x, g_boxs[i].y, g_boxs[i].w, g_boxs[i].h, g_boxs[i].name, g_boxs[i].track_id);
                }
                psz += sprintf(psz, "\r\n");

                struct sockaddr_in  remoteAddr;
                remoteAddr.sin_family = AF_INET;
                remoteAddr.sin_port = htons(OBJ_DETECT_RESULT_UDP_SENDTO_PORT);
                remoteAddr.sin_addr.s_addr = htonl(OBJ_DETECT_RESULT_UDP_SENDTO_IP);
                sendto(localUdp, szSendBuffer, strlen(szSendBuffer)+1, 0, (struct sockaddr *)&remoteAddr, sizeof(remoteAddr));
            }

            double t3 = GetCurrentUTCTimeSec();
            statUsedTimeCb += (t3 - t2) * 1000;

            statUsedFrames++;
        }
    }

    // free udp socket
    shutdown(localUdp, SD_BOTH);
    closesocket(localUdp);
    localUdp = INVALID_SOCKET;

    LOG("[%s] end: OK\n", __func__);
    return 0;
}
