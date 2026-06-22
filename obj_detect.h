#ifndef _OBJ_DETECT_H
#define _OBJ_DETECT_H

// object detect source video
#define OBJ_DETECT_WIDTH     640
#define OBJ_DETECT_HEIGHT    480
#define OBJ_DETECT_FIFO_FILE "/tmp/fifo-detect-image.rgb" // input image RGB32 streaming

// object detect model number
#define OBJ_DETECT_MODEL_NUMBER (6)
#define OBJ_DETECT_MODEL_FOLDER_DEFAULT "od_model_0" // 0 ~ 5

// send detect result to local udp port
#define OBJ_DETECT_RESULT_UDP_SENDTO_PORT (25555)
#define OBJ_DETECT_RESULT_UDP_SENDTO_IP   (0x7F000001) // 127.0.0.1

#define LOG printf
#define LOGE printf

bool obj_detect_load();
void obj_detect_unload();
bool obj_detect_is_loaded();
int  obj_detect_on_timer();

int  obj_detect_get_classes_number();
const char* obj_detect_get_classes_name(int index);

bool obj_detect_capture_start(int width, int height);
void obj_detect_capture_stop();

bool obj_detect_start();
void obj_detect_stop();
bool obj_detect_is_enable();

#endif // _OBJ_DETECT_H
