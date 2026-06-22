#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <memory.h>
#include <errno.h>
#include <signal.h>

#include "obj_detect.h"

bool g_bAppRunning = false;
static void exitapp(int signo)
{
    LOG("[main] app will exit for signal %d\n", signo);
    g_bAppRunning = false;
}

int main(int argc, char* argv[])
{
    // load module
    if(!obj_detect_load())
    {
        LOG("[main] FAILED to load module !\n");
        return -1;
    }

    // start capture video
    if(!obj_detect_capture_start(OBJ_DETECT_WIDTH, OBJ_DETECT_HEIGHT))
    {
        LOG("[main] FAILED to start capture !\n");
        return -1;
    }

    // start detect
    if(!obj_detect_start())
    {
        LOG("[main] FAILED to start detect !\n");
        return -1;
    }

	// capture 'ctrl+c' and 'kill' signal
	signal(SIGINT, exitapp);
	signal(SIGTERM, exitapp);
    signal(SIGPIPE, SIG_IGN); // ignore SIGPIPE

    // main loop
    g_bAppRunning = true;
    LOG("[main] task running ...\n");
    while(g_bAppRunning)
    {
        obj_detect_on_timer(); // timer handle
        usleep(10*1000); // sleep 10 ms
    }
    LOG("[main] task END\n");

    // stop capture
    obj_detect_capture_stop();

    // stop detect
    obj_detect_stop();

    // unload module
    obj_detect_unload();

    LOG("[main] application EXIT\n");
    return 0;
}
