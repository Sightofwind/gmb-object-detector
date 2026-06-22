#ifndef _MY_DETECTOR_H_
#define _MY_DETECTOR_H_

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <NvInfer.h>

class MyDetector
{
public:
    MyDetector();
    ~MyDetector();

    bool init(const std::string &engine_file);
    bool isInitialized() { return initialized; }
    int modelInputWidth()  { return input_width; }
    int modelInputHeight() { return input_height; }
    int modelClassNumber() { return output_height > 4 ? (output_height-4) : 0; }

    void update(const cv::Mat &rgb24_image, float score_threshold = 0.35f);

    typedef struct
    {
        int class_id;
        float score;
        int x; // left
        int y; // top
        int width;
        int height;
    } Object;
    std::vector<Object> detected;

private:
    bool initialized;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
    std::string input_name;
    int input_width;
    int input_height;
    int input_batch;
    std::string output_name;
    int output_width;
    int output_height; // class_number + 4
    void *buffers[2];
    int input_size;
    int output_size;
    float *input_blob;
    float *output_buffer;
    float *output_blob;
};

#endif // _MY_DETECTOR_H_
