#include "MyDetector.h"
#include <iostream>
#include <fstream>

class MyLogger : public nvinfer1::ILogger
{
public:
    explicit MyLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING) : severity_(severity) {}

    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override
    {
        if (severity <= severity_)
        {
            std::cerr << msg << std::endl;
        }
    }
    nvinfer1::ILogger::Severity severity_;
};
static MyLogger logger;

// static bool cmp(MyDetector::Object &obj1, MyDetector::Object &obj2)
// {
//     return obj1.score > obj2.score;
// }
//
// static float sigmoid(float x)
// {
//     return 1.0 / (exp(-x) + 1.0);
// }

static float iou_of(const MyDetector::Object &obj1, const MyDetector::Object &obj2)
{
    float x1_lu = obj1.x;
    float y1_lu = obj1.y;
    float x1_rb = x1_lu + obj1.width;
    float y1_rb = y1_lu + obj1.height;
    float x2_lu = obj2.x;
    float y2_lu = obj2.y;
    float x2_rb = x2_lu + obj2.width;
    float y2_rb = y2_lu + obj2.height;

    float i_x1 = std::max(x1_lu, x2_lu);
    float i_y1 = std::max(y1_lu, y2_lu);

    float i_x2 = std::min(x1_rb, x2_rb);
    float i_y2 = std::min(y1_rb, y2_rb);

    float i_w = i_x2 - i_x1;
    float i_h = i_y2 - i_y1;

    float o_x1 = std::min(x1_lu, x2_lu);
    float o_y1 = std::min(y1_lu, y2_lu);

    float o_x2 = std::max(x1_rb, x2_rb);
    float o_y2 = std::max(y1_rb, y2_rb);

    float o_w = o_x2 - o_x1;
    float o_h = o_y2 - o_y1;

    return (i_w * i_h) / (o_w * o_h);
}

static std::vector<int> hardNMS(std::vector<MyDetector::Object> &input, std::vector<MyDetector::Object> &output, float iou_threshold, unsigned int topk)
{
    const unsigned int box_num = input.size();
    std::vector<int> merged(box_num, 0);
    std::vector<int> indices;

    if (input.empty())
        return indices;
    std::vector<MyDetector::Object> res;
    // sort by score
    std::sort(input.begin(), input.end(),
              [](const MyDetector::Object &a, const MyDetector::Object &b)
              { return a.score > b.score; });

    unsigned int count = 0;
    for (unsigned int i = 0; i < box_num; ++i)
    {
        if (merged[i])
            continue;
        MyDetector::Object buf;
        buf = input[i];
        merged[i] = 1; // delete current bbox

        for (unsigned int j = i + 1; j < box_num; ++j)
        {
            if (merged[j])
                continue;

            float iou = static_cast<float>(iou_of(input[j], input[i]));
            if (iou > iou_threshold)
            {
                merged[j] = 1;
            }
        }
        indices.push_back(i);
        res.push_back(buf);

        // keep top k
        count += 1;
        if (count >= topk)
            break;
    }
    output.swap(res);

    return indices;
}

MyDetector::MyDetector()
{
    initialized = false;
    runtime = nullptr;
    engine = nullptr;
    context = nullptr;
    input_name = "";
    input_width = 0;
    input_height = 0;
    output_name = "";
    output_width = 0;
    output_height = 0;
    buffers[0] = nullptr;
    buffers[1] = nullptr;
    input_size = 0;
    output_size = 0;
    input_blob = nullptr;
    output_blob = nullptr;
    output_buffer = nullptr;
}

MyDetector::~MyDetector()
{
    if (context)
    {
        delete context;
        context = nullptr;
    }
    if (engine)
    {
        delete engine;
        engine = nullptr;
    }
    if (runtime)
    {
        delete runtime;
        runtime = nullptr;
    }
    if (buffers[0])
    {
        cudaFree(buffers[0]);
        buffers[0] = nullptr;
    }
    if (buffers[1])
    {
        cudaFree(buffers[1]);
        buffers[1] = nullptr;
    }
    if (output_blob)
    {
        delete[] output_blob;
        output_blob = nullptr;
    }
    if (output_buffer)
    {
        delete[] output_buffer;
        output_buffer = nullptr;
    }
    if (input_blob)
    {
        delete[] input_blob;
        input_blob = nullptr;
    }
    cudaDeviceReset();
    initialized = false;
}

bool MyDetector::init(const std::string &engine_file)
{
    if (initialized)
        return true;

    //
    // laod model
    //
    std::cout << "load TensorRT engine file: " << engine_file << std::endl;
    if (engine_file.empty())
    {
        std::cout << "engine file is empty!" << std::endl;
        return false;
    }
    std::stringstream engine_file_stream;
    engine_file_stream.seekg(0, engine_file_stream.beg);
    std::ifstream ifs(engine_file);
    if (!ifs.is_open())
    {
        std::cout << "read engine file error!" << std::endl;
        return false;
    }
    engine_file_stream << ifs.rdbuf();
    ifs.close();

    engine_file_stream.seekg(0, std::ios::end);
    const int model_size = engine_file_stream.tellg();
    engine_file_stream.seekg(0, std::ios::beg);
    void *model_mem = malloc(model_size);
    if (model_mem == nullptr)
    {
        std::cout << "malloc model_mem failed!" << std::endl;
        return false;
    }
    engine_file_stream.read(static_cast<char *>(model_mem), model_size);

    runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr)
    {
        free(model_mem);
        model_mem = nullptr;
        std::cout << "createInferRuntime failed!" << std::endl;
        return false;
    }

    engine = runtime->deserializeCudaEngine(model_mem, model_size);
    if (engine == nullptr)
    {
        free(model_mem);
        model_mem = nullptr;
        std::cout << "deserializeCudaEngine failed!" << std::endl;
        return false;
    }

    free(model_mem);
    model_mem = nullptr;

    //
    // infer context
    //
    context = engine->createExecutionContext();

    // enum names, find input and output name
    input_name = "";
    input_width = 0;
    input_height = 0;
    input_batch = 0;
    output_name = "";
    output_width = 0;
    output_height = 0;
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char *tensorName = engine->getIOTensorName(i);
        if (tensorName && tensorName[0])
        {
            nvinfer1::Dims dims = context->getTensorShape(tensorName);
            std::cout << "Tensor #" << i << ": name [" << tensorName << "], dims [";
            for (int i = 0; i < dims.nbDims; ++i)
            {
                std::cout << dims.d[i] << ",";
            }
            std::cout << "]" << std::endl;
            // input dims [1,3,640,640] or [1,3,480,640]
            // name like "images"
            if(dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3 && (dims.d[2] == 640 || dims.d[2] == 480) && dims.d[3] == 640)
            {
                input_name = tensorName;
                input_batch = dims.d[1];
                input_height = dims.d[2];
                input_width = dims.d[3];
            }
            // output dims [1,84,8400] or [1,84,6300]
            // name like "output0"
            // output height is [class_number + 4]
            if(dims.nbDims == 3 && dims.d[0] == 1 && dims.d[1] > 4 && (dims.d[2] == 8400 || dims.d[2] == 6300))
            {
                output_name = tensorName;
                output_height = dims.d[1];
                output_width = dims.d[2];
            }
        }
    }
    std::cout << "Tensor: input [" << input_name << "," << input_width << "x" << input_height << "], output [" << output_name << "," << output_width << "x" << output_height << "]" << std::endl;
    if(input_name == "" || input_batch <= 0 || input_width <= 0 || input_height <= 0 || output_name == "" || output_width <= 0 || output_height <= 0)
    {
        std::cout << "!!! Not found input or output tensor !!!" << std::endl;
        return false;
    }

    // get input size and alloc cuda memory
    input_size = input_batch * input_width * input_height;
    if (input_size < 5)
    {
        std::cout << "Input size error" << std::endl;
        return false;
    }
    cudaMalloc(&buffers[0], input_size * sizeof(float));
    if (!buffers[0])
    {
        std::cout << "cudaMalloc input buffer failed!" << std::endl;
        return false;
    }

    // get output size and alloc cuda memory
    output_size = output_width * output_height;
    if (output_size < 5)
    {
        std::cout << "Output size (" << output_size << ") error" << std::endl;
        return false;
    }
    cudaMalloc(&buffers[1], output_size * sizeof(float));
    if (!buffers[1])
    {
        std::cout << "cudaMalloc output buffer failed!" << std::endl;
        return false;
    }

    // alloc cpu memory
    input_blob = new float[input_height * input_width * 3];
    if (!input_blob)
    {
        std::cout << "malloc input_blob failed!" << std::endl;
        return false;
    }
    output_blob = new float[output_size];
    if (!output_blob)
    {
        std::cout << "malloc output_blob failed!" << std::endl;
        return false;
    }
    output_buffer = new float[output_height * output_width];
    if (!output_buffer)
    {
        std::cout << "malloc output_buffer failed!" << std::endl;
        return false;
    }

    initialized = true;
    return true;
}

// detection
void MyDetector::update(const cv::Mat &rgb24_image, float score_threshold)
{
    detected.clear();

    if (!initialized)
        return;

    if (rgb24_image.empty() || rgb24_image.channels() != 3)
    {
        std::cout << "input image is empty or not rgb format!" << std::endl;
        return;
    }

    //
    // normalization and hwc->chw
    //
    const int channels = rgb24_image.channels();
    const int width = rgb24_image.cols;
    const int height = rgb24_image.rows;
    if (channels * width * height != input_size)
    {
        std::cout << "input image size not match!" << std::endl;
        return;
    }
    float *blob1 = input_blob;
    float *blob2 = blob1 + height * width;
    float *blob3 = blob2 + height * width;
    for (int h = 0; h < height; h++)
    {
        const unsigned char *src = rgb24_image.ptr<unsigned char>(h);
        for (int w = 0; w < width; w++, blob1++, blob2++, blob3++, src += 3)
        {
            *blob1 = src[0] / 255.0f;
            *blob2 = src[1] / 255.0f;
            *blob3 = src[2] / 255.0f;
        }
    }

    //
    // infer and copy data
    //
    cudaMemcpy(buffers[0], input_blob, input_size * sizeof(float), cudaMemcpyHostToDevice);
    if (context->executeV2(buffers))
    {
        //std::cout << "executeV2 infer succeeded" << std::endl;
    }
    else
    {
        std::cout << "executeV2 infer failed" << std::endl;
    }
    cudaMemcpy(output_blob, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost);

    //
    // prase output
    //
    float *ptr = output_blob; // 1x84x8400  =  705600
    for (int i = 0; i < (output_height * output_width); i++)
    {
        output_buffer[(i % output_width) * output_height + i / output_width] = *ptr;
        ptr++;
    }
    std::vector<MyDetector::Object> objs;
    ptr = output_buffer;
    for (int i = 0; i < output_width; i++, ptr += output_height)
    {
        float *max_score_ptr = std::max_element(ptr + 4, ptr + output_height); // address of max score
        if (*max_score_ptr >= score_threshold)
        {
            const float bx = ptr[0];
            const float by = ptr[1];
            const float bw = ptr[2];
            const float bh = ptr[3];
            MyDetector::Object obj;
            obj.x = bx - bw / 2;
            obj.y = by - bh / 2;
            obj.width = bw;
            obj.height = bh;
            obj.class_id = (int)(max_score_ptr - (ptr + 4));
            obj.score = *max_score_ptr;
            objs.push_back(std::move(obj));
        }
    }

    //
    // NMS
    //
    hardNMS(objs, detected, 0.6, 10);
}
