# gmb-object-detector

## Function

The background program running on NVIDIA Jetson ORIN NX.   
Load YOLO11 or other TensorRT models, read real-time image streams from external FIFO file, send them to the model for object detection, and send the detection results to the local UDP port for external programs to read.

## Process

1. By default, the model configuration file **model. conf** is read from the **od_madel_0** sub-folder of the program and the model file is loaded.
2. Create the capture thread to read RGB32 images from the FIFO file **/tmp/fifo-detect-image.rgb** and convert them to RGB24
3. Create the detection thread, write RGB24 to the model for object detection, and send the detection results to the **127.0.0.1:25555** UDP port

## License

The source code of this project follows the GNU AGPL v3.0 protocol. Please refer to the [LICENSE] (./LICENSE.txt) document for details.
