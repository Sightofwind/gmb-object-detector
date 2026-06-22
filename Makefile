GPP = g++


#
# source files and path
#
INCLUDE = -I./ -I/usr/local/cuda/include -I/usr/local/cuda/targets/aarch64-linux/include `pkg-config --cflags-only-I opencv4`
DEFINE = -DOPENCV
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lpthread -lcudart -lnvinfer `pkg-config --libs opencv4`
OUTPUT_PATH = .
EXEC = gmb-object-detector
CFLAGS += -O2 -s -fvisibility=hidden -Wall
SRC_FILES = *.cpp

all:
	$(GPP) -o $(OUTPUT_PATH)/$(EXEC) $(DEFINE) $(SRC_FILES) $(LIBS) $(LIB_DIRS) $(INCLUDE)
	
clean:
	rm -f *.o

