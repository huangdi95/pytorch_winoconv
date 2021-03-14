CXX := nvcc
TARGET := conv_1_32_3_32_3_5_5_32_28_28
CUDNN_PATH := $(LD_LIBRARY_PATH)
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_70 -std=c++11 -O2

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn

.phony: clean

clean:
	rm $(TARGET) || echo -n ""
