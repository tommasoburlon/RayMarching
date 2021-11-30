#ifndef SCENEHANDLER_H
#define SCENEHANDLER_H

#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <Shape.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Camera.h>
#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string>

#define CHECK(x) {\
  int ___k = 0; \
  CUresult ___res; \
  do{ \
    ___res = (x); \
    ___k++; \
    if(___res != CUDA_SUCCESS){ \
      printf("%d) error line: %d, err: %d\n", ___k - 1, __LINE__, ___res); \
      fflush(stdout); \
    } \
  }while(___res != CUDA_SUCCESS && ___k < 1); \
  if(___res != CUDA_SUCCESS){ \
     printf("RESULT: error line: %d, err: %d\n", __LINE__, ___res); \
      fflush(stdout); \
    } \
  } \

struct shape_data{
  size_t id;
};

class SceneHandler{
  std::vector<shape_data> data;

  std::unordered_map<size_t, size_t> idtoindex;
  std::unordered_map<size_t, size_t> idtoroot;
  size_t nextShapeID, nextLightID;

  //asynchrounous CPU copy of the variable shapes_d
  thrust::host_vector<shape_k> shapes_h;
  thrust::host_vector<size_t> roots_h;
  thrust::host_vector<light_k> lights_h;

  //data stored in the device (GPU)
  thrust::device_vector<shape_k> shapes_d;
  thrust::device_vector<size_t> roots_d;
  thrust::device_vector<light_k> lights_d;

  //
  CUlinkState cuState;
  CUmodule module;
  void *prog;
  size_t progSize;
  std::vector<char> kernelData;
  std::vector<std::vector<char>> programs;
public:
  SceneHandler(const char *path = nullptr);
  ~SceneHandler();

  size_t insertShape(shape_k sh, bool isRoot = true);
  bool removeShape(size_t id);
  void updateShape(size_t id, shape_k sh);

  size_t insertLight(light_k lt);
  bool removeLight(size_t id);
  void updateLight(size_t id, light_k lt);

  void flush();

  const thrust::device_vector<shape_k>& getDeviceShape(){ return shapes_d; };
  const thrust::device_vector<size_t>& getDeviceRoot(){ return roots_d; };
  const thrust::device_vector<light_k>& getDeviceLight(){ return lights_d; };

  void drawScene(uchar4 *d_out, int w, int h, Camera cam);
};


#endif
