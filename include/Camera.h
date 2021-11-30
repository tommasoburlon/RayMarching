#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <cuda_runtime.h>
#include <Shape.h>

class Camera{
  vec3 position;
  float angX = 0, angY = 0, angZ = 0;
  float FOV, width, height;
public:

  vec3 getX();
  vec3 getY();
  vec3 getZ();

  vec3 getPosition(){ return position; };
  void setPosition(vec3 _position){ position = _position; }

  float getWidth(){ return width; }
  float getHeight(){ return height; }
  void setWidth(float _width){ width = _width; }
  void setHeight(float _height){ height = _height; }


  void setFOV(float fov){ FOV = fov; }

  float getAngX(){ return angX; }
  float getAngY(){ return angY; }
  float getAngZ(){ return angZ; }

  void setAngX(float _angX){ angX = _angX; }
  void setAngY(float _angY){ angY = _angY; }
  void setAngZ(float _angZ){ angZ = _angZ; }

  Camera(vec3 _position = make_float3(0, 0, 0)) : position(_position){};
  ~Camera(){};
};

#endif
