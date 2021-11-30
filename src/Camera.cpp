#include <Camera.h>


vec3 Camera::getX(){
  glm::vec3 x = glm::vec3(width, 0, 0);
  x = glm::rotateX(x, angX);
  x = glm::rotateY(x, angY);
  return make_float3(x.x, x.y, x.z);
}

vec3 Camera::getY(){
  glm::vec3 y = glm::vec3(0, height, 0);
  y = glm::rotateX(y, angX);
  y = glm::rotateY(y, angY);
  return make_float3(y.x, y.y, y.z);
}

vec3 Camera::getZ(){
  glm::vec3 z = glm::vec3(0, 0, FOV * width);
  z = glm::rotateX(z, angX);
  z = glm::rotateY(z, angY);
  return make_float3(z.x, z.y, z.z);
}
