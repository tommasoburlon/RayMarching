#ifndef SHAPEFUNCTIONS_H
#define SHAPEFUNCTIONS_H

#include <Shape.h>

typedef float (*general_sdf_func) (shape_k*, size_t, vec3);

__device__
float clamp(float _x, float _min, float _max){
  return max(min(_x, _max), _min);
}

__device__
float mix(float a, float b, float t){
  return (b - a) * t + a;
}

__device__
float distanceParallelepiped(shape_k sh, vec3 pos){
  parallelepiped_k s = sh.parallelepiped;
  vec3 q = abs(pos - s.vertex - 0.5f * s.size) - 0.5f * s.size;
  return length(max(q, make_float3(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

__device__
float distanceSphere(shape_k sh, vec3 pos){
  sphere_k s = sh.sphere;
  return sqrtf((pos.x - s.center.x) * (pos.x - s.center.x) +
    (pos.y - s.center.y) * (pos.y - s.center.y) +
    (pos.z - s.center.z) * (pos.z - s.center.z)) - s.radius;
}

__device__
float distanceTorus(shape_k sh, vec3 pos){
  torus_k t = sh.torus;
  vec3 nearestPoint;
  float module, dx, dz;
  dx = pos.x - t.center.x;
  dz = pos.z - t.center.z;
  module = t.centerRadius / sqrtf(dz * dz + dx * dx);
  nearestPoint.y = t.center.y;
  nearestPoint.x = dx * module + t.center.x;
  nearestPoint.z = dz * module + t.center.z;

  return sqrtf((pos.x - nearestPoint.x) * (pos.x - nearestPoint.x) +
    (pos.y - nearestPoint.y) * (pos.y - nearestPoint.y) +
    (pos.z - nearestPoint.z) * (pos.z - nearestPoint.z)) - t.tubeRadius;
}

__device__
float distanceCylinder(shape_k sh, vec3 pos){
  cylinder_k cylinder = sh.cylinder;
  float distAxis, distMid, dist;
  distAxis = sqrtf((pos.x - cylinder.center.x) * (pos.x - cylinder.center.x) + (pos.z - cylinder.center.z) * (pos.z - cylinder.center.z)) - cylinder.radius;
  distMid = abs(pos.y - cylinder.center.y) - 0.5 * cylinder.height;

  if(distMid > 0){
    if(distAxis < 0){
      dist = distMid;
    }else{
      dist = sqrtf(distMid * distMid + distAxis * distAxis);
    }
  }else{
    dist = distAxis;
  }

  if(dist < 0){
    dist = max(distAxis, distMid);
  }

  return dist;
}

__device__
float MDFunion(operator_k op, float d1, float d2, size_t idx, size_t max){ return min(d1, d2); }

__device__
float MDFintersection(operator_k op, float d1, float d2, size_t idx, size_t _max){ return max(d1, d2); }

__device__
float MDFsmoothunion(operator_k op, float d1, float d2, size_t idx, size_t max){
  float h = clamp( 0.5 + 0.5 * (d2 - d1) / op.k, 0.0, 1.0 );
  return mix(d2, d1, h) - op.k * h * (1.0 - h);
}

__device__
float MDFsmoothsub(operator_k op, float d1, float d2, size_t idx, size_t max) {
  float h = clamp( 0.5 - 0.5*(d2 + d1) / op.k, 0.0, 1.0 );
  return mix( d2, -d1, h ) + op.k * h * (1.0-h);
}

__device__
float MDFmixer(operator_k op, float d1, float d2, size_t idx, size_t max){
  float endPart, startPart, ret;

  endPart = (((float)idx) / (max - 1));
  startPart = (((float)idx - 1) / (max - 1));

  if(op.k > endPart){
    ret = d2;
  } if(op.k < startPart){
    ret = d1;
  }else{
    ret = (d2 - d1) * (op.k - startPart) / (endPart - startPart) + d1;
  }
  return ret;
}

__device__
vec3 AIFtwist(alterator_k alt, vec3 p){
  const float k = alt.k;
  float c = cos(k * p.y);
  float s = sin(k * p.y);
  vec3  q = make_float3(c * p.x - s * p.z, p.y,  s * p.x + c * p.z);
  return q;
}



//////// ************************ PROVA ***************************

/*__device__
float absf(float a){ return a > 0 ? a : -a; }

__device__
float clamp(float _x, float _min, float _max){
  float res = _x > _max ? _max : _x;
  return res > _min ? res : _min;
}

__device__
float mix(float a, float b, float t){
  return (b - a) * t + a;
}

__device__
float distanceParallelepiped(shape_k sh, vec3 pos){
  parallelepiped_k s = sh.parallelepiped;
  vec3 q = make_float3(
    absf(pos.x - s.vertex.x - 0.5f * s.size.x),
    absf(pos.y - s.vertex.y - 0.5f * s.size.y),
    absf(pos.z - s.vertex.z - 0.5f * s.size.z)
  );
  q.x = (q.x > 0.5f * s.size.x) ? q.x - 0.5f * s.size.x : 0.0f;
  q.y = (q.y > 0.5f * s.size.y) ? q.y - 0.5f * s.size.y : 0.0f;
  q.z = (q.z > 0.5f * s.size.z) ? q.z - 0.5f * s.size.z : 0.0f;
  return sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
}

__device__
float distanceSphere(shape_k sh, vec3 pos){
  sphere_k s = sh.sphere;
  return sqrtf(
    (pos.x - s.center.x) * (pos.x - s.center.x) +
    (pos.y - s.center.y) * (pos.y - s.center.y) +
    (pos.z - s.center.z) * (pos.z - s.center.z)
  ) - s.radius;
}

__device__
float distanceTorus(shape_k sh, vec3 pos){
  torus_k t = sh.torus;
  vec3 nearestPoint;
  float module, dx, dz;
  dx = pos.x - t.center.x;
  dz = pos.z - t.center.z;
  module = t.centerRadius / sqrtf(dz * dz + dx * dx);
  nearestPoint.y = t.center.y;
  nearestPoint.x = dx * module + t.center.x;
  nearestPoint.z = dz * module + t.center.z;

  return sqrtf((pos.x - nearestPoint.x) * (pos.x - nearestPoint.x) +
    (pos.y - nearestPoint.y) * (pos.y - nearestPoint.y) +
    (pos.z - nearestPoint.z) * (pos.z - nearestPoint.z)) - t.tubeRadius;
}

__device__
float distanceCylinder(shape_k sh, vec3 pos){
  cylinder_k cylinder = sh.cylinder;
  float distAxis, distMid, dist;
  distAxis = sqrtf((pos.x - cylinder.center.x) * (pos.x - cylinder.center.x) + (pos.z - cylinder.center.z) * (pos.z - cylinder.center.z)) - cylinder.radius;
  distMid = abs(pos.y - cylinder.center.y) - 0.5 * cylinder.height;

  if(distMid > 0){
    if(distAxis < 0){
      dist = distMid;
    }else{
      dist = sqrtf(distMid * distMid + distAxis * distAxis);
    }
  }else{
    dist = distAxis;
  }

  if(dist < 0){
    dist = max(distAxis, distMid);
  }

  return dist;
}

__device__
float MDFunion(operator_k op, float d1, float d2, size_t idx, size_t max){ return min(d1, d2); }

__device__
float MDFintersection(operator_k op, float d1, float d2, size_t idx, size_t _max){ return max(d1, d2); }

__device__
float MDFsmoothunion(operator_k op, float d1, float d2, size_t idx, size_t max){
  float h = clamp( 0.5 + 0.5 * (d2 - d1) / op.k, 0.0, 1.0 );
  return mix(d2, d1, h) - op.k * h * (1.0 - h);
}

__device__
float MDFsmoothsub(operator_k op, float d1, float d2, size_t idx, size_t max) {
  float h = clamp( 0.5 - 0.5*(d2 + d1) / op.k, 0.0, 1.0 );
  return mix( d2, -d1, h ) + op.k * h * (1.0-h);
}

__device__
float MDFmixer(operator_k op, float d1, float d2, size_t idx, size_t max){
  float endPart, startPart, ret;

  endPart = (((float)idx) / (max - 1));
  startPart = (((float)idx - 1) / (max - 1));

  if(op.k > endPart){
    ret = d2;
  } if(op.k < startPart){
    ret = d1;
  }else{
    ret = (d2 - d1) * (op.k - startPart) / (endPart - startPart) + d1;
  }
  return ret;
}

__device__
vec3 AIFtwist(alterator_k alt, vec3 p){
  const float k = alt.k;
  float c = cos(k * p.y);
  float s = sin(k * p.y);
  vec3  q = make_float3(c * p.x - s * p.z, p.y,  s * p.x + c * p.z);
  return q;
}*/

#endif
