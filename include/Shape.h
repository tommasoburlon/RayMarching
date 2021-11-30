#ifndef SHAPE_H
#define SHAPE_H


struct uchar4;
struct int2;

typedef float3 vec3;
typedef float4 vec4;

#ifndef __NVCC__
__device__ __host__ float min(float a, float b);
__device__ __host__ float max(float a, float b);
#endif

__device__ __host__ vec3 operator+(vec3 a, vec3 b);
__device__ __host__ vec3 operator-(vec3 a);
__device__ __host__ vec3 operator-(vec3 a, vec3 b);
__device__ __host__ vec3 operator*(float a, vec3 b);
__device__ __host__ vec3 operator*(vec3 b, float a);
__device__ __host__ vec3 operator/(vec3 v, float a);

__device__ __host__ vec3 abs(vec3 a);

__device__ __host__ vec3 max(vec3 a, vec3 b);
__device__ __host__ float length(vec3 a);
__device__ __host__ vec3 normalize(vec3 v);
__device__ __host__ float dot(vec3 a, vec3 b);

enum ShapeType{
  SPHERE,
  PARALLELEPIPED,
  CYLINDER,
  TORUS,
  OPERATOR,
  ALTERATOR,
  MAXSHAPE
};

enum OperatorType{
    INTERSECTION,
    UNION,
    SMOOTHUNION,
    SMOOTHSUB,
    MIXER,
    MAXOPERATOR
};

enum AlteratorType{
  TWIST,
  MAXALTERATOR
};

struct sphere_k{
  vec3 center;
  float radius;
};

struct torus_k{
  vec3 center;
  float centerRadius, tubeRadius;
};

struct parallelepiped_k{
  vec3 vertex;
  vec3 size;
};

struct cylinder_k{
  vec3 center;
  float radius, height;
};

struct operator_k{
  int n;
  OperatorType type;
  float k, k1, k2, k3;
};

struct alterator_k{
  AlteratorType type;
  float k, k1, k2, k3;
};


typedef struct{
  ShapeType type;
  union{
    sphere_k sphere;
    parallelepiped_k parallelepiped;
    cylinder_k cylinder;
    operator_k op;
    alterator_k alt;
    torus_k torus;
  };
  vec4 color;
  size_t child, sibling;
  size_t distanceProg;
} shape_k;

typedef struct{
  vec3 pos;
  float intensity;
} light_k;

typedef struct{
  float k_diffusion, k_reflection, k_refraction;
  vec4 color;
} surface_k;


#endif
