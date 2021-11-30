#include <Shape.h>

#ifndef __NVCC__
__device__ __host__ float min(float a, float b){return a < b ? a : b;};
__device__ __host__ float max(float a, float b){return a > b ? a : b;};
#endif

__device__ __host__ vec3 operator+(vec3 a, vec3 b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __host__ vec3 operator-(vec3 a){ return make_float3(-a.x, -a.y, -a.z); }
__device__ __host__ vec3 operator-(vec3 b, vec3 a){ return make_float3(b.x - a.x, b.y - a.y, b.z - a.z); }
__device__ __host__ vec3 operator*(float a, vec3 b){ return make_float3(a * b.x, a * b.y, a * b.z); }
__device__ __host__ vec3 operator*(vec3 b, float a){ return make_float3(a * b.x, a * b.y, a * b.z); }
__device__ __host__ vec3 operator/(vec3 v, float a){ return make_float3(v.x / a, v.y / a, v.z / a); }

__device__ __host__ vec3 abs(vec3 a){ return make_float3(abs(a.x), abs(a.y), abs(a.z));}

__device__ __host__ vec3 max(vec3 a, vec3 b){ return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));}
__device__ __host__ float length(vec3 a){ return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z); }
__device__ __host__ vec3 normalize(vec3 v){ return v / length(v); }
__device__ __host__ float dot(vec3 a, vec3 b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
