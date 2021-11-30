#include <stdio.h>

#include <cuda.h>

#define GLM_FORCE_CUDA

#include <Shape.h>
#include <ShapeFunctions.h>

#define TX 16
#define TY 16


__device__
unsigned char clip(float n) { return n > 255 ? 255 : (n < 0 ? 0 : (char)n); }

#define MAX_DEPTH 30.0
#define PRECISION 0.001
#define MAX_ITER 60

typedef float (*sdf_func) (shape_k, vec3);
const __device__ sdf_func SDF[MAXSHAPE] = {distanceSphere, distanceParallelepiped, distanceCylinder, distanceTorus, NULL};

typedef float (*merge_func) (operator_k, float, float, size_t, size_t);
const __device__ merge_func MDF[MAXOPERATOR] = {MDFintersection, MDFunion, MDFsmoothunion, MDFsmoothsub, MDFmixer};

typedef vec3 (*alt_func) (alterator_k, vec3);
const __device__ alt_func AIF[MAXALTERATOR] = {AIFtwist};

__device__
extern general_sdf_func *sdfShapes;

__device__
float shapeDistance(shape_k* shapes, size_t idx, vec3 pos){
  float dist = MAX_DEPTH;

  dist = SDF[shapes[idx].type](shapes[idx], pos);

  return dist;
}

__device__
float rootDistance(shape_k* shapes, size_t idx, vec3 pos){
  float dist = MAX_DEPTH;
  shape_k currShape = shapes[idx];

  if(currShape.distanceProg > 0){
    dist = sdfShapes[currShape.distanceProg - 1](shapes, idx, pos);
  }else{
    if(currShape.type == OPERATOR){
      size_t child = currShape.child;

      dist = rootDistance(shapes, child, pos);

      for(size_t i = 1; i < currShape.op.n; i++){
        child = shapes[child].sibling;
        dist = MDF[currShape.op.type](currShape.op, dist, rootDistance(shapes, child, pos), i, currShape.op.n);
      }
    }else if(currShape.type == ALTERATOR){
      dist = shapeDistance(shapes, currShape.child, AIF[currShape.alt.type](currShape.alt, pos));
    }else{
      dist = shapeDistance(shapes, idx, pos);
    }
  }
  return dist;
}

__device__
vec3 getNorm(shape_k* shapes, size_t idx, vec3 pos){
  vec3 ret;
  float epsilon = PRECISION, base;

  base = rootDistance(shapes, idx, pos);
  ret.x = (rootDistance(shapes, idx, make_float3(pos.x + epsilon, pos.y, pos.z)) - base);
  ret.y = (rootDistance(shapes, idx, make_float3(pos.x, pos.y + epsilon, pos.z)) - base);
  ret.z = (rootDistance(shapes, idx, make_float3(pos.x, pos.y, pos.z + epsilon)) - base);
  return normalize(ret);
}

__device__
vec4 getColor(shape_k* shapes, size_t idx, vec3 pos, vec3 cam, vec3 axis){
  vec3 vers1, vers2;
  vers1 = getNorm(shapes, idx, pos);
  vers2 = (pos - cam) / length(pos - cam);
  float prod = clamp(-0.7 * (vers1.x * vers2.x + vers1.y * vers2.y + vers1.z * vers2.z), 0.0, 1.0);
  return make_float4(255, 0, 0, 255);
}

__device__
void removeShape(shape_k* shapes, size_t idx, size_t nShapes, size_t* roots, size_t nRoots, size_t *visibleShapes, int* nVisibleShapes, vec3 pos, vec3 axis, float radius){
  float minD = 0, dist, module;

  vec3 A, curr, versor;
  module = length(axis);
  A = pos;
  versor = axis / module;
  curr = A;
  dist = 0.0;
  minD = MAX_DEPTH;

  do{
    minD = rootDistance(shapes, roots[idx], curr);

    dist += minD;
    curr = dist * versor + A;

  }while(dist < MAX_DEPTH && minD > max(1.5 * radius * dist / module, PRECISION));

  if(dist < MAX_DEPTH){
    size_t ret = atomicAdd(nVisibleShapes, 1);
    visibleShapes[ret] = roots[idx];
  }
}

__device__
float computeShadow(vec3 pos, vec3 ray, shape_k* shapes, size_t idx, size_t* roots, size_t nRoots, light_k* lights, size_t nLight){
  float ret = 0.2;
  float k_d, k_s, k_a, alpha;
  k_a = 0.2;
  k_s = 0.5;
  k_d = 0.5;
  alpha = 10;

  vec3 n = getNorm(shapes, idx, pos);
  for(int i = 0; i < nLight; i++){
    vec3 vecLux = normalize(lights[i].pos - pos);
    vec3 vecLux1 = 2 * dot(vecLux, n) * n - vecLux;
    ret += (k_d * clamp(dot(vecLux, n), 0.0, 1.0) + k_s * clamp(pow(dot(vecLux1, -ray), alpha), 0.0, 1.0)) * lights[i].intensity / pow(length(lights[i].pos - pos), 2);
  }

  return ret;
}


extern "C" __global__
void firstRay(
  uchar4 *d_out, int w, int h,
  shape_k* shapes, size_t nShapes,
  size_t* roots, int *nRoots,
  light_k* lights, size_t nLights,
  vec3 A, vec3 x, vec3 y, vec3 z
){

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const int idx = c +  r * w; // 1D indexing

  vec4 color = make_float4(0, 0, 0, 255);

  vec3 curr;
  float dist = 0.0, module;

  float xcoeff, ycoeff;
  xcoeff = 2 * ((float)c / w) - 1;
  ycoeff = 2 * ((float)r / h) - 1;

  const vec3 B = A + z + x * xcoeff + y * ycoeff;

  module = sqrtf((B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y) + (B.z - A.z) * (B.z - A.z));
  const vec3 versor = (B - A) / module;

  curr = B;
  dist = module;

  float minD, currD;

  __shared__ void* __addr[2];
  __shared__ int n;

  size_t *visibleShapes = roots;
  n = *nRoots;

  if(threadIdx.x + threadIdx.y == 0){
    visibleShapes = new size_t[*nRoots];
    n = 0;
    __addr[1] = (void*)visibleShapes;
  }
  __syncthreads();

  visibleShapes = (size_t*)__addr[1];

  const int cMid = (blockIdx.x + 0.5) * blockDim.x;
  const int rMid = (blockIdx.y + 0.5) * blockDim.y;
  float xcoeffMid, ycoeffMid;
  xcoeffMid = 2 * ((float)cMid / w) - 1;
  ycoeffMid = 2 * ((float)rMid / h) - 1;
  const vec3 Bmid = A + z + x * xcoeffMid + y * ycoeffMid;
  const vec3 axis = (Bmid - A) / length(Bmid - A);
  float dx, dy, radius;
  dx = blockDim.x * length(x) / w;
  dy = blockDim.y * length(y) / h;
  radius = sqrtf(dx * dx + dy * dy);
  for(size_t i = (threadIdx.x + threadIdx.y * blockDim.x); i < (*nRoots); i += (blockDim.x * blockDim.y)){
    removeShape(shapes, i, nShapes, roots, *nRoots, visibleShapes, &n, A, axis, radius);
  }
  __syncthreads();

  size_t shapeHit, counter = 0;
  minD = MAX_DEPTH;

  #pragma unroll 4
  for(counter = 0; counter < MAX_ITER; counter++){
    minD = MAX_DEPTH;

    #pragma unroll 4
    for(int j = 0; j < n; j++){
      currD = rootDistance(shapes, visibleShapes[j], curr);
      shapeHit = (minD < currD) ? shapeHit : visibleShapes[j];
      minD = min(minD, currD);
     }

     if(minD < PRECISION) break;

     dist += minD;
     curr = dist * versor + A;

     if(dist > MAX_DEPTH){ dist = MAX_DEPTH; break;}
   }

   curr = dist * versor + A;

   float lightLevel = 1.0;
   if(dist < MAX_DEPTH){
    color = getColor(shapes, shapeHit, curr, A, z);
    lightLevel = computeShadow(curr, versor, shapes, shapeHit, roots, *nRoots, lights, nLights);
  }

   float intensity = lightLevel;
   d_out[idx].x = clip(intensity * color.x);
   d_out[idx].y = clip(intensity * color.y);
   d_out[idx].z = clip(intensity * color.z);
   d_out[idx].w = 255;

   __syncthreads();

   if(threadIdx.x + threadIdx.y == 0){
    delete[] (size_t*)__addr[1];
  }
}

extern "C" __global__
void getVisibleShape(shape_k* shapes, size_t nShapes, size_t* roots, size_t nRoots, size_t *visibleShapes, int* nVisibleShapes, vec3 pos, vec3 axis, float radius){
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= nRoots) return;

  removeShape(shapes, idx, nShapes, roots, nRoots, visibleShapes, nVisibleShapes, pos, axis, radius);
}
