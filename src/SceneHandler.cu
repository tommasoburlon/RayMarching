#include <SceneHandler.h>
#include <var.h>

#define CHECKNVRTC(x) {nvrtcResult res = (x); if(res != NVRTC_SUCCESS){ printf("error NVRTC line: %d, err: %d\n", __LINE__, res); fflush(stdout); }}

struct tree_idx{
  size_t idx;
  shape_k sh;
  tree_idx *sibling, *child, *parent;
};

std::string getSDF(ShapeType type){
  std::string SDF[MAXSHAPE] = {"distanceSphere", "distanceParallelepiped", "distanceCylinder", "distanceTorus", "", ""};
  return SDF[type];
}

std::string getAIF(AlteratorType type){
  std::string AIF[MAXALTERATOR] = {"AIFtwist"};
  return AIF[type];
}

std::string getMDF(OperatorType type){
  std::string MDF[MAXOPERATOR] = {"MDFintersection", "MDFunion", "MDFsmoothunion", "MDFsmoothsub", "MDFmixer"};
  return MDF[type];
}

std::string getFunctionFromShape(tree_idx *tree){
  std::string str = "", replacePos;
  int i = 0;
  tree_idx *c1, *c2;

  if(tree->parent == NULL)
    str = "\tpos_0 = pos;\n";
  else
    str = "\tpos_" + std::to_string(tree->idx) + " = pos_" + std::to_string(tree->parent->idx) + ";\n";

  switch(tree->sh.type){
    case OPERATOR:
      c1 = tree->child;
      c2 = c1->sibling;
      str += getFunctionFromShape(tree->child);
      str += "\tdist_" + std::to_string(tree->idx) + " = dist_" + std::to_string(tree->child->idx) + ";\n";
      i = 1;
      for(tree_idx *child = c1->sibling; child != nullptr; child = child->sibling){
        str += getFunctionFromShape(child);
        str += "\tdist_" + std::to_string(tree->idx) + " = " + getMDF(tree->sh.op.type) + "(sh[indexes_" + std::to_string(tree->idx) + "].op, dist_" + std::to_string(tree->idx) + ", dist_" + std::to_string(child->idx) + ", " + std::to_string(i) + ", " + std::to_string(tree->sh.op.n) + ");\n";
        i++;
      }
    break;
    case ALTERATOR:
      str += "\tpos_" + std::to_string(tree->idx) + " = " + getAIF(tree->sh.alt.type) + "(sh[indexes_" + std::to_string(tree->idx) + "].alt, pos_" + std::to_string(tree->idx) + ");\n";
      str += getFunctionFromShape(tree->child);
      str += "\tdist_" + std::to_string(tree->idx) + " = dist_" + std::to_string(tree->child->idx) + ";\n";
    break;
    default:
      str += "\tdist_" + std::to_string(tree->idx) + " = "+ getSDF(tree->sh.type) + "(sh[indexes_" + std::to_string(tree->idx) + "], pos_" + std::to_string(tree->idx) + ");\n";
    break;
  }
  return str;
}

size_t getNumberShape(shape_k *sh, size_t idx){
  size_t ret = 1, child;

  switch(sh[idx].type){
    case OPERATOR:
      child = sh[idx].child;
      for(int i = 0; i < sh[idx].op.n; i++){
        ret += getNumberShape(sh, child);
        child = sh[child].sibling;
      }
    break;
    case ALTERATOR:
      ret += getNumberShape(sh, sh[idx].child);
    break;
    default:
      ret = 1;
    break;
  }

  return ret;
}

std::string fillIndexes(shape_k* sh, size_t idx, tree_idx* tree, size_t previous, size_t &counter, bool isParent){
  std::string str, conn = isParent == true ? "child" : "sibling";
  size_t child, thisN = counter;

  if(counter == 0){
    str = "\tconst size_t indexes_0 = idx;\n";
    str += "\tfloat dist_0;\n";
    str += "\tvec3 pos_0;\n";
  }else{
    str = "\tconst size_t indexes_" +  std::to_string(thisN) + " = sh[indexes_" + std::to_string(previous) + "]." + conn + ";\n";
    str += "\tfloat dist_" +  std::to_string(thisN) + ";\n";
    str += "\tvec3 pos_" +  std::to_string(thisN) + ";\n";
  }

  tree[0].idx  = thisN;
  tree[0].sh   = sh[idx];
  tree[0].sibling = nullptr;
  tree[0].child   = nullptr;
  counter += 1;

  switch(sh[idx].type){
    case OPERATOR:
      size_t curr, prev;
      child = sh[idx].child;
      tree[0].child = &tree[1];
      tree[1].parent = &tree[0];
      str += fillIndexes(sh, child, &tree[1], thisN, counter, true);
      prev = 1;
      for(int i = 1; i < sh[idx].op.n; i++){
        curr = counter;
        child = sh[child].sibling;
        str += fillIndexes(sh, child, &tree[counter - thisN], counter - 1, counter, false);
        tree[prev].sibling = &tree[curr];
        tree[curr].parent = &tree[0];
        prev = curr;
        child = sh[child].sibling;
      }
    break;
    case ALTERATOR:
      child = sh[idx].child;
      tree[0].child = &tree[1];
      tree[1].parent = &tree[0];
      str += fillIndexes(sh, child, &tree[1], thisN, counter, true);
    break;
    default:
    break;
  }

  return str;

}

std::string getProgramFromShape(shape_k* sh, size_t idx, size_t id, size_t func_id){
  std::string str = "";
  size_t nShapes = getNumberShape(sh, idx), counter = 0;
  tree_idx* tree = new tree_idx[nShapes];
  memset(tree, 0, nShapes * sizeof(tree_idx));

  str = str + "#include <Shape.h>\n";
  str = str + "#include <ShapeFunctions.h>\n";
  str = str +  "__device__ float distance" + std::to_string(id) + "(shape_k *sh, size_t idx, vec3 pos){\n";
  str = str + fillIndexes(sh, idx, &tree[0], 0, counter, true);
  str = str + getFunctionFromShape(&tree[0]);
  str = str + "\treturn dist_0;\n";
  str = str + "}\n";
  str = str + "__device__ extern volatile general_sdf_func SDFfunc" + std::to_string(func_id) + " = distance" + std::to_string(id) + ";\n";

  delete[] tree;
  return str;
}

SceneHandler::SceneHandler(const char* path){
  nextShapeID = 0;
  nextLightID = 0;

  lights_d = thrust::device_vector<light_k>(0);
  lights_h = thrust::host_vector<light_k>(0);
  roots_d = thrust::device_vector<size_t>(0);
  roots_h = thrust::host_vector<size_t>(0);
  shapes_d = thrust::device_vector<shape_k>(0);
  shapes_h = thrust::host_vector<shape_k>(0);


  programs.push_back(std::vector<char>(kernel_var, &kernel_var[strlen(kernel_var + 1)]));
  programs.push_back(std::vector<char>(vecUtility_var, &vecUtility_var[strlen(vecUtility_var + 1)]));

  if(path){

    CHECK( cuLinkCreate (0, nullptr, nullptr, &cuState) );

    CHECK( cuLinkAddData(cuState, CU_JIT_INPUT_PTX, programs[0].data(), programs[0].size(), nullptr, 0, nullptr, nullptr) );

    CHECK( cuLinkAddData(cuState, CU_JIT_INPUT_PTX, programs[1].data(), programs[1].size(), nullptr, 0, nullptr, nullptr) );

    CHECK( cuLinkComplete(cuState, &prog, &progSize) );

    CHECK( cuModuleLoadData(&module, prog) );

    CUdeviceptr ptr;
    size_t bytes;
    CHECK( cuModuleGetGlobal(&ptr, &bytes, module, "sdfShapes") );

    CHECK( cuLinkDestroy(cuState) );
  }
}

SceneHandler::~SceneHandler(){
}

size_t SceneHandler::insertShape(shape_k sh, bool isRoot){
  idtoindex[nextShapeID] = shapes_h.size();
  sh.distanceProg = 0;
  shapes_h.push_back(sh);

  if(isRoot){
    idtoroot[nextShapeID] = roots_h.size();
    roots_h.push_back(shapes_h.size() - 1);

    std::string programSource = getProgramFromShape(shapes_h.data(), shapes_h.size() - 1, nextShapeID, programs.size() - 1);

    //std::cout << programSource << std::endl;

    const char* headers[] = {Shape_var, ShapeFunctions_var};
    const char* headerNames[] = {"Shape.h", "ShapeFunctions.h"};
    nvrtcProgram nvrtcProg;

    nvrtcCreateProgram(&nvrtcProg, programSource.c_str(),  "prog.cu", 2, headers, headerNames);

    const char* opts[] = {"-Xptxas", "--use_fast_math", "--extra-device-vectorization"};

    nvrtcCompileProgram(nvrtcProg, 3, opts);

    size_t shapeProgSize;
    nvrtcGetPTXSize(nvrtcProg, &shapeProgSize);

    std::vector<char> dataProgram(shapeProgSize);
    nvrtcGetPTX(nvrtcProg, dataProgram.data());

    nvrtcDestroyProgram(&nvrtcProg);

    programs.push_back(dataProgram);

    shapes_h[shapes_h.size() - 1].distanceProg = (programs.size() - 2);


    CHECK( cuLinkCreate (0, nullptr, nullptr, &cuState) );

    for(int i = 0; i < programs.size(); i++){
      CHECK( cuLinkAddData(cuState, CU_JIT_INPUT_PTX, programs[i].data(), programs[i].size(), nullptr, 0, NULL, NULL) );
    }

    CHECK( cuLinkComplete(cuState, &prog, &progSize) );

    CHECK( cuModuleLoadData(&module, prog) );

    CHECK( cuLinkDestroy(cuState) );

    CUdeviceptr ptr;
    size_t bytes;
    CHECK( cuModuleGetGlobal(&ptr, &bytes, module, "sdfShapes") );

    size_t* newMemory;
    cudaMalloc((void**)&newMemory, (programs.size() - 1) * bytes);

    cudaMemcpy((void*)ptr, &newMemory, bytes, cudaMemcpyHostToDevice);

    for(size_t i = 0; i < programs.size() - 2; i++){
      std::string funcName = "SDFfunc" + std::to_string(i + 1);

      CHECK( cuModuleGetGlobal(&ptr, &bytes, module, funcName.c_str()) );
      cudaMemcpy((void*)(&newMemory[i]), (void*)ptr, bytes, cudaMemcpyHostToDevice);
    }

  }

  return nextShapeID++;
}

bool SceneHandler::removeShape(size_t id){
  return true;
}

void SceneHandler::updateShape(size_t id, shape_k sh){
  size_t idx = idtoindex.at(id), progId;
  progId = shapes_h[idx].distanceProg;
  shapes_h[idx] = sh;
  shapes_h[idx].distanceProg = progId;
}

size_t SceneHandler::insertLight(light_k lt){
  //idtoindex[nextID] = shapes_h.size();
  lights_h.push_back(lt);

  return nextLightID++;
}

bool SceneHandler::removeLight(size_t id){
  return true;
}

void SceneHandler::updateLight(size_t id, light_k lt){
  //size_t idx = idtoindex.at(id);
  lights_h[id] = lt;
}


void SceneHandler::flush(){
  shapes_d = shapes_h;
  lights_d = lights_h;

  roots_d = thrust::device_vector<size_t>(2 * roots_h.size());
  thrust::copy(roots_h.begin(), roots_h.end(), roots_d.begin());
}

void drawSceneLauncher(CUfunction firstRay, CUfunction getVisibleShape, uchar4 *d_out, int w, int h, SceneHandler scene, Camera cam) {

  #define TX 16
  #define TY 16

  const dim3 blockSize(TX, TY, 1);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY, 1);

  scene.flush();

  vec3 pos, x, y, z;

  pos = cam.getPosition();
  x = cam.getX();
  y = cam.getY();
  z = cam.getZ();

  thrust::device_vector<shape_k> vecShapes;
  thrust::device_vector<size_t> vecRoots;
  thrust::device_vector<light_k> vecLights;
  shape_k* shapes;
  light_k* lights;
  size_t* roots;
  size_t nShapes, nRoots, nLights;
  int *nVisibleShapes;

  vecShapes = scene.getDeviceShape();
  vecRoots  = scene.getDeviceRoot();
  vecLights = scene.getDeviceLight();

  shapes   = thrust::raw_pointer_cast(&vecShapes[0]);
  roots    = thrust::raw_pointer_cast(&vecRoots[0]);
  lights   = thrust::raw_pointer_cast(&vecLights[0]);

  nShapes  = vecShapes.size();
  nRoots   = vecRoots.size() / 2;
  nLights  = vecLights.size();

  size_t tempN = nRoots;
  cudaMalloc(&nVisibleShapes, sizeof(int));
  cudaMemcpy(nVisibleShapes, &tempN, sizeof(int), cudaMemcpyHostToDevice);

  float radius = length(x + y);


  void* paramsFirstRay[] = {
    &d_out,
    &w,
    &h,
    &shapes,
    &nShapes,
    &roots,
    &nVisibleShapes,
    &lights,
    &nLights,
    &pos,
    &x,
    &y,
    &z
  };

  CHECK(
    cuLaunchKernel(firstRay,
      gridSize.x,  gridSize.y,  gridSize.z,
      blockSize.x, blockSize.y, blockSize.z,
      0,              NULL,
      paramsFirstRay, NULL
    )
  );
}

void SceneHandler::drawScene(uchar4 *d_out, int w, int h, Camera cam){
  CUfunction firstRay, getVisibleShape;

  CHECK( cuModuleGetFunction(&firstRay, module, "firstRay") );
  CHECK( cuModuleGetFunction(&getVisibleShape, module, "getVisibleShape") );

  drawSceneLauncher(firstRay, getVisibleShape, d_out, w, h, *this, cam);
}
