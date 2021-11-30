//#include <CUDA/kernel.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
 #include <GLUT/glut.h>
 #else
 #include <GL/glew.h>
 #include <GL/freeglut.h>
 #endif
 #include <cuda_runtime.h>
 #include <cuda_gl_interop.h>
 #include <interactions.h>
#include <chrono>
#include <Camera.h>
#include <SceneHandler.h>

GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

SceneHandler scene;

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
  scene.drawScene(d_out, WIDTH, HEIGHT, cam);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);
   glEnable(GL_TEXTURE_2D);
   glBegin(GL_QUADS);
   glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
   glTexCoord2f(0.0f, 1.0f); glVertex2f(0, HEIGHT);
   glTexCoord2f(1.0f, 1.0f); glVertex2f(WIDTH, HEIGHT);
   glTexCoord2f(1.0f, 0.0f); glVertex2f(WIDTH, 0);
   glEnd();
   glDisable(GL_TEXTURE_2D);
 }

void display() {
   render();
   drawTexture();
   glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
   glutInit(argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize(WIDTH, HEIGHT);
   glutCreateWindow(TITLE_STRING);
 #ifndef __APPLE__
   glewInit();
 #endif
}

void initPixelBuffer() {
   glGenBuffers(1, &pbo);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glBufferData(GL_PIXEL_UNPACK_BUFFER,  4 * WIDTH * HEIGHT * sizeof(GLubyte), 0,
                GL_STREAM_DRAW);
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_2D, tex);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
   if (pbo) {
     cudaGraphicsUnregisterResource(cuda_pbo_resource);
     glDeleteBuffers(1, &pbo);
     glDeleteTextures(1, &tex);
  }

  printf("\n");
}

std::chrono::time_point<std::chrono::high_resolution_clock> preT;
double fpscap, timeElapsed;

void update(){
  auto newT = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = newT - preT;
  preT = newT;
  double dT = elapsed.count();
  fpscap += dT;
  timeElapsed += dT;

  //printf("%f\n", 1 / dT);
  vec3 pos = make_float3(0, 0, 0);
  if(isPressed['w']) pos.z += SPEED * dT;
  if(isPressed['s']) pos.z -= SPEED * dT;
  if(isPressed['a']) pos.x -= SPEED * dT;
  if(isPressed['d']) pos.x += SPEED * dT;

  pos = cam.getPosition() + pos.z * cam.getZ() + pos.x * cam.getX();
  cam.setPosition(pos);

  shape_k sh;

  sh.type = OPERATOR;
  sh.op.type = MIXER;
  sh.op.n = 3;
  sh.child = 0;
  sh.op.k =  (cos(timeElapsed) + 1.0) / 2;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.updateShape(3, sh);

  sh.type = ALTERATOR;
  sh.alt.type = TWIST;
  sh.alt.k = cos(timeElapsed);
  sh.child = 4;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.updateShape(5, sh);

  sh.type = OPERATOR;
  sh.op.type = MIXER;
  sh.op.n = 3;
  sh.op.k = (cos(2 * timeElapsed) + 1.0) / 2;
  sh.child = 7;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.updateShape(10, sh);

  light_k lgt;
  lgt.pos = cam.getPosition() + make_float3(0.0, -1.0, 0.0);
  lgt.intensity = 5.0;
  scene.updateLight(0, lgt);

  #define MAXFPS 500
  if(1 / fpscap < MAXFPS){
    fpscap -= 1.0 / MAXFPS;
    glutPostRedisplay();
  }
}

int main(int argc, char** argv) {
  CHECK( cuInit(0) );

  CUdevice dev;
  CUcontext ctx;
  CHECK( cuDeviceGet(&dev, 0) );

  char name[255];

  cuDeviceGetName ( name, 255, dev);
  printf("device found: %s\n", name);

  timeElapsed = 0;

  shape_k sh;

  CHECK( cuCtxCreate(&ctx, 0, dev) );

  scene = SceneHandler("resources/kernel/kernel.ptx");

  cam = Camera(make_float3(6.9, 5.68, 8.8));
  cam.setAngX(0);
  cam.setAngY(3.56);
  cam.setWidth((float)WIDTH / 1000);
  cam.setHeight((float)HEIGHT / 1000);
  cam.setFOV(1);

  sh.type = PARALLELEPIPED;
  sh.parallelepiped.vertex = make_float3(5.0, 5.0, 5.0);
  sh.parallelepiped.size = make_float3(2.0, 2.0, 2.0);
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  sh.sibling = 1;
  scene.insertShape(sh, false);

  sh.type = SPHERE;
  sh.sphere.center = make_float3(6.0, 6.0, 6.0);
  sh.sphere.radius = 1.5;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  sh.sibling = 2;
  scene.insertShape(sh, false);

  sh.type = CYLINDER;
  sh.cylinder.center = make_float3(6.0, 6.0, 6.0);
  sh.cylinder.radius = 1.0;
  sh.cylinder.height = 2.0;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, false);

  sh.type = OPERATOR;
  sh.op.type = MIXER;
  sh.op.n = 3;
  sh.op.k = 0;
  sh.child = 0;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, true);
  //CHECK( cuCtxCreate(&ctx, 0, dev) );

  sh.type = PARALLELEPIPED;
  sh.parallelepiped.vertex = make_float3(-1.0, 0.0, -1.0);
  sh.parallelepiped.size = make_float3(2.0, 5.0, 2.0);
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, false);

  sh.type = ALTERATOR;
  sh.alt.type = TWIST;
  sh.alt.k = 1.0;
  sh.child = 4;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, true);
  //CHECK( cuCtxCreate(&ctx, 0, dev) );

  sh.type = SPHERE;
  sh.sphere.center = make_float3(4.0, 2.5, 0.0);
  sh.sphere.radius = 0.5;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, true);

  sh.type = PARALLELEPIPED;
  sh.parallelepiped.vertex = make_float3(2.0, 5.0, 5.0);
  sh.parallelepiped.size = make_float3(2.0, 2.0, 2.0);
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  sh.sibling = 8;
  scene.insertShape(sh, false);

  sh.type = SPHERE;
  sh.sphere.center = make_float3(3.0, 6.0, 4.5);
  sh.sphere.radius = 1.5;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  sh.sibling = 9;
  scene.insertShape(sh, false);

  sh.type = CYLINDER;
  sh.cylinder.center = make_float3(3.0, 6.0, 6.0);
  sh.cylinder.radius = 1.0;
  sh.cylinder.height = 2.0;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, false);

  sh.type = OPERATOR;
  sh.op.type = MIXER;
  sh.op.n = 3;
  sh.op.k = 0;
  sh.child = 7;
  sh.color = make_float4(rand() % 255, rand() % 255, rand() % 255, 255);
  scene.insertShape(sh, true);

  light_k lgt;
  lgt.pos = make_float3(4.0, 2.5, 0.0);
  lgt.intensity = 500.0;
  scene.insertLight(lgt);

  initGLUT(&argc, argv);
  gluOrtho2D(0, WIDTH, HEIGHT, 0);
  glutKeyboardUpFunc(keyup);
  glutKeyboardFunc(keydown);
  glutSpecialFunc(handleSpecialKeypress);
  glutPassiveMotionFunc(mouseMove);
  glutMotionFunc(mouseDrag);
  glutDisplayFunc(display);
  glutIdleFunc(update);
  initPixelBuffer();

  preT = std::chrono::high_resolution_clock::now();

  glutSetCursor(GLUT_CURSOR_NONE);
  initIO();

  glutMainLoop();
  atexit(exitfunc);

  return 0;
}
