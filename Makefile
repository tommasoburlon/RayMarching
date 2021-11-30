
CXX := g++
CXXFLAGS := -lGLEW -lglfw -lglut -lGL -lGLU -lm -Wall -Ofast
BINDIR := bin
SRCDIR := src
OBJDIR := build
INCDIR := include
RESDIR := resources
APPNAME := app
NVCC := nvcc
NVCCFLAGS := -Xcompiler -fPIC -Xptxas -O3 --use_fast_math -ccbin clang-8
CLIBS := -L/usr/local/cuda/lib64 -lcudart -lcuda -lnvrtc
IGNORE := kernel0.cu kernel.cu
IGNOPT := $(addprefix -not -name , $(IGNORE))

CXXFILENAME = $(basename $(shell find $(SRCDIR) -name '*.cpp' $(IGNOPT)))
CUDAFILENAME = $(basename $(shell find $(SRCDIR) -name '*.cu' $(IGNOPT)))
CXXSRC = $(addsuffix .cpp, $(CXXFILENAME))
CUDASRC = $(addsuffix .cu, $(CUDAFILENAME))

OBJSRC  = $(addsuffix .o, $(addprefix $(OBJDIR)/, $(CXXFILENAME)))
OBJSRC += $(addsuffix .o, $(addprefix $(OBJDIR)/, $(CUDAFILENAME)))

define cppfy
	cp -f $(1) temp
	sed -i 's/\"/\\"/g' temp
	sed -i 's/^/\t\"/g' temp
	sed -i 's/$$/\\n\"/g' temp
	sed -i '1 i\const char* $(notdir $(basename $(1)))_var = ""' temp
	sed -i '$$a;\n' temp
	cat temp >> $(2)
	rm -f temp

endef

all : $(CXXFILENAME) $(CUDAFILENAME)
	mkdir -p $(BINDIR)
	$(CXX) -I $(INCDIR) $(OBJSRC) $(CXXFLAGS) $(CLIBS) -o $(BINDIR)/$(APPNAME)
	cp -r $(RESDIR) $(BINDIR)

gen_var:
	rm -f $(INCDIR)/var.h
	$(call cppfy, $(INCDIR)/Shape.h, $(INCDIR)/var.h)
	$(call cppfy, $(INCDIR)/ShapeFunctions.h, $(INCDIR)/var.h)

	nvcc -I $(INCDIR) -ccbin clang-8 -dlink -ptx $(SRCDIR)/CUDA/vecUtility.cu -o vecUtility
	$(call cppfy, vecUtility, $(INCDIR)/var.h)

	nvcc -I $(INCDIR) -ccbin clang-8 -ptx $(SRCDIR)/CUDA/kernel.cu -o kernel
	$(call cppfy, kernel, $(INCDIR)/var.h)

	rm kernel vecUtility

$(CXXFILENAME): gen_var
	mkdir -p $(OBJDIR)/$(dir $@)
	$(CXX) $(CXXFLAGS) -I $(INCDIR) -c $@.cpp -o $(OBJDIR)/$@.o

$(CUDAFILENAME): gen_var
	mkdir -p $(OBJDIR)/$(dir $@)
	$(NVCC) -I $(INCDIR) $(NVCCFLAGS) -c $@.cu -o $(OBJDIR)/$@.o

clean:
	rm -r $(BINDIR) $(OBJDIR)
	rm -r $(RESDIR)/kernel

run:
	./$(BINDIR)/$(APPNAME)

.PHONY: all clean run
