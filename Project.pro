TEMPLATE=app

OBJECTS_DIR=obj
CUDA_OBJECTS_DIR=cudaobj

TARGET=project

INCLUDEPATH += \
  $${PWD}/include \
  $${PWD}/thirdparty \
  $${PWD}/thirdparty/stb \
  $${PWD}/thirdparty/cub \
  $${PWD}/thirdparty/gsl-lite/include \
  $${PWD}/thirdparty/value-ptr-lite/include \
  ${CUDA_PATH}/include \
  ${HOME}/cusplibrary \
  ${CUDA_PATH}/include/cuda 

HEADERS += $$files(include/*.(h | hpp | cuh), true) 

# Standard flags
QMAKE_CXXFLAGS += -std=c++14 -g
# Optimisation flags
QMAKE_CXXFLAGS += -Ofast -march=native -frename-registers -funroll-loops -ffast-math -fassociative-math
# Intrinsics flags
QMAKE_CXXFLAGS += -mfma -mavx2 -m64 -msse -msse2 -msse3
# Enable all warnings
QMAKE_CXXFLAGS += -Wall -Wextra -pedantic-errors
# Vectorization info
QMAKE_CXXFLAGS += -ftree-vectorize -ftree-vectorizer-verbose=5 

# Cuda sources must be compiled seperately with nvcc
CUDA_SOURCES += $$files(src/*.cu, true) 
SOURCES += $$files(src/*.cpp, true) 

LIBS += -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_PATH}/lib64/nvidia
LIBS += -lcudart -lcurand -licudata -lcudart_static -lcudadevrt -lcusolver -lcusparse

CUDA_INC += $$join(INCLUDEPATH, ' -I', '-I', ' ')

NVCCFLAGS += -ccbin ${HOST_COMPILER} -pg -g -lineinfo --std=c++14 -O3 --expt-extended-lambda --expt-relaxed-constexpr
NVCCFLAGS += -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
NVCCFLAGS += -Xcompiler -fno-strict-aliasing -Xcompiler -fPIC 
NVCCFLAGS += -Xptxas -O3 --use_fast_math --restrict
NVCCFLAGS += $$join(DEFINES, ' -D', '-D', ' ')
#NVCCFLAGS += -v
#NVCCFLAGS += -G

NVCCBIN = ${CUDA_PATH}/bin/nvcc

CUDA_COMPILE_BASE = $${NVCCBIN} $${NVCCFLAGS} $${CUDA_INC} ${QMAKE_FILE_NAME}
CUDA_COMPILE = $${CUDA_COMPILE_BASE} -o ${QMAKE_FILE_OUT} $${LIBS}

# Compile cuda (device) code into object files
cuda.input = CUDA_SOURCES
cuda.output = $${CUDA_OBJECTS_DIR}/${QMAKE_FILE_BASE}.o
cuda.commands += $${CUDA_COMPILE} -dc  
cuda.CONFIG = no_link 
cuda.variable_out = CUDA_OBJ 
cuda.variable_out += OBJECTS
cuda.clean = $${CUDA_OBJECTS_DIR}/*.o
QMAKE_EXTRA_COMPILERS += cuda

# Link cuda object files into one object file with symbols that GCC can recognise
cudalink.input = CUDA_OBJ
cudalink.output = $${OBJECTS_DIR}/cuda_link.o
cudalink.commands = $${CUDA_COMPILE} -dlink
cudalink.CONFIG = combine
cudalink.dependency_type = TYPE_C
cudalink.depend_command = $${CUDA_COMPILE_BASE} -M
QMAKE_EXTRA_COMPILERS += cudalink

