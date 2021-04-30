################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/matrix_mul.cu 

CPP_SRCS += \
../src/main.cpp 

OBJS += \
./src/main.o \
./src/matrix_mul.o 

CU_DEPS += \
./src/matrix_mul.d 

CPP_DEPS += \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -I"/usr/samples/0_Simple" -I"/usr/samples/common/inc" -I"/home/argalad/Documentos/GPU/Practica1/Ejemplo5/nsight/matrix_mul_shared" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -I"/usr/samples/0_Simple" -I"/usr/samples/common/inc" -I"/home/argalad/Documentos/GPU/Practica1/Ejemplo5/nsight/matrix_mul_shared" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -I"/usr/samples/0_Simple" -I"/usr/samples/common/inc" -I"/home/argalad/Documentos/GPU/Practica1/Ejemplo5/nsight/matrix_mul_shared" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -I"/usr/samples/0_Simple" -I"/usr/samples/common/inc" -I"/home/argalad/Documentos/GPU/Practica1/Ejemplo5/nsight/matrix_mul_shared" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


