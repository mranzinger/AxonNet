CC=g++-4.7
FLAGS=-std=c++11 -g -msse4.2
DFLAGS=$(FLAGS)
RFLAGS=-O3 $(FLAGS)

THIRD_PARTY   ?= ../ThirdParty
EIGEN_PATH    ?= $(THIRD_PARTY)/eigen-3-2-1
BOOST_PATH    ?= $(THIRD_PARTY)/boost_1_55_0
GTEST_PATH    ?= $(THIRD_PARTY)/gtest
AXON_PATH     ?= ../axon

INC_ROOT = include
SRC_ROOT = src

SRC = $(wildcard $(SRC_ROOT)/*.cpp)

OBJ_ROOT = obj

OBJS = $(patsubst $(SRC_ROOT)/%.cpp,$(OBJ_ROOT)/%.o,$(SRC))
OBJS_D = $(patsubst %.o, %.od,$(OBJS))

NET_SRC=$(wildcard $(SRC_ROOT)/*.cpp)
TRAINER_SRC=$(wildcard Trainer/*.cpp)
UNIT_SRC=$(wildcard unit_tests/*.cpp)

TRAINER_EXE=trainer
TRAINER_EXE_D=d_trainer

UNIT_EXE=unit_test
UNIT_EXE_D=d_unit_test

INCLUDES= -I$(AXON_PATH)/include \
	      -I$(EIGEN_PATH) \
	      -I$(BOOST_PATH)/include \
	      -I$(GTEST_PATH)/include \
          -I$(INC_ROOT)

LIBS_BASE=-L$(BOOST_PATH)/lib \
		  -L$(AXON_PATH)/lib \
		  -L$(GTEST_PATH)/lib \
		  -lboost_system -lboost_filesystem -lboost_program_options
		  
LIBS=$(LIBS_BASE) -laxcomm -laxser -laxutil
LIBS_D=$(LIBS_BASE) -laxcommd -laxserd -laxutild

.PHONY: all clean setup

all: debug release 

debug: setup $(OBJS_D) $(TRAINER_EXE_D) $(UNIT_EXE_D)

release: setup $(OBJS) $(TRAINER_EXE) $(UNIT_EXE)

$(OBJ_ROOT)/%.od: $(SRC_ROOT)/%.cpp
	$(CC) $(DFLAGS) -c $< $(INCLUDES) -o $@ $(LIBS_D)

$(OBJ_ROOT)/%.o: $(SRC_ROOT)/%.cpp
	$(CC) $(RFLAGS) -c $< $(INCLUDES) -o $@ $(LIBS)

$(TRAINER_EXE_D): $(TRAINER_SRC) $(OBJS_D)
	$(CC) $(DFLAGS) $(TRAINER_SRC) -o $@ $(INCLUDES) $(OBJS_D) $(LIBS_D)

$(TRAINER_EXE): $(TRAINER_SRC) $(OBJS)
	$(CC) $(RFLAGS) $(TRAINER_SRC) -o $@ $(INCLUDES) $(OBJS) $(LIBS)

$(UNIT_EXE_D): $(UNIT_SRC) $(OBJS_D)
	$(CC) $(DFLAGS) $(UNIT_SRC) -o $@ $(INCLUDES) $(OBJS_D) $(LIBS_D) -lgtest_main -lpthread
	
$(UNIT_EXE): $(UNIT_SRC) $(OBJS)
	$(CC) $(RFLAGS) $(UNIT_SRC) -o $@ $(INCLUDES) $(OBJS_D) $(LIBS) -lgtest_main -lpthread

setup:
	mkdir -p obj

clean:
	rm -f obj/* $(TRAINER_EXE_D) $(TRAINER_EXE) $(UNIT_EXE_D) $(UNIT_EXE)
