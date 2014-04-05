CC=g++
FLAGS=-std=c++11 -g -mavx2
DFLAGS=$(FLAGS)
RFLAGS=-O3 $(FLAGS)

INC_ROOT = include
SRC_ROOT = src

SRC = $(wildcard $(SRC_ROOT)/*.cpp)

OBJ_ROOT = obj

OBJS = $(patsubst $(SRC_ROOT)/%.cpp,$(OBJ_ROOT)/%.o,$(SRC))
OBJS_D = $(patsubst %.o, %.od,$(OBJS))

NET_SRC=$(wildcard $(SRC_ROOT)/*.cpp)
TRAINER_SRC=$(wildcard Trainer/*.cpp)

TRAINER_EXE=trainer
TRAINER_EXE_D=d_trainer

INCLUDES=-I$(INC_ROOT) -I../axon/include -I../ThirdParty/eigen-3-2-1 -I../ThirdParty/boost_1_55_0

LIBS_BASE=-L../ThirdParty/boost_1_55_0/lib -L../axon/lib  -lboost_system -lboost_filesystem -lboost_program_options
LIBS=$(LIBS_BASE) -laxcomm -laxser -laxutil
LIBS_D=$(LIBS_BASE) -laxcommd -laxserd -laxutild

.PHONY: all clean setup

all: debug release 

debug: setup $(OBJS_D) $(TRAINER_EXE_D)

release: setup $(OBJS) $(TRAINER_EXE)

$(OBJ_ROOT)/%.od: $(SRC_ROOT)/%.cpp
	$(CC) $(DFLAGS) -c $< $(INCLUDES) -o $@ $(LIBS_D)

$(OBJ_ROOT)/%.o: $(SRC_ROOT)/%.cpp
	$(CC) $(RFLAGS) -c $< $(INCLUDES) -o $@ $(LIBS)

d_trainer: $(TRAINER_SRC) $(OBJS_D)
	$(CC) $(DFLAGS) $(TRAINER_SRC) -o $@ $(INCLUDES) $(OBJS_D) $(LIBS_D)

trainer: $(TRAINER_SRC) $(OBJS)
	$(CC) $(RFLAGS) $(TRAINER_SRC) -o $@ $(INCLUDES) $(OBJS) $(LIBS)

setup:
	mkdir -p obj

clean:
	rm -f obj/* d_trainer trainer
