CC = g++
LD = $(CC)
 
#TARGET = $(notdir $(CURDIR))
TARGET = app
SRC_DIR = .
INCLUDE_DIR += .

# x86_64
C_FLAGS = -g -Wall -std=c++11 -O3 -march=native -fopenmp
LD_FLAGS =
LD_LIBS = -fopenmp -lpthread
INCLUDES = -I$(INCLUDE_DIR)
 
ifeq ($(CC), g++)
	TYPE = cpp
	SRCS += $(wildcard $(SRC_DIR)/*.$(TYPE))
	OBJS += $(patsubst %.$(TYPE), %.o, $(SRCS))
else
	TYPE = c
	SRCS += $(wildcard $(SRC_DIR)/*.$(TYPE))
	OBJS += $(patsubst %.$(TYPE), %.o, $(SRCS))
endif

all : $(TARGET)
	@echo "Builded target:" $^
	@echo "Done"

$(TARGET) : $(OBJS)
	@echo "Linking" $@ "from" $^ "..."
	$(LD) -o $@ $^ $(LD_FLAGS) $(LD_LIBS)
	@echo "Link finished\n"

$(OBJS) : %.o:%.$(TYPE)
	@echo "Compiling" $@ "from" $< "..."
	$(CC) -c -o $@ $< $(C_FLAGS) $(INCLUDES)
	@echo "Compiled finished\n"

.PHONY : clean cleanobj
clean : cleanobj
	@echo "Remove all executable file"
	rm -f $(TARGET)
cleanobj :
	@echo "Remove binary files"
	rm -f *.o