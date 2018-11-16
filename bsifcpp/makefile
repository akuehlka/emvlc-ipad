CXX?=g++

CXXFLAGS=-O0 -std=c++11 -Wall -fmessage-length=0 \
					-g -gdwarf-3 -Wformat  -fPIC
INCLUDES=-I/afs/crc.nd.edu/user/a/akuehlka/.local/opencv-3.2.0/include
LINKFLAGS=-L/afs/crc.nd.edu/user/a/akuehlka/.local/opencv-3.2.0/lib \
					-lopencv_imgproc \
					-lopencv_core \
					-lopencv_highgui \
					-lopencv_imgcodecs

# this is to link the wrapper module
MODULEINCLUDES:=$(INCLUDES) -I/afs/crc.nd.edu/user/a/akuehlka/.local/boost/include \
					-I/afs/crc.nd.edu/x86_64_linux/p/python/3.6.4/gcc/include/python3.6m
MODULELINKFLAGS:=$(LINKFLAGS) -L/afs/crc.nd.edu/x86_64_linux/p/python/3.6.4/gcc/lib \
					-lpython3.6m \
					-L/afs/crc.nd.edu/user/a/akuehlka/.local/boost/lib \
					-lboost_python3 \
					-lboost_numpy3

all: libbsif.so

libbsif.so: filtermap.o bsif.o bsifmodule.o
	$(CXX) -shared $(MODULEINCLUDES) -o $@ $(CXXFLAGS) $^ $(MODULELINKFLAGS)

bsif.o: filtermap.o bsif.cpp
	$(CXX) -c $(INCLUDES) -o $@ $(CXXFLAGS) $^ $(LINKFLAGS)

test: test.cpp
	$(CXX) $(MODULEINCLUDES) -o $@ $(CXXFLAGS) $^ -L. -lbsif $(MODULELINKFLAGS)

bsifmodule.o: bsif.o bsifmodule.cpp
	$(CXX) -c $(MODULEINCLUDES) -o $@ $(CXXFLAGS) $^ $(MODULELINKFLAGS)

filtermap.o: filtermap.cpp
	$(CXX) -c $(INCLUDES) -o $@ $(CXXFLAGS) $^ $(LINKFLAGS)

clean:
	rm -f *o bsifcpp test
