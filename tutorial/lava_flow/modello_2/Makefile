# Aggiunto un ../ ovunque per compensare la sottocartella "lava_flow"
CPPFLAGS ?= -I../../../include -I../../json/single_include/nlohmann/ -I../../json/single_include -I../../cdf_timer/src
CXXFLAGS ?= -std=c++17 -O3
THRUST_CPU_CXXFLAGS ?= -std=c++20 -O3
QUADGRIDSRC ?= ../../../src/particles.cpp
TIMERSRC ?= ../../cdf_timer/src/timer.cpp
LDFLAGS ?=
LIBS ?=


modello_2 : modello_2.cpp modello_2.h init_scenario.h
	nvcc $(CXXFLAGS) $(CPPFLAGS) -arch=sm_86 --expt-relaxed-constexpr -DUSE_THRUST -x cu $(QUADGRIDSRC) $(TIMERSRC) modello_2.cpp -o modello_2 $(LDFLAGS) $(LIBS)

hip : modello_2.cpp modello_2.h init_scenario.h
	hipcc $(CXXFLAGS) $(CPPFLAGS) -DUSE_THRUST $(QUADGRIDSRC) $(TIMERSRC) modello_2.cpp -o modello_2.hip $(LDFLAGS) $(LIBS)

thrust_cpu : modello_2.cpp modello_2.h init_scenario.h
	g++ $(THRUST_CPU_CXXFLAGS) $(CPPFLAGS) -DUSE_THRUST -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP -DTHRUST_CPU $(QUADGRIDSRC) $(TIMERSRC) modello_2.cpp -o modello_2.cpu $(LDFLAGS) $(LIBS)

clean :
	$(RM) *.o *.csv *.vts *.json modello_2 modello_2.cpu modello_2.hip
