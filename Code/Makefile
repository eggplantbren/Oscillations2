CFLAGS = -O3
LIBS =  -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_system -lboost_thread
OBJECTS = Data.o MyDistribution.o MyModel.o NotPeriodogram.o ODE.o StateSpace.o

default:
	g++ $(CFLAGS) -I/home/brendon/Projects/Forks/HODLR/header -c *.cpp
	g++ -o main main.o $(OBJECTS) $(LIBS)
	g++ -o test_ODE test_ODE.o $(OBJECTS) $(LIBS)
	rm -f *.o

