CFLAGS = -O3
LIBS =  -lrjobject -ldnest3 -lgsl -lgslcblas -lboost_system -lboost_thread

default:
	g++ $(CFLAGS) -I/home/brendon/Projects/Forks/HODLR/header -c *.cpp
	g++ -o main *.o $(LIBS)
	rm -f *.o

