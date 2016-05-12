# Put the directory above 'DNest4' into an environment variable
# called DNEST4_PATH for this to work.

CC = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -pedantic -DNDEBUG
LIBS = -ldnest4 -lpthread -pthread -Wl,--no-as-needed
EIGEN_PATH = /usr/local/shared/eigen3/include/eigen3/

default:
	$(CC) -I$(DNEST4_PATH) -I$(EIGEN_PATH) $(CXXFLAGS) -c Data.cpp MyConditionalPrior.cpp MyModel.cpp main.cpp
	$(CC) -L$(DNEST4_PATH)/DNest4/code -o main *.o $(LIBS)
	rm *.o
