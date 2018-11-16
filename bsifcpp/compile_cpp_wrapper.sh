# https://stackoverflow.com/questions/15631882/python-extension-symbols-not-found-for-architecture-x86-64-error
# see answer by hupantingxue
g++ -fPIC -c \
  -I/Users/akuehlka/anaconda/envs/cv/include/python3.5m \
  -I/Users/akuehlka/anaconda/envs/cv/include/python3.5m \
  -o spammodule.o \
  spammodule.cpp

g++ -shared \
  -L/Users/akuehlka/anaconda/envs/cv/lib \
  -lpython3.5m \
  -ldl \
  -framework CoreFoundation \
  -framework CoreFoundation \
  spammodule.o \
  -o spam.so
