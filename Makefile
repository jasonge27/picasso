ifndef config
ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif


ROOTDIR = $(CURDIR)

ifeq ($(OS), Windows_NT)
	UNAME="Windows"
else
	UNAME=$(shell uname)
endif


# set compiler defaults for OSX versus *nix
# let people override either
OS := $(shell uname)
ifeq ($(OS), Darwin)
ifndef CC
export CC = $(if $(shell which clang), clang, gcc)
endif
ifndef CXX
export CXX = $(if $(shell which clang++), clang++, g++)
endif
else
# linux defaults
ifndef CC
export CC = gcc
endif
ifndef CXX
export CXX = g++
endif
endif

export LDFLAGS= -pthread -lm $(ADD_LDFLAGS)$(PLUGIN_LDFLAGS)
export CFLAGS=  -std=c++11 -Wall -Wno-unknown-pragmas -I ./include $(ADD_CFLAGS) $(PLUGIN_CFLAGS)

ifeq ($(TEST_COVER), 1)
	CFLAGS += -g -O0 -fprofile-arcs -ftest-coverage
else
	CFLAGS += -O3 -funroll-loops -msse2
endif

ifndef LINT_LANG
	LINT_LANG= "all"
endif

ifneq ($(UNAME), Windows)
	CFLAGS += -fPIC
	PICASSO_DYLIB = lib/libpicasso.so
else
	PICASSO_DYLIB = lib/libpicasso.dll
endif

ifeq ($(UNAME), Linux)
	LDFLAGS += -lrt
endif

# specify tensor path
.PHONY: clean all lint clean_all doxygen rcpplint Pypack Pyinstall Rpack Rbuild Rcheck pylint


all: lib/libpicasso.a $(PICASSO_DYLIB) picasso

dylib: $(PICASSO_DYLIB)

SRC = $(wildcard src/*.cpp src/*/*.cpp)
ALL_OBJ = $(patsubst src/%.cpp, build/%.o, $(SRC)) $(PLUGIN_OBJS)
AMALGA_OBJ = amalgamation/picasso-all0.o
ALL_DEP = $(filter-out build/cli_main.o, $(ALL_OBJ)) $(LIB_DEP)
CLI_OBJ = build/cli_main.o

build/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) $< -o $@

# The should be equivalent to $(ALL_OBJ)  except for build/cli_main.o
amalgamation/picasso-all0.o: amalgamation/picasso-all0.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

# Equivalent to lib/libpicasso_all.so
lib/libpicasso_all.so: $(AMALGA_OBJ) $(LIB_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

lib/libpicasso.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

lib/libpicasso.dll lib/libpicasso.so: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o %a,  $^) $(LDFLAGS)

picasso:  $(CLI_OBJ) $(ALL_DEP)
	$(CXX) $(CFLAGS) -o $@  $(filter %.o %.a, $^)  $(LDFLAGS)

# TODO: lint check
# rcpplint:
# 	python2  picasso ${LINT_LANG} R-package/src
#
# lint: rcpplint
# 	python2 picasso ${LINT_LANG} include src  python-package
#
# pylint:
# 	flake8 --ignore E501 python-package

ifeq ($(TEST_COVER), 1)
cover: check
	@- $(foreach COV_OBJ, $(COVER_OBJ), \
		gcov -pbcul -o $(shell dirname $(COV_OBJ)) $(COV_OBJ) > gcov.log || cat gcov.log; \
	)
endif

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o picasso

clean_all: clean

doxygen:
	doxygen doc/Doxyfile

# create standalone python tar file.
pypack: ${PICASSO_DYLIB}
	cp ${PICASSO_DYLIB} python-package/pycasso
	cd python-package; tar cf pycasso.tar pycasso; tar rf pycasso.tar data; cd ..

# install python-package
Pyinstall: ${PICASSO_DYLIB}
	rm -rf python-package/pycasso/lib/
	mkdir python-package/pycasso/lib/
	cp -rf ${PICASSO_DYLIB} python-package/pycasso/lib/
	cd python-package; python setup.py install; cd ..

# create pip installation pack for PyPI
pippack:
	$(MAKE) clean_all
	rm -rf picasso-python
	cp -rf python-package picasso-python
	rm -rf picasso-python/doc
	mkdir picasso-python/pycasso/src/
	cp -rf amalgamation picasso-python/pycasso/src/
	cp -rf Makefile picasso-python/pycasso/src/
	cp -rf make picasso-python/pycasso/src/
	cp -rf src picasso-python/pycasso/src/
	cp -rf include picasso-python/pycasso/src/
	cp picasso-python/setup-pip.py picasso-python/setup.py
	rm picasso-python/setup-pip.py
	rm -rf picasso-python/pycasso/lib/
	mkdir picasso-python/pycasso/lib/

# run pippack first!
pipupload:
	cd picasso-python; python setup.py register sdist upload; cd ..


# Script to make a clean installable R package.
Rpack:
	$(MAKE) clean_all
	rm -rf picasso picasso*.tar.gz
	cp -r R-package picasso
	rm -rf picasso/src/*.o picasso/src/*.so picasso/src/*.dll
	rm -rf picasso/src/*/*.o
	rm -rf picasso/demo/*.model picasso/demo/*.buffer picasso/demo/*.txt
	rm -rf picasso/demo/runall.R
	cp -r src picasso/src/src
	cp -r include picasso/src/include
	cp -r amalgamation picasso/src/amalgamation
	cat R-package/src/Makevars.in|sed '2s/.*/PKGROOT=./' | sed '3s/.*/ENABLE_STD_THREAD=0/' > picasso/src/Makevars.in
	cp picasso/src/Makevars.in picasso/src/Makevars.win
	cp picasso/src/Makevars.in picasso/src/Makevars

Rbuild:
	$(MAKE) Rpack
	R CMD build --no-build-vignettes picasso
	rm -rf picasso

Rcheck:
	$(MAKE) Rbuild
	R CMD check  picasso*.tar.gz

Rinstall:
	$(MAKE) Rbuild
	R CMD INSTALL  picasso*.tar.gz

-include build/*.d
-include build/*/*.d
