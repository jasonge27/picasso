ifndef config
ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif
endif

ifndef DMLC_CORE
	DMLC_CORE = dmlc-core
endif

ROOTDIR = $(CURDIR)

ifeq ($(OS), Windows_NT)
	UNAME="Windows"
else
	UNAME=$(shell uname)
endif

include $(config)
ifeq ($(USE_OPENMP), 0)
	export NO_OPENMP = 1
endif
include $(DMLC_CORE)/make/dmlc.mk

# include the plugins
include $(PICASSO_PLUGINS)

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

export LDFLAGS= -pthread -lm $(ADD_LDFLAGS) $(DMLC_LDFLAGS) $(PLUGIN_LDFLAGS)
export CFLAGS=  -std=c++11 -Wall -Wno-unknown-pragmas -Iinclude $(ADD_CFLAGS) $(PLUGIN_CFLAGS)
CFLAGS += -I$(DMLC_CORE)/include 

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


ifeq ($(USE_OPENMP), 1)
	CFLAGS += -fopenmp
else
	CFLAGS += -DDISABLE_OPENMP
endif


# specify tensor path
.PHONY: clean all lint clean_all doxygen rcpplint pypack Rpack Rbuild Rcheck pylint


all: lib/libpicasso.a $(PICASSO_DYLIB) picasso

$(DMLC_CORE)/libdmlc.a: $(wildcard $(DMLC_CORE)/src/*.cc $(DMLC_CORE)/src/*/*.cc)
	+ cd $(DMLC_CORE); $(MAKE) libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)


SRC = $(wildcard src/*.cc src/*/*.cc)
ALL_OBJ = $(patsubst src/%.cc, build/%.o, $(SRC)) $(PLUGIN_OBJS)
AMALGA_OBJ = amalgamation/picasso-all0.o
LIB_DEP = $(DMLC_CORE)/libdmlc.a 
ALL_DEP = $(filter-out build/cli_main.o, $(ALL_OBJ)) $(LIB_DEP)
CLI_OBJ = build/cli_main.o
include tests/cpp/picasso_test.mk

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CFLAGS) $< -o $@

build_plugin/%.o: plugin/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build_plugin/$*.o $< >build_plugin/$*.d
	$(CXX) -c $(CFLAGS) $< -o $@

# The should be equivalent to $(ALL_OBJ)  except for build/cli_main.o
amalgamation/picasso-all0.o: amalgamation/picasso-all0.cc
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

picasso: $(CLI_OBJ) $(ALL_DEP)
	$(CXX) $(CFLAGS) -o $@  $(filter %.o %.a, $^)  $(LDFLAGS)

rcpplint:
	python2 dmlc-core/scripts/lint.py picasso ${LINT_LANG} R-package/src

lint: rcpplint
	python2 dmlc-core/scripts/lint.py picasso ${LINT_LANG} include src plugin python-package

pylint:
	flake8 --ignore E501 python-package
	flake8 --ignore E501 tests/python

test: $(ALL_TEST)

check: test
	./tests/cpp/picasso_test

ifeq ($(TEST_COVER), 1)
cover: check
	@- $(foreach COV_OBJ, $(COVER_OBJ), \
		gcov -pbcul -o $(shell dirname $(COV_OBJ)) $(COV_OBJ) > gcov.log || cat gcov.log; \
	)
endif

clean:
	$(RM) -rf build build_plugin lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o picasso
	$(RM) -rf build_tests *.gcov tests/cpp/picasso_test

clean_all: clean
	cd $(DMLC_CORE); $(MAKE) clean; cd $(ROOTDIR)

doxygen:
	doxygen doc/Doxyfile

# create standalone python tar file.
pypack: ${PICASSO_DYLIB}
	cp ${PICASSO_DYLIB} python-package/picasso
	cd python-package; tar cf picasso.tar picasso; cd ..

# create pip installation pack for PyPI
pippack:
	$(MAKE) clean_all
	rm -rf picasso-python
	cp -r python-package picasso-python
	cp -r Makefile picasso-python/picasso/
	cp -r make picasso-python/picasso/
	cp -r src picasso-python/picasso/
	cp -r include picasso-python/picasso/
	cp -r dmlc-core picasso-python/picasso/

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
	mkdir -p picasso/src/dmlc-core
	cp -r dmlc-core/include picasso/src/dmlc-core/include
	cp -r dmlc-core/src picasso/src/dmlc-core/src
	cp ./LICENSE picasso
	cat R-package/src/Makevars.in|sed '2s/.*/PKGROOT=./' | sed '3s/.*/ENABLE_STD_THREAD=0/' > picasso/src/Makevars.in
	cp picasso/src/Makevars.in picasso/src/Makevars.win
	sed -i -e 's/@OPENMP_CXXFLAGS@/$$\(SHLIB_OPENMP_CFLAGS\)/g' picasso/src/Makevars.win

Rbuild:
	$(MAKE) Rpack
	R CMD build --no-build-vignettes picasso
	rm -rf picasso

Rcheck:
	$(MAKE) Rbuild
	R CMD check  picasso*.tar.gz

-include build/*.d
-include build/*/*.d
-include build_plugin/*/*.d
