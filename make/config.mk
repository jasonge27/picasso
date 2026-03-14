#-----------------------------------------------------
#  the configuration compile script from the xgboost project.
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory.
#  First copy this file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries in the copied file and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#----------------------------------------------------

# choice of compiler, by default use system preference.
# export CC = gcc
# export CXX = g++
# export MPICXX = mpicxx

# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

# Whether enable openmp support, needed for multi-threading.
USE_OPENMP = 1