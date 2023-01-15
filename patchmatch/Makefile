#!/usr/bin/env make -f
#
# Makefile
# Jiayuan Mao, 2019-01-09 13:59
# Matthias Wild, 2022-01-14 13:37
#

SRC_DIR = ./csrc
BUILD_DIR := ./build
TARGET := libpatchmatch.so

LIB_TARGET := $(TARGET)
OBJ_DIR := $(BUILD_DIR)/obj
INCLUDE_DIR = -I $(SRC_DIR)

CXX = $(ENVIRONMENT_OPTIONS) g++
CXXFLAGS = -std=c++14 -Ofast -ffast-math -w
# CXXFLAGS += -g
CXXFLAGS += $(shell pkg-config --cflags opencv &>/dev/null && pkg-config --cflags opencv)
CXXFLAGS += $(shell pkg-config --cflags opencv4 &>/dev/null && pkg-config --cflags opencv4)
CXXFLAGS += -fPIC
CXXFLAGS += $(INCLUDE_DIR)
LDFLAGS = -shared -fPIC
LDFLAGS += $(shell pkg-config --libs opencv &>/dev/null && pkg-config --libs opencv)
LDFLAGS += $(shell pkg-config --libs opencv4 &>/dev/null && pkg-config --libs opencv4)

CXXSOURCES = $(shell find $(SRC_DIR) -name "*.cpp")
OBJS = $(addprefix $(OBJ_DIR)/,$(CXXSOURCES:.cpp=.o))
DEPFILES = $(OBJS:.o=.d)

VENV := ../.venv
VPYTHON := $(VENV)/bin/python
DIST:= ../dist

.PHONY: all
all: $(LIB_TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@echo "[CC] $< ..."
	@$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJ_DIR)/%.d: %.cpp
	@mkdir -pv $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(INCLUDE_DIR) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cpp=.o) $(OBJ_DIR)/$(<:.cpp=.d)" "$<" > "$@"

$(LIB_TARGET): $(OBJS)
	@echo "[link] $(LIB_TARGET) ..."
	@$(CXX) $(OBJS) -o $@ $(CXXFLAGS) $(LDFLAGS)

$(VENV):
	python3 -m venv $(VENV)
	export PATH=$(VENV)/bin:$PATH
	$(VPYTHON) -m pip install -e ../".[dev,dist]"

$(DIST): $(VENV)
	$(VPYTHON) -m build .. --outdir $(DIST)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(DIST) $(VENV)

.PHONY: rebuild
rebuild:
	@make clean
	@make $(LIB_TARGET)
	@make $(DIST)

-include $(DEPFILES)

# vim:ft=make
#
