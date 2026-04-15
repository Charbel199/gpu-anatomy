NVCC      := nvcc
NVCCFLAGS := -O3 -lineinfo
LDFLAGS   := -lnvidia-ml
ARCH      := -arch=native
INCLUDES  := -I common

SOURCES := $(shell find . -name '*.cu' -not -path './common/*')
BINS    := $(patsubst ./%.cu, bin/%, $(SOURCES))

all: $(BINS)

bin/%: %.cu common/*.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(INCLUDES) $< -o $@ $(LDFLAGS)

folder:
	@$(MAKE) $(filter bin/$(folder)/%, $(BINS))

run:
	./bin/$(EXP)

clean:
	rm -rf bin/

.PHONY: all clean run folder
