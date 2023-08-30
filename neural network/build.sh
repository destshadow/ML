#!/bin/sh

set -xe

clang -Wall -Wextra -lm nn.c -o nn
clang -Wall -Wextra -lm adder.c -o adder
clang -Wall -Wextra -lm dump_nn.c -o dump