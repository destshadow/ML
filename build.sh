#!/bin/sh

#mostra il comando che esegue
set -xe

clang -Wall -Wextra -lm twice.c -o twice
clang -Wall -Wextra -lm gates.c -o gates
clang -Wall -Wextra -lm xor.c -o xor