##!/usr/bin/env bash

# This whole script is written only for my osx

DEST="/usr/local/opt"
CURR=`pwd`
SYMLINK="../Cellar/gcc@9/9.3.0_2"
FSYMLINK="../Cellar/gcc/10.2.0_4"

echo "changing directory to $DEST"
cd "$DEST"
echo "making new symlink gcc -> $SYMLINK"
rm -f gcc && ln -s "$SYMLINK" gcc
echo "moving directory to $CURR"
cd "$CURR"
echo "running program $1"
python3 "$1"
echo "moving back to gcc-10 in directory $DEST"
cd "$DEST"
rm -f gcc && ln -s "$FSYMLINK" gcc
cd "$CURR"

