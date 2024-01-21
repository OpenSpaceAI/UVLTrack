#!/usr/bin/env bash

tracker=$1
template=${2:-'mae'}

cp -r experiments/${template} experiments/$tracker
cp -r lib/config/${template} lib/config/$tracker
cp -r lib/models/${template} lib/models/$tracker
mv lib/models/$tracker/${template}.py lib/models/$tracker/$tracker.py
cp lib/train/actors/${template}.py lib/train/actors/$tracker.py

cp lib/test/parameter/${template}.py lib/test/parameter/$tracker.py
cp lib/test/tracker/${template}.py lib/test/tracker/$tracker.py

echo "\n"from .$tracker import $tracker >> lib/models/__init__.py
echo "\n"from .$tracker import '*' >> lib/train/actors/__init__.py

echo "The following file need to be modified: "
echo "lib/models/$tracker/$tracker.py"
echo "lib/train/actors/$tracker.py"
echo "lib/test/parameter/$tracker.py"
echo "lib/test/tracker/$tracker.py"