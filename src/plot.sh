#!/bin/bash
s0() { stdbuf -i0 -o0 "$@"; }
./gaze_pyside_app.py --dev 0 -P 1 |
	s0 grep '^PLOT:' |
	s0 sed -e 's/^PLOT://' -e 's/$/ y0:-.4 yz:.8/' |
	s0 splotty --stdin -f splotty-fields.yaml 

