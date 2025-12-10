#!/bin/bash
./gaze_pyside_app.py --dev 0 -P 1 |
	grep '^PLOT:' |
	sed -e 's/^PLOT://' -e 's/$/ y0:-.4 yz:.8/' |
	splotty --stdin -f splotty-fields.yaml 
