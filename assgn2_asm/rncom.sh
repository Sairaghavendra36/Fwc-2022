#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

cd /home/user/fwc/assembly/assgn1_asm/asm
avra assembly1.asm
avrdude -p atmega328p -c arduino -P /dev/ttyACM0 -b 115200 -U flash:w:assembly1.hex

cd /home/user/fwc/assembly/assgn1_asm
pdflatex assembly.tex
xdg-open assembly.pdf
#texfot pdflatex gvv_math_eg.tex
#termux-open gvv_math_eg.tex


#Test Python Installation
#Uncomment only the following line
#python3 /data/data/com.termux/files/home/storage/shared/training/math/codes/tri_sss.py

