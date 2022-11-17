#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

cd /sdcard/fwc/circle
python3 circle.py

cd /sdcard/fwc/circle
#pdflatex assembly.tex
#xdg-open assembly.pdf
texfot pdflatex circle.tex
termux-open circle.pdf


#Test Python Installation
#Uncomment only the following line
#python3 /data/data/com.termux/files/home/storage/shared/training/math/codes/tri_sss.py

