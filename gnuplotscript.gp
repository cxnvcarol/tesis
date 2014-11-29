reset
width=3 #TODO adjust width depending on histog.dat
set boxwidth  width*0.9
set style fill solid 0.5
set xlabel "x"
set ylabel "Frequency"
plot "histog.dat" using 1:2 smooth freq w boxes lc rgb"blue" notitle
pause -1

