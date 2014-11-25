reset
set datafile separator ","
width=3
set boxwidth  width*0.9
set style fill solid 0.5
set xlabel "x"
set ylabel "Frequency"
plot "histogram.dat" using 1:2 smoot freq w boxes lc rgb"green" notitle
pause -1

