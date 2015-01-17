reset
width=22200 #TODO adjust width depending on histog.dat
set boxwidth  width*0.9
set style fill solid 0.5
set xlabel "Pressure (Pa)"
set ylabel "Frequency"
set title "Histogram"
plot "histog.dat" using 1:2:3 w boxes lc rgb variable notitle
pause -1

