outputNameSEQ := main
outputNameMP := mainMP
outputNameCUDA := mainCUDA
n = 50
filename = "data/tsp250.txt"
OLevel = -O3

buildSEQ:
	@g++ ${OLevel} ${wildcard SEQ/*.cpp} -o ${outputNameSEQ}
buildMP:
	@g++ ${OLevel} -fopenmp ${wildcard MP/*.cpp} -o ${outputNameMP}


runSEQ:
	@time ./${outputNameSEQ} ${filename}
runMP:
	@time ./${outputNameMP} ${filename} ${n}


SEQ: buildSEQ runSEQ
MP: buildMP runMP


clean:
	rm -f ${outputNameseq} ${outputNameMP} ${outputNameCUDA}