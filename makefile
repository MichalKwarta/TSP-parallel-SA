outputNameSEQ := main
outputNameMP := mainMP
outputNameCUDA := mainCUDA
n = 50
filename = "data/berlin52.txt"
OLevel = -O3

buildSEQ:
	@g++ ${OLevel} ${wildcard SEQ/*.cpp} -o ${outputNameSEQ}
buildMP:
	@g++ ${OLevel} -fopenmp ${wildcard MP/*.cpp} -o ${outputNameMP}
buildCUDA:
	@nvcc ${OLevel} ${wildcard CUDA/*.cu} -o ${outputNameCUDA}

runSEQ:
	@time ./${outputNameSEQ} ${filename}
runMP:
	@time ./${outputNameMP} ${filename} ${n}
runCUDA:
	@time ./${outputNameCUDA} ${filename} ${n}

SEQ: buildSEQ runSEQ
MP: buildMP runMP
CUDA: buildCUDA runCUDA	


clean:
	rm -f ${outputNameseq} ${outputNameMP} ${outputNameCUDA}