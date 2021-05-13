for r in $(seq 10 10 100); do 
	mkdir "r_$r"
	cd "r_$r"
	cp ../run.sh .
	ln -s ../faces.py .
	sed -i "s/__R__/$r/g" run.sh
	# sbatch run.sh 
    qsub run.sh
	cd ..
done
