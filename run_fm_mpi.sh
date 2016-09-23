#!/bin/bash
process_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/factorization-machine-ftrl-mpi/train
done
scp train worker@10.101.2.89:/home/worker/xiaoshu/factorization-machine-ftrl-mpi/.
scp train worker@10.101.2.90:/home/worker/xiaoshu/factorization-machine-ftrl-mpi/.
#mpirun -f ./hosts -np $process_number ./train ftrl 1000 500 0.0 0.1 1.0 0.001 0.0 ./data/v2v_train ./data/v2v_test
#mpirun -f ./hosts -np $process_number ./train ftrl 1 500 1.0 1.0 1.0 0.1 1.0 ./data/traindataold ./data/testdataold
mpirun -f ./hosts -np $process_number ./train ftrl 10 10 0.0 0.1 1.0 0.001 0.0 ./data/agaricus.txt.train ./data/agaricus.txt.test
