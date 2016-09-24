#!/bin/bash
process_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/factorization-machine-ftrl-mpi/fm_ftrl_mpi
done
scp fm_ftrl_mpi worker@10.101.2.89:/home/worker/xiaoshu/factorization-machine-ftrl-mpi/.
scp fm_ftrl_mpi worker@10.101.2.90:/home/worker/xiaoshu/factorization-machine-ftrl-mpi/.
mpirun -f ./hosts -np $process_number ././fm_ftrl_mpi ftrl 200 500 0.0 0.1 1.0 0.001 0.0 ./data/v2v_train ./data/v2v_test
#mpirun -f ./hosts -np $process_number ./fm_ftrl_mpi ftrl 10 50 0.0 0.1 1.0 0.001 0.0 ./data/agaricus.txt.train ./data/agaricus.txt.test
