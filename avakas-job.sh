#!/usr/bin/env bash

#############################
# Your job name (displayed by the queue)
#PBS -N PricesEco-simulation

#change output file's name
#PBS -e /home/anioche/aurelien/PricesEco-master/avakas_logs/PricesEco-simulation.err

#PBS -o /home/anioche/aurelien/PricesEco-master/avakas_logs/PricesEco-simulation.log


# Specify the working directory
#PBS -d /home/anioche/aurelien/PricesEco-master/PricesEco

# walltime (hh:mm::ss)
#PBS -l walltime=20:00:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(ppn=) to be used
#PBS -l nodes=1:ppn=1

# Specify physical memory: kb for kilobytes, mb for megabytes, gb for gigabytes
#PBS -l mem=1gb

#PBS -m abe
#PBS -M clusterresultssimulation@gmail.com

# fin des directives PBS
#############################

module purge # modules cleaning
module add torque
pyenv local 3.5.2

# useful informations to print
echo "#############################"
echo "User:" ${USER}
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "PBS_JOBID:" ${PBS_JOBID}
echo "PBS_O_WORKDIR:" ${PBS_O_WORKDIR}
echo "PBS_NODEFILE: " `cat ${PBS_NODEFILE} | uniq`
echo "#############################"

#############################

# What you actually want to launch
echo "Start the job"
echo main.py

# launch python script with pickle object for parameters
python main.py

echo "#############################"
echo "Date:" `date`
echo "Job finished"
echo "#############################"
