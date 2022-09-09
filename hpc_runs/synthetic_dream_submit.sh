for i in 0,100
do
    IFS=",";
    set -- $i;
    qsub -v "DS=$1,DE=$2" -oe -N "ce_D4S2C-DS-$1-DE-$2" hpc_runs/synthetic_dream_runs.pbs;
done