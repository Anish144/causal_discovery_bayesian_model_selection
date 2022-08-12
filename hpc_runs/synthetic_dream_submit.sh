for i in 100,200 200,300 300,400 400,500 500,600
do
    IFS=",";
    set -- $i;
    qsub -v "DS=$1,DE=$2" -oe -N "ce_D4S2A-DS-$1-DE-$2" hpc_runs/synthetic_dream_runs.pbs;
done