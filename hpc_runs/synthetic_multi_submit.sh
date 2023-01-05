for i in 180,200
do
    IFS=",";
    set -- $i;
    qsub -v "DS=$1,DE=$2" -oe -N "ce_multi-DS-$1-DE-$2" hpc_runs/synthetic_multi_runs.pbs;
done