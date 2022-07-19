for i in 0,20 20,40 40,60 60,80 80,100 100,120 120,140 140,160 160,180 180,200 200,220 220,240 240,260 260,280 280,300
do
    IFS=",";
    set -- $i;
    qsub -v "DS=$1,DE=$2" -oe -N "ce_gauss-DS-$1-DE-$2" hpc_runs/synthetic_ce_runs.pbs;
done