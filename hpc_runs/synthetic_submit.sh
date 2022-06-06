funcnames='mult_a-normal mult_a-uniform mult_a-exp
mult_b-normal mult_b-uniform mult_b-exp
mult_c-normal mult_c-uniform mult_c-exp
add_a-normal add_a-uniform add_a-exp
add_b-normal add_b-uniform add_b-exp
add_c-normal add_c-uniform add_c-exp
complex_a-normal complex_a-uniform complex_a-exp
complex_b-normal complex_b-uniform complex_b-exp
complex_c-normal complex_c-uniform complex_c-exp '

for i in $funcnames
do
    qsub -v "FN=$i" -oe -N "FN-$i" hpc_runs/synthetic_runs.pbs;
done