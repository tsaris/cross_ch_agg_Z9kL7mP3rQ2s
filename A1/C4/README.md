# Parse the logs examples, Fig.14
* `cat logs/* | grep HERE1  | grep "orbit_token" | grep tensor_par_size=16`
* `cat logs/* | grep HERE1  | grep "orbit_token" | grep tensor_par_size=32`
* `cat logs/* | grep HERE1  | grep "orbit_token" | grep tensor_par_size=64`
* `cat logs/* | grep HERE1  | grep "orbit_linear'" | grep tensor_par_size=16`
* `cat logs/* | grep HERE1  | grep "orbit_linear'" | grep tensor_par_size=32`
* `cat logs/* | grep HERE1  | grep "orbit_linear'" | grep tensor_par_size=64`
* `cat logs/* | grep HERE1  | grep "orbit_linear_token_agg'" | grep tensor_par_size=16`
* `cat logs/* | grep HERE1  | grep "orbit_linear_token_agg'" | grep tensor_par_size=32`
* `cat logs/* | grep HERE1  | grep "orbit_linear_token_agg'" | grep tensor_par_size=64`

# Run scripts example, Fig.14
* `sbatch sub_run_2N.sh`
* `sbatch sub_run_4N.sh`
* `sbatch sub_run_8N.sh`


