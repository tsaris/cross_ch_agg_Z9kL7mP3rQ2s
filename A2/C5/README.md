# Parse the logs examples, Fig.16
* `cat logs/xvit_7B_orbit_4N.o* | grep HERE1`
* `cat logs/xvit_7B_orbit_32N.o* | grep HERE1`
* `cat logs/xvit_7B_orbit_128N.o* | grep HERE1`
* `cat logs/xvit_7B_orbit_linear1_8N.o* | grep HERE1`
* `cat logs/xvit_7B_orbit_linear1_32N.o* | grep HERE1`
* `cat logs/xvit_7B_orbit_linear1_64N.o* | grep HERE1`

# Run scripts example, Fig.16
* `sbatch sub_run_7B_baseline_4N.sh`
* `sbatch sub_run_7B_dchag_4N.sh`
* `sbatch sub_run_7B_baseline_128N.sh`
* `sbatch sub_run_7B_dchag_128N.sh`
