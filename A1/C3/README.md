# Parse the logs examples, Fig.13
* `cat logs/xvit* | grep HERE1  | grep "orbit'" | grep "embed_dim=4096" | grep "channels=512" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit'" | grep "embed_dim=6144" | grep "channels=256" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit'" | grep "embed_dim=8192" | grep "channels=128" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit_linear'" | grep "embed_dim=4096" | grep "channels=512" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit_linear'" | grep "embed_dim=6144" | grep "channels=256" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit_linear'" | grep "embed_dim=8192" | grep "channels=128" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit_hier'" | grep "embed_dim=4096" | grep "channels=512" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit_hier'" | grep "embed_dim=6144" | grep "channels=256" | grep "tensor_par_size=16"`
* `cat logs/xvit* | grep HERE1  | grep "orbit_hier'" | grep "embed_dim=8192" | grep "channels=128" | grep "tensor_par_size=16"`

# Run scripts example, Fig.13
* `sbatch sub_run_512_channels.sh`
* `sbatch sub_run_256_channels.sh`
* `sbatch sub_run_128_channels.sh`


