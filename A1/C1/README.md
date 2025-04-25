# Parse the logs examples, Fig.7
* `cat logs/* | grep HERE1  | grep "orbit\'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/* | grep HERE1  | grep "orbit_token'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/* | grep HERE1  | grep "orbit_token_agg'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/* | grep HERE1  | grep "orbit\'" | grep "embed_dim=4096" | grep "channels=256"`

# Run scripts example, Fig.7
* `sbatch sub_run_256_channels.sh`
* `sbatch sub_run_512_channels.sh`

