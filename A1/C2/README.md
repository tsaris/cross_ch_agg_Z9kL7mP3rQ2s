# Parse the logs examples, Fig.8
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_token_agg'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_token'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_hier_token'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_hier_token_allGather'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_token_agg'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_token'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_hier_token'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit.o* | grep HERE1  | grep "orbit_hier_token_allGather'" | grep "embed_dim=2048" | grep "channels=1024"`

# Parse the logs examples, Fig.9
* `cat logs/xvit_tree_2.o* | grep HERE1  | grep "orbit_hier_tree'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit_tree_4.o* | grep HERE1  | grep "orbit_hier_tree'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit_tree_8.o* | grep HERE1  | grep "orbit_hier_tree'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit_tree_2.o* | grep HERE1  | grep "orbit_linear_tree'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit_tree_4.o* | grep HERE1  | grep "orbit_linear_tree'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit_tree_8.o* | grep HERE1  | grep "orbit_linear_tree'" | grep "embed_dim=2048" | grep "channels=512"`
* `cat logs/xvit_tree_2_1024.o* | grep HERE1  | grep "orbit_hier_tree'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit_tree_4_1024.o* | grep HERE1  | grep "orbit_hier_tree'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit_tree_8_1024.o* | grep HERE1  | grep "orbit_hier_tree'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit_tree_2_1024.o* | grep HERE1  | grep "orbit_linear_tree'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit_tree_4_1024.o* | grep HERE1  | grep "orbit_linear_tree'" | grep "embed_dim=2048" | grep "channels=1024"`
* `cat logs/xvit_tree_8_1024.o* | grep HERE1  | grep "orbit_linear_tree'" | grep "embed_dim=2048" | grep "channels=1024"`

# Run scripts example, Fig.8,9
* `sbatch sub_run_512_channels.sh`
* `sbatch sub_run_1024_channels.sh`


