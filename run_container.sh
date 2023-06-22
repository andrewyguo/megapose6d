docker run --gpus all -it --name megapose -v /home/andrewg/pose/megapose6d/:/megapose  --ipc=host   --network=host megapose:latest bash 

# -v /home/andrewg/instant-ngp/:/instant-ngp 