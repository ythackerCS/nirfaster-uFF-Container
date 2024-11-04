cp Dockerfile.base Dockerfile && \
./command2label.py ./xnat/command.json >> Dockerfile && \
docker build --no-cache -t xnat/nirfaster-uFF:latest .
docker tag xnat/nirfaster-uFF:latest registry.nrg.wustl.edu/docker/nrg-repo/yash/nirfaster-uFF:latest
rm Dockerfile