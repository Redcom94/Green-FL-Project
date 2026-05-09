# Benchmarking platform for Federated Learning
## Set up the project
### Install dependencies and project

This branch is if you'd rather use Docker images rather than a virtual environment. 
Make sure docker has acccess to NVIDIA GPU
```bash
sudo apt-get install -y nvidia-container-toolkit
```
```bash
sudo systemctl restart docker
```
## Build and launch the docker Image

```bash
sudo docker-compose up --build
```
After that, open localhost:8501.
