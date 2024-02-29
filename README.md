# FL_K8S
Implementing K8S with Jetson TX2 for Federated Learning

## Prerequisites
Ensure the following prerequisites are met:
- Jetson TX2 flashed with JetPack SDK.
- Kubernetes (K8s) installed.

## Environment
- Master Node (Server)
- Worker Nodes (Jetson TX2)

## Steps
Follow these steps to set up and run Federated Learning (FL) with Kubernetes (K8s) on Jetson TX2:

### 1. Clone Repository
Clone the FL_K8S repository on each Jetson TX2 (Worker Node):

```bash
# Clone the repository
git clone https://github.com/your-username/FL_K8S.git
cd FL_K8S
```

### 2. Build Docker Image on Jetson
On each Jetson TX2 (Worker Node), execute the following steps to build the Docker image required for FL:
```
# Navigate to the directory containing your Dockerfile
cd path/to/your/Dockerfile

# Build the Docker image
docker build -t fl_image .
```

### 3. Deploy Kubernetes Manifests on Master Node
On the Master Node (Server), apply the Kubernetes manifests to deploy the FL components:
```
# Apply the Kubernetes manifests
kubectl apply -f your-manifest.yaml
```

### 4. Execute run.py on Master Node
On the Master Node (Server), execute the run.py script to start the Federated Learning process:
```
# Execute the run.py script
python run.py
```

### Additional Notes
* Ensure proper network connectivity between the Master Node and Worker Nodes for seamless communication.
* Monitor the Kubernetes pods to ensure they are running correctly using the kubectl get pods command.
* Customize the YAML manifests according to your FL application's requirements.




