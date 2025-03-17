# HZZ_analysis
Docker + Kubernetes for the analysis of CERN data

## Instructions
* cd to the directory (hzz_task_split)
* Paste this into the command line:
```  
docker service create --name rabbitmq --replicas 1 rabbitmq:latest
kubectl apply -f PVC.yaml
kubectl apply -f rabbitmq-deployment.yaml
kubectl apply -f celery-worker-deployment.yaml
kubectl cp chunk_sizes.txt <rabbitmq>:app/data/chunk_sizes.txt
kubectl apply -f celery-job.yaml![image](https://github.com/user-attachments/assets/fb113079-4ebb-4337-bc3d-27f35062e481)
```
