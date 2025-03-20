# HZZ_analysis
Docker + Kubernetes for the analysis of CERN data.

## Instructions
* cd to the directory (hzz_task_split).
* Modify chunk_sizes.txt to change chunk size for each process.
* 'data' and 'zztbar' have low sample lengths, maximum of 400 and 1031 respectively.
* 'signal' and 'zz' have higher sample lengths, maximum of 191126 and 554279 respectively.
* Keep this in mind when choosing chunk sizes.
* Worker replica numbers can be modified in celery-worker-deployment.yaml
* Paste this into the command line:
```  
docker service create --name rabbitmq --replicas 1 rabbitmq:latest
kubectl apply -f PVC.yaml
kubectl apply -f rabbitmq-deployment.yaml
kubectl apply -f celery-worker-deployment.yaml
```
* Use kubectl get pods to get the name of the rabbitmq pod, then:
```
kubectl cp chunk_sizes.txt <rabbitmq>:app/data/chunk_sizes.txt
kubectl apply -f celery-job.yaml
```
* Once all jobs are completed, type this to save the plot and copy to your local filesystem:
```
kubectl apply -f saveplot.yaml
kubectl cp <rabbitmq>:app/data/fig.png fig.png
```

## Information
* hzz_task is an old version where each sample isnt analysed in parallel. It also uses a slower method of collection with celery.
* add_test is a folder to test celery with splitting tasks between workers.
