docker run --restart unless-stopped -p 6379:6379 -d  --name redis redis:latest
celery -A main worker -l info
docker build -t age -f docker/Dockerfile .    (build image)
docker run -p 8000:8000 --rm -it --name age age  (run container based on image)
python3 manage.py loaddata fixtures/admin_user.json
python3 manage.py dumpdata auth.user -o admin_user.json --indent 4

docker login registry.git.chalmers.se
docker build -t registry.git.chalmers.se/courses/dit825/2022/group03/dit825-age-detection .


./google-cloud-sdk/bin/gcloud container clusters get-credentials age-detection-k8s
./google-cloud-sdk/bin/gcloud container clusters get-credentials age-detection-k8s --region europe-north1
helm create age-detection-helm-chart


helm /dynamic work with yaml files/ 
kubectl   kubectl get ns
gcloud
gcloud auth


/google-cloud-sdk/bin/gcloud gcloud components install gke-gcloud-auth-plugin    

glpat-JfB_-wL--oXTozRS4pi-

kubectl create secret docker-registry gitlab --docker-server=registry.git.chalmers.se --docker-username=k8s --docker-password=glpat-JfB_-wL--oXTozRS4pi-

helm upgrade --install age-detection .
helm uninstall age-detection

kubectl logs -f age-detection-age-detection-helm-chart-0
kubectl delete pod age-detection-age-detection-helm-chart-0
kubectl describe pods age-detection-age-detection-helm-chart-0



cd age-detection-helm-chart  
kubectl port-forward age-detection-age-detection-helm-chart-0 8000:8000
kubectl port-forward service/age-detection-age-detection-helm-chart 8000:8000

helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm repo list
helm search repo postgresql
helm search repo postgresql -l
helm dep build
kubectl exec -it age-detection-postgresql-0 bash 
kubectl exec -it age-detection-age-detection-helm-chart-0 bash

helm uninstall
helm upgrade
kubectl exec -it age-detection-age-detection-helm-chart-0 bash
cd media/models


