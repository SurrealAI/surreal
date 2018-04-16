# URL='surrealai/surreal-cpu'
URL = 'us.gcr.io/surreal-dev-188523/surreal-cpu'
test:
	docker run -ti --privileged -v `pwd`/:/code -v $(HOME)/.mujoco:/mujoco $(URL) --cmd python /code/test_install.py "a b" 'c d e' f

push:
	gcloud docker -- push $(URL):`cat VERSION`
	gcloud docker -- push $(URL):latest

pull:
	docker pull $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` -f Dockerfile-cpu .
	docker tag $(URL):`cat VERSION` $(URL):latest

# Can't do as container requires gpu hosts
# test-gpu:

# GPU_URL='surrealai/surreal-gpu'
GPU_URL = 'us.gcr.io/surreal-dev-188523/surreal-gpu'
push-gpu:
	gcloud docker -- push $(GPU_URL):`cat VERSION`
	gcloud docker -- push $(GPU_URL):latest

pull-gpu:
	docker pull $(GPU_URL):latest

build-gpu:
	docker build -t $(GPU_URL):`cat VERSION` -f Dockerfile-gpu .
	docker tag $(GPU_URL):`cat VERSION` $(GPU_URL):latest
