.PHONY: push-c04 pull-c04 build-c04 publish-c04 push-g04 pull-g04 build-g04 publish-g04 

# URL='surrealai/surreal-cpu'
URL = 'us.gcr.io/surreal-dev-188523/surreal-cpu'
test:
	docker run -ti --privileged -v `pwd`/:/code -v $(HOME)/.mujoco:/mujoco $(URL) --cmd python /code/test_install.py "a b" 'c d e' f

push:
	docker push $(URL):`cat VERSION`
	docker push $(URL):latest

pull:
	docker pull $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` -f Dockerfile-cpu .
	docker tag $(URL):`cat VERSION` $(URL):latest


rebuild:
	docker build --no-cache -t $(GPU_URL):`cat VERSION` -f Dockerfile-gpu .
	docker tag $(GPU_URL):`cat VERSION` $(GPU_URL):latest

# TODO: add version
CPU_04_URL = 'us.gcr.io/surreal-dev-188523/surreal-cpu-th04'
push-c04:
	docker push $(CPU_04_URL):latest

pull-c04:
	docker pull $(CPU_04_URL):latest

build-c04:
	docker build -t $(CPU_04_URL):latest -f Dockerfile-cpu04 .

publish-c04: build-c04 push-c04

# TODO: add version
GPU_04_URL = 'us.gcr.io/surreal-dev-188523/surreal-gpu-th04'
push-g04:
	docker push $(GPU_04_URL):latest

pull-g04:
	docker pull $(GPU_04_URL):latest

build-g04:
	docker build -t $(GPU_04_URL):latest -f Dockerfile-gpu04 .

publish-g04: build-g04 push-g04

GPU_URL = 'us.gcr.io/surreal-dev-188523/surreal-gpu'
push-gpu:
	docker push $(GPU_URL):`cat VERSION`
	docker push $(GPU_URL):latest

pull-gpu:
	docker pull $(GPU_URL):latest

build-gpu:
	docker build -t $(GPU_URL):`cat VERSION` -f Dockerfile-gpu .
	docker tag $(GPU_URL):`cat VERSION` $(GPU_URL):latest

rebuild-gpu:
	docker build --no-cache -t $(GPU_URL):`cat VERSION` -f Dockerfile-gpu .
	docker tag $(GPU_URL):`cat VERSION` $(GPU_URL):latest

CPU_BASE_URL = 'us.gcr.io/surreal-dev-188523/surreal-base-cpu'
build-cpubase:
	docker build -t $(CPU_BASE_URL):latest -f DockerfileCPUBase .

push-cpubase:
	docker push $(CPU_BASE_URL):latest

pull-cpubase:
	docker pull $(CPU_BASE_URL):latest

GPU_BASE_URL = 'us.gcr.io/surreal-dev-188523/surreal-base-gpu'
build-gpubase:
	docker build -t $(GPU_BASE_URL):latest -f DockerfileGPUBase .

push-gpubase:
	docker push $(GPU_BASE_URL):latest

pull-gpubase:
	docker pull $(GPU_BASE_URL):latest
