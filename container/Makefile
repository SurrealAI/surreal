.PHONY: push-cpubase pull-cpubase build-cpubase publish-cpubase push-gpubase pull-gpubase build-gpubase publish-gpubase setup-gcloud-credential

setup-gcloud-credential:
	gcloud auth configure-docker

publish-base-all: publish-cpubase publish-gpubase

CPU_BASE_URL = us.gcr.io/surreal-dev-188523/surreal-base-cpu-new
build-cpubase:
	docker build -t $(CPU_BASE_URL):latest -f DockerfileCPUBase .

push-cpubase:
	docker push $(CPU_BASE_URL):latest

pull-cpubase:
	docker pull $(CPU_BASE_URL):latest

publish-cpubase: build-cpubase push-cpubase

GPU_BASE_URL = us.gcr.io/surreal-dev-188523/surreal-base-gpu-new
build-gpubase:
	docker build -t $(GPU_BASE_URL):latest -f DockerfileGPUBase .

push-gpubase:
	docker push $(GPU_BASE_URL):latest

pull-gpubase:
	docker pull $(GPU_BASE_URL):latest

publish-gpubase: build-gpubase push-gpubase

CU90_URL = us.gcr.io/surreal-dev-188523/surreal-base-gpu-cuda9
build-gpu90:
	docker build -t $(CU90_URL):latest -f DockerfileGPUBaseCuda90 .

push-gpu90:
	docker push $(CU90_URL):latest

pull-gpu90:
	docker pull $(CU90_URL):latest

publish-gpu90: build-gpu90 push-gpu90