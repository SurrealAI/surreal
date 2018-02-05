URL='stanfordvl/surreal-cpu'

test:
	docker run -ti --privileged -v `pwd`/:/code -v $(HOME)/.mujoco:/mujoco $(URL) --cmd python /code/test_install.py "a b" 'c d e' f

push:
	docker push $(URL):`cat VERSION`
	docker push $(URL):latest

pull:
	docker pull $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` .
	docker tag $(URL):`cat VERSION` $(URL):latest

# Can't do as container requires gpu hosts
# test-gpu:

GPU_URL='stanfordvl/surreal-cpu'

push-gpu:
	docker push $(GPU_URL):`cat VERSION`
	docker push $(GPU_URL):latest

pull:
	docker pull $(GPU_URL):latest

build:
	docker build -t $(GPU_URL):`cat VERSION` .
	docker tag $(GPU_URL):`cat VERSION` $(GPU_URL):latest
