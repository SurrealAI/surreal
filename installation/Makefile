URL='stanfordvl/surreal-cpu'

test:
	docker run -ti --privileged -v `pwd`/:/code $(URL) --cmd python /code/test_entry.py "a b" 'c d e' f
	docker run -ti --privileged -v `pwd`/:/code -v $(HOME)/.mujoco:/mujoco $(URL) --py /code/test_libs.py

push:
	docker push $(URL):`cat VERSION`
	docker push $(URL):latest

pull:
	docker pull $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` .
	docker tag $(URL):`cat VERSION` $(URL):latest
