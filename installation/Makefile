URL='stanfordvl/surreal-cpu'

test:
	docker run -ti --privileged -v `pwd`/:/code $(URL) --py /code/test.py

push:
	docker push $(URL):`cat VERSION`
	docker push $(URL):latest

pull:
	docker pull $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` .
	docker tag $(URL):`cat VERSION` $(URL):latest
