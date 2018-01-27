URL='stanfordvl/surreal-cpu'

test:
	docker run -ti --privileged -v `pwd`/:/code $(URL) --cmd 'python /code/test.py'

push:
	docker push $(URL):`cat VERSION`
	docker push $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` .
	docker tag $(URL):`cat VERSION` $(URL):latest
