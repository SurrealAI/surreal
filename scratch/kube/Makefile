URL='stanfordvl/surreal-cpu'
options=-ti --network=host -h 127.0.0.1 -p 8001 --privileged -v `pwd`/:/code -v `pwd`/../:/mylibs/surreal  -v `pwd`/../../Tensorplex:/mylibs/tensorplex $(URL)

s:
	docker run $(options) --py /code/run_zmq_server.py

c:
	docker run $(options) --py /code/run_zmq_client.py

push:
	docker push $(URL):`cat VERSION`
	docker push $(URL):latest

build:
	docker build -t $(URL):`cat VERSION` .
	docker tag $(URL):`cat VERSION` $(URL):latest
