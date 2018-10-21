FROM surrealai/surreal-nvidia:v0.0
# ~/surreal/docker/Dockerfile-contribute
COPY surreal /mylibs/surreal-dev
RUN pip uninstall -y surreal
RUN pip install -e /mylibs/surreal-dev
