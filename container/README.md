# Docker images

All images have two versions: base and full. Base images contain third-party dependencies that almost never change. Full image builds upon base image and includes libraries from the Surreal stack, which are subject to frequent code updates. The actual pods will run the full image.

Base image dockerfile names are camelCase and always end with `Base`. Base docker image names are hyphenated lowercase and always end with `-base`. For brevity, we only refer to the full images. Below are the descriptions for `<Dockerfile> (<gcloud-docker-image-url>)`:


- `DockerfileCPU (us.gcr.io/jimfan2018-208323/surreal-base-cpu` ??? @Jiren):
- `DockerfileGPU` ... DEPRECATED?? @Jiren
- `DockerfileMujocopyGPU`:
- `DockerfileLearnToRunCPU (us.gcr.io/jimfan2018-208323/learn-to-run-cpu)`: NIPS 2018 learning to run challenge
- `DockerfileLearnToRunGPU (us.gcr.io/jimfan2018-208323/learn-to-run-gpu)`: NIPS 2018 learning to run challenge
