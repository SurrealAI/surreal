# Installing Surreal
[Install and Run SURREAL Locally](#install-and-run-surreal-locally)  
[Develop Algorithms Locally](#develop-algorithms-locally)  
[Run Surreal on Google Cloud](#run-surreal-on-google-cloud)  
[Setup Custom Docker Build Process](#setup-custom-docker-build-process)  
[Develop Algorithms on Kubernets](#develop-algorithms-on-kubernetes)  
[Develop Surreal on Kubernets](#develop-surreal-on-kubernetes)  
[Reporting Issues](#reporting-issues)  
[Contributing](#contributing)  

---


## Develop Algorithms on Kubernetes
!! This section is under construction, surrealers please move on to the next section
Note: If you want to change the Surreal library, please refer to [Develop Surreal on Kubernetes](#developing-surreal-on-kubernetes).

To develop your own RL library using surreal. You need to do two things:
* First, you need to setup the build process which builds an image containing your own library.
* Second, you need to tell kurreal how to find your executable.

## Develop Surreal on Kubernetes
Note: If you are using Surreal but not changing the Surreal library itself, please refer to [Develop Algorithms on Kubernetes](#developing-algorithms-on-kubernetes).

We provide launch settings and docker build settings that build an image with your custom Surreal Library. To use them, you need to first setup the following fields (`<enclosed by brackets>`) in `~/.surreal.yml`.

```yaml
# ~/.surreal.yml
creation_settings:
  contrib:
    mode: basic
    ...
    agent:
      image: <my-registry>/<repo-name>
      build_image: contrib-image
      ...
    nonagent:
      image: <my-registry>/<repo-name>
      build_image: contrib-image
      ...

docker_build_settings:
  - name: contrib-image
    temp_directory: <~/symph_temp/contrib or anywhere you want image build to happen>
    verbose: true
    dockerfile: <path to your surreal fork>/docker/Dockerfile-contribute
    context_directories:
      - name: surreal
        path: <path to your surreal fork>
        force_update: true
```
For reference, here are the contents in `Dockerfile-contribute`
```
FROM 
# ~/surreal/docker/Dockerfile-contribute
COPY surreal /mylibs/surreal-dev
RUN pip install -e -U /mylibs/surreal-dev
```

Now `kurreal` will automatically build your image, push it and refer to them properly in launched experiments.
```bash
kurreal create contrib ...
```

## Reporting Issues

## Contributing

