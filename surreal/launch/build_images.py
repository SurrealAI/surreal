"""
    Utilities for building custom docker images
"""
from symphony.addons import DockerBuilder


class SurrealDockerBuilder:
    """
        Manages the following:
        1) Figure out what docker images need to be built
        2) Build (and push) them
    """

    def __init__(self,
                 build_settings,
                 images_requested,
                 tag,
                 push=False):
        """
            Figures out what docker images need to be built
            Populates self.images_provided which computes
                {name: identifier} for all images

        Args:
            build_settings (dict):
                {build_settings_name: {<docker build setting accepted by
                    symphony.addons.DockerBuilder.from_dict>}}
            images_requested (dict):
                {
                    image_name:
                    {
                        repo: image_identifier
                        build_image: build_settings_name
                    },
                    ...
                }
                Images that need to be built.

                If build_image is None, the provided image_identifier
                should be a valid docker image, DockerBuilder
                sets `image_name: image_identifier` in self.images_provided

                If build_image is not None, the provided image_identifier
                should be a valid docker image repo, DockerBuilder
                sets image_name: image_identifier+':'+tag
                in self.images_provided. It will also build the image specified
                by the build_image_field

            tag: specifies a tag for all images
            push: whether to push built images
        """
        self.build_settings = build_settings
        self.images_requested = images_requested
        self.tag = tag

        self.images_to_build = {}
        self.images_provided = {}
        for name, di in images_requested.items():
            identifier = di['identifier']
            build_config = di['build_config']
            if build_config is None:
                self.images_provided[name] = identifier
            else:
                if build_config not in self.images_to_build:
                    self.images_to_build[build_config] = []
                self.images_provided[name] = '{}:{}'.format(identifier, tag)
                self.images_to_build[build_config].append((identifier, tag))
        self.push = push

    def build(self):
        """
            Build all images and push them
        """
        for name, tag_settings in self.images_to_build.items():
            builder = DockerBuilder.from_dict(self.build_settings[name])
            builder.build()
            for repo, tag in tag_settings:
                builder.tag(repo, self.tag)
                if self.push:
                    builder.push(repo, self.tag)
