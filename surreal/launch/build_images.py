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
                {name: repo:tag} for all images

        Args:
            build_settings (dict):
                {name: {<docker build setting accepted by
                    symphony.addons.DockerBuilder.from_dict>}}
            images_requested (dict): {name: repo}
                Images needed. If repo has no ":", look for entry
                in build_settings with key @name, build it under
                repo:tag and push
            tag: specifies a tag for all images
            push: whether to push built images
        """
        self.build_settings = build_settings
        self.images_requested = images_requested
        self.tag = tag

        self.images_provided = {}
        self.images_to_build = []
        for key, repo in images_requested:
            if ":" in repo:
                self.images_provided[key] = repo
            else:
                self.images_to_build.append(key)
                self.images_provided[key] = '{}:{}'.format(repo, tag)

    def build(self):
        """
            Build all images and push them
        """
        for name, repo in self.images_to_build.items():
            builder = DockerBuilder.from_dict(self.build_settings[name])
            builder.build()
            builder.tag(repo, self.tag)
            builder.push(repo, self.tag)
