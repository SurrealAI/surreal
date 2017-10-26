"""
Defines packs that will be sent to redis
"""
class Pack:
    def serialize(self):
        raise NotImplementedError

    @classmethod
    def deserialize(cls, binary):
        """
        @Returns:
            the pack object created from the binary blob
        """
        raise NotImplementedError

    def get_key(self):
        """
        Key of this pack to be stored in Redis. None if this pack is not stored.
        """
        return None

    def get_data(self):
        raise NotImplementedError

    def jsonify(self):
        # TODO
        pass
