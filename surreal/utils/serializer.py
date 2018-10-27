"""
Serializes numpy and JSON-like objects
"""
import pickle
import base64
import hashlib
import json
import pyarrow as pa


def pa_serialize(obj):
    return pa.serialize(obj).to_buffer()


def pa_deserialize(binary):
    return pa.deserialize(binary)


_SERIALIZER = pa_serialize
_DESERIALIZER = pa_deserialize

# _SERIALIZER = pickle.dumps
# _DESERIALIZER = pickle.loads


def set_global_serializer(serializer, deserializer):
    """
    Call at the start of a script
    """
    assert callable(serializer) and callable(deserializer)
    global _SERIALIZER, _DESERIALIZER
    _SERIALIZER = serializer
    _DESERIALIZER = deserializer


def serialize(obj):
    """
    We can improve this function if we *really* need more memory efficiency
    """
    return _SERIALIZER(obj)


def deserialize(binary):
    """
    We can improve this function if we *really* need more memory efficiency
    """
    return _DESERIALIZER(binary)


def string_hash(s):
    assert isinstance(s, str)
    return binary_hash(s.encode('utf-8'))


def binary_hash(binary):
    """
    Low collision hash of any binary string
    For designating the 16-char object key in Redis.
    Runs at 200 mu-second per hash on Macbook pro.
    Only contains characters from [a-z][A-Z]+_
    """
    s = hashlib.md5(binary).digest()
    s = base64.b64encode(s)[:16]
    s = s.decode('utf-8')
    # return s.replace('/','_')
    return s


def pyobj_hash(obj):
    return binary_hash(serialize(obj))


def bytes2str(bytestring):
    if isinstance(bytestring, str):
        return bytestring
    else:
        return bytestring.decode('UTF-8')


def str2bytes(string):
    if isinstance(string, bytes):
        return string
    else:
        return string.encode('UTF-8')
