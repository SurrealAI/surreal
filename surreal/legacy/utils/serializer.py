"""
Serializes numpy and JSON-like objects
"""
import pickle
import base64
import hashlib


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
    return s.replace('/','_')


def np_serialize(obj):
    """
    We can improve this function if we *really* need more memory efficiency
    """
    return pickle.dumps(obj)


def np_deserialize(binary):
    """
    We can improve this function if we *really* need more memory efficiency
    """
    return pickle.loads(binary)


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
