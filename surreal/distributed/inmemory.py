from threading import Thread
import zmq
import pyarrow as pa
import os
import uuid

TEMP_FOLDER = '/tmp/surreal'
os.makedirs(TEMP_FOLDER, exist_ok=True)
memory_usage = {}

class SharedMemoryObject(object):
    def __init__(self, filename):
        # print('Shared memory with name {} created'.format(filename))
        # assert not filename in memory_usage or not memory_usage[filename]
        # memory_usage[filename] = True
        self.filename = filename
        self.file = pa.memory_map(filename)
        self.buffer = self.file.read_buffer()
        self.data = pa.deserialize(self.buffer)
        self.deleted = False

    def delete(self):
        if not self.deleted:
            # assert memory_usage[self.filename]
            # del memory_usage[self.filename]
            # print('Shared memory with name {} deleted'.format(self.filename))
            # print('Memory entries: {}'.format(len(memory_usage)))
            self.file.close()
            # print('Deleting file: {}'.format(self.filename))
            os.remove(self.filename)
            self.deleted = True

    def __del__(self):
        self.delete()

def inmem_dump(data, name=None):
    """
        Dump data to a memory mapped file, return filename
    Args:
        @data: bytes data
        @name (Optional):
    """
    if name is None:
        name = os.path.join(TEMP_FOLDER, str(uuid.uuid4()))
    with pa.MemoryMappedFile.create(name, len(data)) as f:
        f.write(data)
    return name.encode()

def inmem_serialize(data, name=None):
    """
        Serialize data into pyarrow format,
        Save to a memory mapped file, return filename
    Args:
        @data: python object to be serialized. 
               At least supports native types, dict, list and numpy.array
               If data is pyarrow.lib.Buffer, saves directly
        @name (Optional):
    """
    if name is None:
        name = os.path.join(TEMP_FOLDER, str(uuid.uuid4()))
    buf = pa.serialize(data).to_buffer()
    with pa.MemoryMappedFile.create(name, buf.size) as f:
        f.write(buf)
    return name.encode()

def inmem_deserialize(name_bin):
    """
        Deserailize data sent by inmem_serialize by
        putting it inside a SharedMemoryObject.data
    """
    return SharedMemoryObject(name_bin.decode())