from threading import Thread
import zmq
import pyarrow as pa
import os
import uuid


memory_usage = {}

class SharedMemoryObject(object):
    def __init__(self, filename):
        print('Shared memory with name {} created'.format(filename))
        assert not filename in memory_usage or not memory_usage[filename]
        memory_usage[filename] = True
        self.filename = filename
        self.file = pa.memory_map(filename)
        self.buffer = self.file.read_buffer()
        self.data = pa.deserialize(self.buffer)
        self.deleted = False

    def delete(self):
        if not self.deleted:
            assert memory_usage[filename]
            memory_usage[filename] = False
            print('Shared memory with name {} deleted'.format(filename))
            print('Memory entries: {}'.format(memory_usage))
            self.file.close()
            os.remove(self.filename)
            self.deleted = True

    def __del__(self):
        self.delete()

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
        name = str(uuid.uuid4())
    if isinstance(data, pa.lib.Buffer):
        buf = data
    else:
        buf = pa.serialize(data).to_buffer()
    with pa.MemoryMappedFile.create(name, buf.size) as f:
        file.write(buf)
    return name.encode()

def inmem_deserialize(name_bin):
    """
        Deserailize data sent by inmem_serialize by
        putting it inside a SharedMemoryObject.data
    """
    return SharedMemoryObject(name.decode())