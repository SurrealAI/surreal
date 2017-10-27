"""
http://asyncio.readthedocs.io/en/latest/producer_consumer.html
https://stackoverflow.com/questions/35127520/asyncio-queue-consumer-coroutine
"""
import asyncio
import random

q = asyncio.Queue()

async def producer(num):
    while True:
        await q.put(num + random.random())
        await asyncio.sleep(random.random())

async def consumer(num):
    while True:
        value = await q.get()
        print('Consumed', num, value)

loop = asyncio.get_event_loop()

for i in range(6):
    loop.create_task(producer(i))

for i in range(3):
    loop.create_task(consumer(i))

loop.run_forever()