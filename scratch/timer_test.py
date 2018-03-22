import surreal.utils as U
import time

moving_average = U.MovingAverageRecorder()
recorder = U.TimeRecorder()
last_time = time.time()
recorder.start()
for i in range(500000):
    time.sleep(0.001)
    recorder.lap()
    new_time = time.time()
    moving_average.add_value((new_time - last_time))
    last_time = new_time
    if i % 1000 == 0:
        print('Recorder: {}'.format(recorder.avg))
        print('Moving average: {}'.format(moving_average.cur_value()))
