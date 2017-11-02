from scratch.utils import *


t = PeriodicTracker(7, 0, 7)
for i in range(20):
    print(t.track_increment(3), t.value, t._endpoint)