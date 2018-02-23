from surreal.session.tracker import *
import surreal.utils


pt = PeriodicTensorplex(
    tensorplex=None,
    period=5,
    is_average=1,
    keep_full_history=1
)


for i in range(20):
    if i < 10:
        print(pt.update({
            'lr': i,
            'yo': i/10,
            'mini': i/100
        }))
    else:
        print(pt.update({
            'lr': i,
            'yo': i/10,
        }))

U.print_(pt.get_history())
