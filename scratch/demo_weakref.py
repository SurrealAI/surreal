import weakref


D = weakref.WeakValueDictionary()

class MyObj:
    pass

obj = MyObj()
D['shti'] = obj
A = [3, 4, obj]
del obj

print(D['shti'])
o = D['shti']

A.pop()
del o
print(D['shti'])

