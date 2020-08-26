import pyglet
from pyglet.window import Window

def f():
    return 1, 2, 3


if __name__ == "__main__":

    a = f()
    a = list(a)
    print(a)
    print(type(a))