"""
This is just a general file for seeing how Python handles certain things, especially with regards to inheritance.

This is mostly just for understanding the Stable Baselines 3 codebase... it does some things that I'm not quite used to.

The result/ realization!

So here was the line of code I was trying to understand:
https://github.com/DLR-RM/stable-baselines3/blob/646d6d38b6ba9aac612d4431176493a465ac4758/stable_baselines3/common/vec_env/vec_video_recorder.py#L76

You can see below what I was trying to figure out; what I didn't look at close enough was how VideoRecorder was being
imported!
What was actually going on was as follows:

# This is what I saw which looked confusing
self.video_recorder = None
self.video_recorder = video_recorder.VideoRecorder()

# Its this!
from package.filename import filename
self.variable = None
self.variable = filename.VideoRecorder()
# The file name is the same as the variable name which caused the confusion.

"""


class DummyMethod:
    def __init__(self, name="yolo"):
        self.name = name

    def print_name(self):
        print(self.name)


class DummyClass:
    def __init__(self, name="honda"):
        self.name = name
        self.var = None

    def testing_dummy_method(self):
        self.var = var.DummyMethod()


def main():
    dummy_class = DummyClass()
    dummy_class.testing_dummy_method()
    dummy_class.var.print_name()


if __name__ == "__main__":
    main()
