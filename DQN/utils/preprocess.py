import numpy as np

def preprocess(image, constant):
    image = image[34:194, :, :]
    image = np.mean(image, axis=2, keepdims=False)
    image = image[::2, ::2]
    image = image/256
    # remove background
    image = image - constant/256
    return image

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.vals = [0 for _ in range(2*capacity - 1)]

    def retrive(self, num):
        ind = 0
        # not a leaf
        while ind < self.capacity-1:
            left = 2*ind + 1
            right = left + 1
            if num > self.vals[left]:
                num -= self.vals[left]
                ind = right
            else:
                ind = left
        return ind - self.capacity + 1

    def update(self, delta, ind):
        ind += self.capacity - 1
        while True:
            self.vals[ind] += delta
            if ind == 0:
                break
            ind -= 1
            ind //= 2

