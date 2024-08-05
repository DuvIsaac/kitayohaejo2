

class PriorityQueue:

    '''Implements a bounded capacity queue. You can insert items of the form (k, p) where k is an integer between 0 and size - 1 and p is the priority of k. '''
    
    def __init__(self, size = 5):
        self.index = [None for _ in range(size)]
        self.heap = [0]
        self.priorities = [- float('inf') for _ in range(size)]
    
    def length(self):
        return self.heap[0]
    
    def swap(self, i, j):
        u = self.heap[i]
        v = self.heap[j]
        self.heap[i], self.heap[j] = v, u
        self.index[u], self.index[v] = self.index[v], self.index[u]
    
    def priority(self, u):
        return self.priorities[u]

    def prio(self, i):
        return self.priorities[self.heap[i]]
    
    def push(self, x, key):
        self.heap.append(x)
        self.heap[0] += 1
        n = self.length()
        self.index[x] = n
        self.priorities[x] = key

        while n > 1 and self.prio(n // 2) < self.prio(n):
            self.swap(n, n // 2)
            n = n // 2
        
    def percolate(self, i):
        k = i
        if 2 * i + 1 < self.length():
            if self.prio(2 * i) > self.prio(i):
                if self.prio(2 * i + 1) > self.prio(i):
                    if self.prio(2 * i) < self.prio(2 * i + 1):
                        k = 2 * i + 1
                    else:
                        k = 2 * i
                else:
                    k = 2 * i
            elif self.prio(2 * i + 1) > self.prio(i):
                    k = 2 * i + 1
        elif 2 * i < self.length():
            if self.prio(2 * i) > self.prio(i):
                k = 2 * i
        if k != i:
            self.swap(i, k)
            self.percolate(k)
    
    def pop(self):
        self.swap(1, self.length())
        u = self.heap.pop()
        self.heap[0] -= 1
        key = self.priorities[u]
        self.index[u] = None
        self.percolate(1)
        return u, key
    
    def changePrio(self, u, newKey):
        if self.index[u] == None:
            self.push(u, newKey)
        else:
            self.priorities[u] = newKey
            self.percolate(self.index[u])
    
    



                    

        

