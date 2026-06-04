import threading

class ReadLockContext:
    def __init__(self, rwlock):
        self.rwlock = rwlock
    def __enter__(self):
        self.rwlock.acquire_read()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rwlock.release_read()

class WriteLockContext:
    def __init__(self, rwlock):
        self.rwlock = rwlock
    def __enter__(self):
        self.rwlock.acquire_write()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rwlock.release_write()

class RWLock:
    """
    A simple Read-Write Lock with writer priority to prevent writer starvation.
    Allows concurrent reads, but exclusive writes.
    """
    def __init__(self):
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._write_requests = 0
        
    def acquire_read(self):
        with self._condition:
            while self._writers > 0 or self._write_requests > 0:
                self._condition.wait()
            self._readers += 1
            
    def release_read(self):
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()
                
    def acquire_write(self):
        with self._condition:
            self._write_requests += 1
            while self._readers > 0 or self._writers > 0:
                self._condition.wait()
            self._write_requests -= 1
            self._writers += 1
                
    def release_write(self):
        with self._condition:
            self._writers -= 1
            self._condition.notify_all()
            
    def read_lock(self):
        return ReadLockContext(self)
        
    def write_lock(self):
        return WriteLockContext(self)
