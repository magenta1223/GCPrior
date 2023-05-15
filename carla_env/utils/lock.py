import time
from threading import Lock, Thread


def lock_release_after(lock: Lock, seconds: float):
    def release():
        time.sleep(seconds)
        lock.release()

    Thread(target=release).start()
