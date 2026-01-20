from rq import SimpleWorker, Queue

queue = Queue('first_1111')
worker = SimpleWorker([queue])
worker.work()
