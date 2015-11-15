"""
In this example the client starts the server and sends things to it.

import zmq
import sys


def server():
    print('[server] Running zeromq server')
    # ZeroMQ Context
    context = zmq.Context()

    # Define the socket using the "Context"
    sock = context.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5678")

    # Run a simple "Echo" server
    while True:
        message = sock.recv()
        sock.send("Echo: " + message)
        print("[server] Echo: " + message)


def client():
    print('[client] Running zeromq client')
    # ZeroMQ Context
    context = zmq.Context()

    # Define the socket using the "Context"
    sock = context.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:5678")

    # Send a "message" using the socket
    sock.send("message is hello world")
    responce = sock.recv()
    print('[client] response=%s' % (responce,))


if __name__ == '__main__':
    if sys.argv[-1] == 'server':
        server()
    if sys.argv[-1] == 'client':
        client()
    if sys.argv[-1] == 'both':
        import utool as ut
        proc = ut.spawn_background_process(server)
        client()
        client()
        client()
        proc.terminate()


Simple task farm, with routed replies in pyzmq
For http://stackoverflow.com/questions/7809200/implementing-task-farm-messaging-pattern-with-zeromq
Note that things are run in threads to keep stuff in one file, there is no
reason they need to be.
License: Public Domain
"""
#import os
import utool as ut
import time
import random
import zmq
import threading


# BASICALLY DO A CLIENT/SERVER TO SPAWN PROCESSES
# AND THEN A PUBLISH SUBSCRIBE TO RETURN DATA


ctx = zmq.Context.instance()
client_iface = "tcp://127.0.0.1:5555"
engine_iface = "tcp://127.0.0.1:5556"

collect_pull_iface = "tcp://127.0.0.1:5557"


def result_collector():
    collector = ctx.socket(zmq.PULL)
    collector.bind(collect_pull_iface)
    collector.setsockopt(zmq.IDENTITY, 'Controller.COLLECTOR')
    collecter_data = {}
    if False:
        poller = zmq.Poller()
        poller.register(collector, zmq.POLLIN)
        while True:
            print('polling')
            evnts = poller.poll(10)
            evnts = dict(poller.poll())
            if collector in evnts:
                print('Collecting...')
                reply_result = collector.recv_json()
                print('COLLECT Controller.COLLECTOR: Collected: %r' % (reply_result,))
                result_id = reply_result['result_id']
                collecter_data[result_id] = reply_result
            else:
                print('waiting')
                pass
    else:
        collector.RCVTIMEO = 1000
        while True:
            print('Collecting...')
            try:
                reply_result = collector.recv_json()
                print('COLLECT Controller.COLLECTOR: Collected: %r' % (reply_result,))
                result_id = reply_result['result_id']
                collecter_data[result_id] = reply_result
            except Exception:
                # TODO: add in a responce if a job status request is given
                print('loop')
                pass

        #result = collector.recv_json()
        #if collecter_data.has_key(result['consumer']):  # NOQA
        #    collecter_data[result['consumer']] = collecter_data[result['consumer']] + 1
        #else:
        #    collecter_data[result['consumer']] = 1
        #if x == 999:
        #    print(collecter_data)


def scheduler():
    """
    IBEIS:
        THis will belong to a thread on the webserver main process.

    ROUTER-DEALER queue device, for load-balancing requests from clients
    across engines, and routing replies to the originating client."""
    # ----
    router = ctx.socket(zmq.ROUTER)
    router.bind(client_iface)
    # ----
    dealer = ctx.socket(zmq.DEALER)
    # this is optional, it just makes identities more obvious when they appear
    dealer.setsockopt(zmq.IDENTITY, 'Controller.DEALER')
    dealer.bind(engine_iface)
    # the remainder of this function can be entirely replaced with
    if False:
        zmq.device(zmq.QUEUE, router, dealer)
    else:
        # but this shows what is really going on:
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        poller.register(dealer, zmq.POLLIN)
        while True:
            evts = dict(poller.poll())
            # poll() returns a list of tuples [(socket, evt), (socket, evt)]
            # dict(poll()) turns this into {socket:evt, socket:evt}
            if router in evts:
                msg = router.recv_multipart()
                # ROUTER sockets prepend the identity of the sender, for routing replies
                client = msg[0]   # NOQA
                print("Controller.ROUTER received %s, relaying via DEALER" % msg)
                dealer.send_multipart(msg)
            if dealer in evts:
                msg = dealer.recv_multipart()
                client = msg[0]  # NOQA
                print("Controller.DEALER received %s, relaying via ROUTER" % msg)
                router.send_multipart(msg)


def process_request(msg):
    """process the message (reverse letters)"""
    #return ['foobar']/en
    import time
    time.sleep(.1)
    return [ part[::-1] for part in msg ]


def engine(id):
    """
    IBEIS:
        This will be part of a worker process with its own IBEISController
        instance.

        Needs to send where the results will go and then publish the results there.


    The engine - receives messages, performs some action, and sends a reply,
    preserving the leading two message parts as routing identities
    """
    engine_sock = ctx.socket(zmq.ROUTER)
    engine_sock.connect(engine_iface)

    push_sock = ctx.socket(zmq.PUSH)
    push_sock.connect(collect_pull_iface)
    while True:
        msg = engine_sock.recv_multipart()
        print("engine %s recvd message:" % id, msg)
        # note that the first two parts will be ['Controller.ROUTER', 'Client.<id>']
        # these are needed for the reply to propagate up to the right client
        idents, request = msg[:2], msg[2:]
        result_id = 'result_%s' % (id,)
        future_msg = 'Job started. Check for status at ' + str(result_id)
        reply_future = idents + [future_msg, result_id]
        print("engine %s sending reply:" % id, reply_future)
        engine_sock.send_multipart(reply_future)
        reply_result = dict(idents=idents, result=process_request(request), result_id=result_id)
        #print('ENGINE %s computed results, but has nowhere to send it!' % (id,))
        print('ENGINE %s computed results, pushing it to collector' % (id,))
        push_sock.send_json(reply_result)
        print('...pushed')


def client(id, n):
    """
    IBEIS:
        This is just a function that lives in the main thread and ships off a
        job.

    The client - sends messages, and receives replies after they
    have been processed by the

    """
    s = ctx.socket(zmq.DEALER)
    s.identity = "Client.%s" % id
    s.connect(client_iface)
    for i in range(n):
        print('')
        msg = ["hello", "world", str(random.randint(10, 100))]
        print("client %s sending :" % id, msg)
        s.send_multipart(msg)
        msg = s.recv_multipart()
        print("client %s received:" % id, msg)


class BackgroundJobQueue(object):
    def __init__(self, id):
        self.id = id
        self.initialize_background_processes()
        self.initialize_main_thread()

    def initialize_background_processes(self):
        print('Initialize Background Processes')
        #spawner = ut.spawn_background_process
        spawner = ut.spawn_background_daemon_thread
        self.st = spawner(scheduler)
        self.ct = spawner(result_collector)
        self.engines = [spawner(engine, i) for i in range(2)]

    def initialize_main_thread(self):
        s = ctx.socket(zmq.DEALER)
        s.identity = "Client.%s" % id
        s.connect(client_iface)
        self.s = s

    def queue_job(self):
        msg = ["hello", "world", str(random.randint(10, 100))]
        print('\n+-------')
        print("client %s sending :" % id, msg)
        s = self.s
        s.send_multipart(msg)
        msg = s.recv_multipart()
        print("client %s received:" % id, msg)
        print('\nL______')

    def get_job_status(self, job_id):
        pass


def _on_ctrl_c(signal, frame):
    print('[ibeis.main_module] Caught ctrl+c')
    print('[ibeis.main_module] sys.exit(0)')
    import sys
    sys.exit(0)


def _init_signals():
    import signal
    signal.signal(signal.SIGINT, _on_ctrl_c)


def main():
    import utool as ut  # NOQA
    _init_signals()
    # now start a few clients, and fire off some requests
    #clients = []
    time.sleep(1)

    print('Initializing Main Thread')
    sender = BackgroundJobQueue(1)
    print('... waiting for jobs')
    if False:
        #ut.embed()
        sender.queue_job()
    else:
        time.sleep(.5)
        sender.queue_job()
        #time.sleep(1.5)
        #sender.queue_job()
        time.sleep(1.5)
        time.sleep(1.5)

    if False:
        clients = [threading.Thread(target=client, args=(i, 2)) for i in range(4)]
        for t in clients:
            t.start()
            # remove this t.join() to allow clients to be run concurrently.
            # this will work just fine, but the print-statements will
            # be harder to follow
            while True:
                t.join(600)
                if not t.is_alive():
                    break


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    main()
    #while True:
    #    if not t.is_alive():
    #        break
    #t.join()

"""
python ibeis/zmq_stuff.py
python zmq_stuff.py both
"""
