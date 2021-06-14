import zmq
import signal
import time


#  Prepare our context and sockets
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5559")


try:
    #  Do 10 requests, waiting each time for a response
    for request in range(1, 11):
        try:
            socket.send(b"Hello")
            message = socket.recv()
            print("Received reply %s [%s]" % (request, message))
        except Exception as e:
            print(e.__class__.__name__)


except KeyboardInterrupt:
    print("W: interrupt received, stopping...")

finally:
    socket.close()
    context.term()
socket.close()
context.term()