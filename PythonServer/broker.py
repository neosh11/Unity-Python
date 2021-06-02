
import zmq

from protos.transfer_pb2 import Transfer

# Prepare our context and sockets
context = zmq.Context()
frontend = context.socket(zmq.ROUTER)
backend = context.socket(zmq.DEALER)
frontend.bind("tcp://*:5559")
backend.bind("tcp://*:5560")

# Initialize poll set
poller = zmq.Poller()
poller.register(frontend, zmq.POLLIN)
poller.register(backend, zmq.POLLIN)

try:
    # Switch messages between sockets
    while True:
        socks = dict(poller.poll())
        if socks.get(frontend) == zmq.POLLIN:
            message = frontend.recv_multipart()
            # Unpack protobuf?
            transfer_data = Transfer()
            transfer_data.ParseFromString(message)
            
            print(transfer_data.type)
            
            backend.send_multipart(message)
        if socks.get(backend) == zmq.POLLIN:
            message = backend.recv_multipart()
            frontend.send_multipart(message)

except KeyboardInterrupt:
    print("W: interrupt received, stopping...")
finally:
    # clean up
    backend.close()
    frontend.close()
    context.term()
