import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect("tcp://localhost:5560")

try:
    while True:
        rec = socket.recv()
        print("Received Request with Message: " + rec.decode('utf-8'))
        socket.send_string("abc")
except KeyboardInterrupt:
    print("W: interrupt received, stopping...")
finally:
    # clean up
    socket.close()
    context.term()
