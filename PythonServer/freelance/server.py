#
# Freelance server - Model 1
# Trivial echo service
#
# Author: Daniel Lundin <dln(at)eintr(dot)org>
#

from os import error
import sys
import zmq

from remove_table import process_dicom, hello

if len(sys.argv) < 2:
    print(f"I: Syntax: {sys.argv[0]} <endpoint>")
    sys.exit(0)

endpoint = sys.argv[1]
context = zmq.Context()
server = context.socket(zmq.REP)
server.bind(endpoint)

hello()

print(f"I: Echo service is ready at {endpoint}")
while True:

    msg = server.recv_multipart()
    
    # TODO everything is is dodgy = needs to be fixed later on.
    # However - just to test.
    # Assume that the imput that is received is the location of the dicom files.
    # We will now process the dicom files.
    # By runnning a python process

    print("hello")

    if(msg):
        print(msg)
        print(type(msg))
        responses = []
        for x in msg:
            print(x)
            inputDicomLocation = x.decode("ascii")
            try:
                process_dicom(inputDicomLocation,
                              "C:\\Users\\SingularHealthUnit\\Desktop\\OMG")
            except error:
                print(error)

    # Check if the location exists
    # and run a simple python program.

    # print()

    if not msg:
        break  # Interrupted
    server.send_multipart(["C:\\Users\\SingularHealthUnit\\Desktop\\OMG".encode("ascii")])

server.setsockopt(zmq.LINGER, 0)  # Terminate immediately
