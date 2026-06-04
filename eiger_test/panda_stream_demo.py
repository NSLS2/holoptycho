import zmq
ZMQ_ADDRESS = "tcp://10.66.19.45:6666"

def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ZMQ_ADDRESS)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        message_dict = socket.recv_json()
        print(message_dict)
        print("\n")

if __name__ == "__main__":
    main()
