import time
import zmq
import argparse

def zmq_proxy():
    context = zmq.Context()
    sock_recv = context.socket(zmq.PULL)
    sock_recv.connect(zmq_recv_addr)
    sock_snd1 = context.socket(zmq.PUSH)
    sock_snd1.bind(zmq_send1_addr)
    sock_snd1.setsockopt(zmq.SNDTIMEO, 1000)
    sock_snd2 = context.socket(zmq.PUSH)
    sock_snd2.bind(zmq_send2_addr)
    sock_snd2.setsockopt(zmq.SNDTIMEO, 1000)

    time.sleep(1)
    print(f"0MQ Proxy is started successfully.")

    nmsg = 0
    while(True):
        msg = sock_recv.recv()
        nmsg += 1
        #print(f"Processed message #{nmsg}")
        try:
            sock_snd1.send(msg, zmq.NOBLOCK)
        except Exception as ex:
            pass
        try:
            sock_snd2.send(msg, zmq.NOBLOCK)
        except Exception as ex:
            pass

if __name__ == "__main__":

    def formatter(prog):
        # Set maximum width such that printed help mostly fits in the RTD theme code block (documentation).
        return argparse.RawDescriptionHelpFormatter(prog, max_help_position=20, width=90)

    parser = argparse.ArgumentParser(
        description="0MQ Proxy: Connects to a single PUSH socket and republishes all received messages.\n"
        "to two different PUSH sockets. The proxy is used to distribute messages to multiple consumers.\n",
        formatter_class=formatter,
    )

    parser.add_argument(
        "--zmq-recv-addr",
        dest="zmq_recv_addr",
        action="store",
        default=None,
        help="Address of the remote PUSH socket, e.g. 'tcp://127.0.0.1:5557'.",
    )

    parser.add_argument(
        "--zmq-send-addr1",
        dest="zmq_send_addr1",
        action="store",
        default=None,
        help="Address of the first PUSH socket, e.g. 'tcp://127.0.0.1:5558'.",
    )

    parser.add_argument(
        "--zmq-send-addr2",
        dest="zmq_send_addr2",
        action="store",
        default=None,
        help="Address of the first PUSH socket, e.g. 'tcp://127.0.0.1:5559'.",
    )

    args = parser.parse_args()

    try:
        if args.zmq_recv_addr is None:
            raise ValueError("The address of the remote PUSH socket must be provided.")
        if args.zmq_send_addr1 is None:
            raise ValueError("The address of the first PUSH socket must be provided.")
        if args.zmq_send_addr2 is None:
            raise ValueError("The address of the second PUSH socket must be provided.")

        zmq_recv_addr = args.zmq_recv_addr
        zmq_send1_addr = args.zmq_send_addr1
        zmq_send2_addr = args.zmq_send_addr2

        zmq_proxy()

    except ValueError as ex:
        print(f"Error: {ex}")
