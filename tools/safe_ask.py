import sys


def safe_ask(q):
    try:
        answer = q.ask()
        if answer is None:
            print("\nCancelled by user.")
            sys.exit(0)
        return answer
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    q = input('q')
    safe_ask(q)