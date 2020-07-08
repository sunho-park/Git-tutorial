import traceback
def raise_error():
    raise RuntimeError('something bad happened!')

def do_something_that_might_error():
    raise_error()
try:
    do_something_that_might_error()
except Exception as error:
    traceback.print_exc()