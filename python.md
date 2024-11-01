try catch syntax:

try:
    blah blah blah
except OSError as e: # catches OSError specifically and its subclasses
    teehee yaya
except Exception as e: # all Exception subclasses aka most exceptions like FileNotFound
    wawawaaa

"as e" captures specific error object and u can do stuff with them
print(e) calls prebuilt `__str__()` func (basically toString())