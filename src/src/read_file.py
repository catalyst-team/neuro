with open("./src/label_protocol.txt","r") as f:
    t = f.read()
print([int(x) for x in t.split(",")])