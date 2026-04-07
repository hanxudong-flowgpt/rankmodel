import pickle

with open("./signature_def", "r") as f:
    sig_def = pickle.load(f)
    print sig_def

    for k, v in sig_def.inputs.items():
        new_k = "examples"
        sig_def.inputs[new_k].CopyFrom(v)
        del sig_def.inputs[k]
        break

    for k, v in sig_def.outputs.items():
        new_k = "output"
        sig_def.outputs[new_k].CopyFrom(v)
        del sig_def.outputs[k]
        break

    print sig_def

with open("./pns_signature_def", "w+") as f:
    pickle.dump(sig_def, f)
