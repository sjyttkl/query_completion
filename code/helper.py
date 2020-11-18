import bunch
import hashlib
import json
import os


def GetPrefixLen(user, query, n=None):
    # choose a random prefix length
    hasher = hashlib.md5()
    print(hasher.update(user.encode("utf8")))
    print(hasher.update(''.join(query).encode("utf8")))
    if n:
        hasher.update(str(n).encode("utf8"))
    prefix_len = int(hasher.hexdigest(), 16) % (len(query) - 1)
    prefix_len += 1  # always have at least a single character prefix
    return prefix_len


def GetParams(filename, mode, expdir):
    param_filename = os.path.join(expdir, 'params.json')
    if mode == 'train':
        with open(filename, 'r') as f:
            param_dict = json.load(f)
            params = bunch.Bunch(param_dict)
        with open(param_filename, 'w') as f:
            json.dump(param_dict, f)
    else:
        with open(param_filename, 'r') as f:
            params = bunch.Bunch(json.load(f))
    return params


if __name__ == "__main__":
    query = "娃娃上衣"
    prefix_len = GetPrefixLen("s10101", query, 2)
    prefix = query[:prefix_len + 1]  # 前缀

    print(prefix_len, "  ", prefix)
