import zipfile, io, collections, random
Z = r"C:\Users\Divya\OneDrive\Documents\amazon ml\student_resource.zip"
z = zipfile.ZipFile(Z); pre = "student_resource/dataset/train/"

def rows(name):
    with z.open(pre + name) as f:
        t = io.TextIOWrapper(f, encoding="utf-8", newline=""); next(t)
        for line in t:
            yield line.rstrip("\r\n").split("\t")

owner = {}; multi = 0; s1_ids = []
for a, b in rows("train_ground_truth.tsv"):
    s1_ids.append(a)
    for m in filter(None, b.split(",")):
        if m in owner: multi += 1
        else: owner[m] = a
print(f"distinct matched S2/S3 ids={len(owner)}  ids claimed by >1 S1={multi}")

# country agreement + sample of examples
s1 = {r[0]: r for r in rows("train_source1.tsv")}
agree = collections.Counter(); per_src_per_s1 = collections.Counter()
examples = []
random.seed(0)
want = set(random.sample(s1_ids, 6))
cluster = collections.defaultdict(list)
for src in ("train_source2.tsv", "train_source3.tsv"):
    for r in rows(src):
        o = owner.get(r[0])
        if o is None: continue
        agree[s1[o][3] == r[3]] += 1
        if o in want: cluster[o].append(r)
print("country agreement S1 vs matched:", dict(agree))
for o in want:
    print("\n##", " | ".join(s1[o][1:]))
    for r in cluster[o]:
        print("   ", r[0], " | ".join(r[1:]))
