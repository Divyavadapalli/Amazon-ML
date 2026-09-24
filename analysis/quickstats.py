import zipfile, io, collections, sys
Z = r"C:\Users\Divya\OneDrive\Documents\amazon ml\student_resource.zip"
z = zipfile.ZipFile(Z)
pre = "student_resource/dataset/"
for split in ("train", "test"):
    for s in (1, 2, 3):
        name = f"{pre}{split}/{split}_source{s}.tsv"
        n = 0; bad = 0; ctry = collections.Counter(); empty_addr = 0; empty_name = 0
        with z.open(name) as f:
            t = io.TextIOWrapper(f, encoding="utf-8", newline="")
            next(t)
            for line in t:
                parts = line.rstrip("\r\n").split("\t")
                n += 1
                if len(parts) != 4:
                    bad += 1; continue
                ctry[parts[3]] += 1
                if not parts[2].strip(): empty_addr += 1
                if not parts[1].strip(): empty_name += 1
        print(f"{split} S{s}: rows={n} bad_cols={bad} empty_name={empty_name} empty_addr={empty_addr} country={dict(ctry.most_common(6))}", flush=True)

gt = collections.Counter(); n = 0; tot = 0; s2 = 0; s3 = 0
with z.open(pre + "train/train_ground_truth.tsv") as f:
    t = io.TextIOWrapper(f, encoding="utf-8", newline="")
    next(t)
    for line in t:
        a, _, b = line.rstrip("\r\n").partition("\t")
        ids = [x for x in b.split(",") if x]
        n += 1; gt[min(len(ids), 10)] += 1; tot += len(ids)
        s2 += sum(i.startswith("S2-") for i in ids); s3 += sum(i.startswith("S3-") for i in ids)
print(f"GT rows={n} total_matches={tot} S2={s2} S3={s3} mean={tot/n:.3f}")
print("matches-per-S1 distribution (10 = 10+):", sorted(gt.items()))
