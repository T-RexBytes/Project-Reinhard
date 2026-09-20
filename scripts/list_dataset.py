import os
os.environ["KAGGLE_API_TOKEN"] = os.environ.get("KAGGLE_API_TOKEN", "")

from kaggle import api

all_files = []
page_token = None
page = 0

while True:
    page += 1
    resp = api.dataset_list_files("trexbytes/rfiqwav", page_token=page_token, page_size=1000)
    files = resp.files if hasattr(resp, "files") else resp
    all_files.extend(files)
    print(f"Page {page}: +{len(files)} (total {len(all_files)})", flush=True)
    page_token = resp.next_page_token if hasattr(resp, "next_page_token") else None
    if not page_token:
        break

print(f"\nTOTAL: {len(all_files)} files")

from collections import defaultdict
l3 = defaultdict(lambda: {"data": 0, "meta": 0, "size": 0})
total = 0
lines = []
for f in all_files:
    name = f.name
    size = f.size or 0
    total += size
    lines.append(f"{name} {size}")
    parts = name.split("/")
    if len(parts) >= 3:
        key = "/".join(parts[:3])
        ext = name.rsplit(".", 1)[-1] if "." in name else "none"
        if ext == "sigmf-data":
            l3[key]["data"] += 1
        elif ext == "sigmf-meta":
            l3[key]["meta"] += 1
        l3[key]["size"] += size

print(f"TOTAL SIZE: {round(total/1e9, 2)} GB")
print("\n=== dataset/<split>/<signal> ===")
for k in sorted(l3.keys()):
    v = l3[k]
    print(f"  {k}: {v['data']} data + {v['meta']} meta = {round(v['size']/1e6,1)} MB")

with open(r"D:\sih\SIH26147\outputs\dataset_file_list.txt", "w") as fp:
    fp.write("\n".join(lines))
print("\nSaved to outputs/dataset_file_list.txt")
