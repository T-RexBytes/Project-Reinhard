import os, urllib.request, urllib.error, json

token = os.environ.get("KAGGLE_API_TOKEN", "")
base = "https://www.kaggle.com/api/v1/datasets/list/trexbytes/rfiqwav"

# 1) Try larger page size first
for ps in [100, 500, 1000]:
    try:
        url = base + f"?pageSize={ps}"
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
        r = urllib.request.urlopen(req, timeout=30)
        data = json.loads(r.read())
        files = data.get("datasetFiles", [])
        print(f"pageSize={ps}: got {len(files)} files, has_next={data.get('nextPageTokenNullable') is not None}")
        if len(files) > 20:
            break
    except Exception as e:
        print(f"pageSize={ps} ERROR: {e}")
