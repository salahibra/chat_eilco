import json

with open("final.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

    for d in dataset:
        source_chunk = d.get("source_chunks")
        if not source_chunk:
            continue
        metadata = source_chunk[0].pop("metadata",None)
        if not metadata:
            continue
        og = metadata.get("original_content")
        metadata["type"] = "table" if og else "text"
        source_chunk[0].update(metadata)

with open("final_norm.json","w", encoding="utf-8") as f:
    json.dump(dataset,f,indent=4, ensure_ascii=True)