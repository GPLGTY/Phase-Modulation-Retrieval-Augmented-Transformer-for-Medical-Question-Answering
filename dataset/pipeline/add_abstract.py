import json
import os
from config import PROJ_TOP_DIR


if __name__ == "__main__":
    json_dir = os.path.join(PROJ_TOP_DIR, "docs", "json")
    abstract_dir = os.path.join(PROJ_TOP_DIR, "docs", "abstract")
    json_names = os.listdir(json_dir)

    for j in json_names:
        json_path = os.path.join(json_dir, j)
        abstract_path = os.path.join(abstract_dir, j)

        if not os.path.exists(abstract_path):
            # raise KeyError("Abstract file not found")
            abstract = ""

        else:
            with open(abstract_path, "r") as f:
                data = json.load(fp=f)
                abstract = data["chatgpt_abstract"]

        with open(json_path, "r+") as f:
            data = json.load(fp=f)
            data.update({"abstract": abstract})
            f.seek(0)
            json.dump(data, fp=f, indent=4, ensure_ascii=False)
