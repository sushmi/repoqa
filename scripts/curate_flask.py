# scripts/curate_flask.py
import json
d = json.load(open('data/evaluation/dataset.json'))
flask = [q for q in d['qa_pairs']
         if q['repo_name'] == 'pallets/flask'
         and q.get('answer_score', 0) >= 5
         and len(q['question']) >= 80
         and 'https://' not in q['question'][:30]]
out = {**{k: d[k] for k in ('name', 'version', 'description', 'metadata')},
       'qa_pairs': flask}
json.dump(out, open('data/evaluation/flask_curated.json', 'w'), indent=2)
print(f"Wrote {len(flask)} curated Flask questions")
