"""Reassemble lossless file chunks and verify their SHA256 hashes offline."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
for entry in json.loads((ROOT/'large-files/manifest.json').read_text()):
 target=ROOT/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']:continue
 temporary=target.with_name(target.name+'.reassembling')
 with temporary.open('wb') as out:
  for name in entry['parts']:out.write((ROOT/name).read_bytes())
 if hashlib.sha256(temporary.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Archive integrity failure: '+entry['path'])
 temporary.replace(target);print('Restored',entry['path'])
