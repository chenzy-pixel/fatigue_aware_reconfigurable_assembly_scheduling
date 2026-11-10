"""Download and verify public Release assets listed in results/current/assets.json."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
from scripts.publish_results import OUT, sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--name',help='Download one asset by its exact name; default: all assets')
    a=p.parse_args();manifest=json.loads((OUT/'assets.json').read_bytes())
    selected=[x for x in manifest['assets'] if a.name is None or x['name']==a.name]
    if not selected:raise ValueError('asset name not in manifest')
    a.assets.mkdir(parents=True,exist_ok=True)
    for asset in selected:
        name=asset['name']
        if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('unsafe asset name')
        target=a.assets/name
        if target.exists() and target.stat().st_size==asset['bytes'] and sha(target)==asset['sha256']:
            print('verified',name,flush=True);continue
        url=f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}/{name}'
        temp=target.with_suffix(target.suffix+'.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as out:
                    while block:=response.read(1024**2):out.write(block)
                if temp.stat().st_size!=asset['bytes'] or sha(temp)!=asset['sha256']:raise ValueError('download checksum differs')
                temp.replace(target);break
            except (OSError,ValueError):
                if attempt==2:raise
                time.sleep(2)
        print('downloaded and verified',name,flush=True)

if __name__=='__main__':main()
