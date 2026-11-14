#!/usr/bin/env python3
"""Minimal prompt-only view of large CHStone sources (never use for HLS).

Based on MailoHLS evaluation_set/chstone-{aes,jpeg}/kernel.c: retain every
Lk-bearing function verbatim apart from comments and large global test data;
replace unrelated helper bodies by prototypes. Full source stays authoritative.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ACTION = re.compile(r'/\*\s*(L[1-9][0-9]*)\s*:\s*\*/',re.I)
FN = re.compile(r'([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$',re.S)
ARRAY = re.compile(r'(^|\n)([ \t]*(?:(?:static|extern|const|volatile|unsigned|signed)\s+)*(?:char|int|short|long|bit[0-9]+|uint[0-9]+_t|unsigned\s+char|unsigned\s+int)\s+([A-Za-z_]\w*)(?:\s*\[[^\]]+\])+\s*)=\s*\{',re.M)
KEEP_TOP = {'chstone_aes','aes_main','chstone_jpeg','jpeg2bmp_main','main'}

def hashstr(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def strip_comments(code):
    # CHStone source contains no comments inside string constants of concern here.
    # Keep action registration comments; all other comments are prompt noise.
    def block(m):return m.group(0) if ACTION.fullmatch(m.group(0)) else ''
    code=re.sub(r'/\*[\s\S]*?\*/',block,code)
    return re.sub(r'//[^\n]*','',code)

def functions(code):
    """Top-level brace matching; ignore quoted strings/chars."""
    found=[];depth=0;first=-1;quote=None;escaped=False
    for i,c in enumerate(code):
        if quote:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c==quote:quote=None
            continue
        if c in ('"',"'"):quote=c;continue
        if c=='{':
            if depth==0:
                pre=code[max(0,i-500):i]
                first=i if pre.rstrip().endswith(')') else -1
            depth+=1
        elif c=='}':
            depth-=1
            if depth<0:raise ValueError('Unbalanced braces')
            if depth==0 and first>=0:
                pre=code[max(0,first-500):first]
                m=FN.search(pre)
                if m:
                    name=m.group(1)
                    labels=set(ACTION.findall(code[first:i]))
                    found.append((name,first,i+1,labels))
    if depth:raise ValueError('Unbalanced braces at end')
    return found

def remove_large_global_initializers(code,minimum=500):
    replacements=[]
    for m in ARRAY.finditer(code):
        # Ignore an array inside a retained function.
        offset=m.start(2)
        if any(start<=offset<end for _,start,end,_ in functions(code)):
            continue
        start=m.end()-1;level=0;end=None
        for i in range(start,len(code)):
            if code[i]=='{':level+=1
            elif code[i]=='}':
                level-=1
                if level==0:
                    end=i+1;break
        if end is None or not re.match(r'\s*;',code[end:]):continue
        end+=re.match(r'\s*;',code[end:]).end()
        if end-m.start(2)<minimum:continue
        decl=m.group(2)
        decl=re.sub(r'\bstatic\s+','',decl)
        decl=re.sub(r'\bextern\s+','',decl)
        replacements.append((m.start(2),end,'extern '+decl.strip()+'; /* literal initializer omitted in prompt view */'))
    for a,b,value in reversed(replacements):code=code[:a]+value+code[b:]
    return code,[x[2].split(';')[0] for x in replacements]

def compact(src,kernel):
    if kernel not in ('chstone-aes','chstone-jpeg'):
        raise ValueError('Only the two inspected CHStone kernels are supported')
    markers=[x.upper() for x in ACTION.findall(src)]
    expected=['L%d'%i for i in range(1,7 if kernel=='chstone-aes' else 22)]
    if sorted(markers,key=lambda s:int(s[1:]))!=expected:
        raise ValueError('Canonical source action IDs differ from expected %s'%expected)
    code=strip_comments(src)
    removed=[]
    for name,start,end,labels in reversed(functions(code)):
        if labels or name in KEEP_TOP:continue
        removed.append(name)
        code=code[:start]+'; /* helper body omitted in prompt only */'+code[end:]
    code,removed_arrays=remove_large_global_initializers(code)
    code=re.sub(r'[ \t]+(?=\n)','',code)
    code=re.sub(r'\n\s*\n\s*\n+', '\n\n',code)
    code=code.strip()+'\n'
    if sorted((x.upper() for x in ACTION.findall(code)),key=lambda s:int(s[1:]))!=expected:
        raise RuntimeError('Compact view lost source actions')
    return '/* PROMPT-ONLY CHStone view. Not a compilable HLS source. */\n'+code,sorted(removed),removed_arrays

def main():
    p=argparse.ArgumentParser();p.add_argument('--kernel',required=True,choices=('chstone-aes','chstone-jpeg'))
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();s=a.source.read_text(encoding='utf-8');view,removed,arrays=compact(s,a.kernel)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(view,encoding='utf-8')
    record={'kernel':a.kernel,'full_source_sha256':hashstr(s),'view_sha256':hashstr(view),
        'original_chars':len(s),'view_chars':len(view),'omitted_helper_bodies':removed,
        'omitted_large_literal_arrays':arrays,'status':'PROMPT_VIEW_ONLY_NEVER_SYNTHESIZE'}
    (a.output.with_suffix(a.output.suffix+'.provenance.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
