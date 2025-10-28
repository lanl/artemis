#!/usr/bin/env python3
import sys
from pathlib import Path
import difflib

def split_line(line):
    key = ''
    val = ''
    node = ''
    comment = ''

    #print(0, line)
    split = line.split('#',1)
    #print(1, split)
    line = split[0].strip()
    #print(2,line)
    if len(split) > 1:
        comment = split[1].strip()
        if len(comment) == 0:
            comment = ' '
    if len(line) == 0:
        return key, val, comment, False

    split = line.split('&',1)
    #print(3,split)
    line = split[0].strip()
    #print(4,line)
    linecont = len(split) > 1

    split = line.split('=',1)
    #print(5,split)
    #print(6,key)
    if len(split) > 1:
        key = split[0].strip()
        val = split[1].strip()
    else:
        val = split[0].strip()

    return key,val, comment, linecont

def output_block(block, debug=False):
    isnode = lambda x: '>' in x and '<' in x
    comm = '#'.rjust(4) + ' '
    max_key = max([len(_[0]) for  _ in block])
    max_val = max([len(_[1]) for  _ in block if not isnode(_[1])])
    max_val += 2*any([_[3] for _ in block])
    if debug:
        print(max_key, max_val)

    blines=[]
    for b in block:
        k,v,c,lc = b[0],b[1],b[2],b[3]
        res=''
        if len(k) == 0:
            if len(v) == 0:
                if len(c) == 0:
                    if lc:
                        # just &
                        res += ' &'.rjust(max_key + max_val + 3)
                    else:
                        # empty
                        pass
                else:
                    if lc:
                        # & + comment
                        res += ' &'.rjust(max_key + max_val + 3) + comm + c
                    else:
                        # pure comment
                        res += '# ' + c
            else:
                if lc:
                    res += ' '.rjust(max_key + 3) + v.ljust(max_val-2) + ' &'
                else:
                    if isnode(v):
                        # node
                        res = v
                        if len(c) > 0:
                            res += comm + c
                    else:
                        # end of a list
                        res += ' '.rjust(max_key + 3) + v.ljust(max_val)
        else:
            res = k.ljust(max_key)
            if len(k) > 0:
                res += ' = '
            else:
                res += '   '
            if len(v) > 0:
                res += v.ljust(max_val - 2*lc)
            if lc:
                res += ' &'
            if len(c) > 0:
                res += comm + c
        blines.append(res)
    return '\n'.join(blines)


def format_file(fname):
    with open(fname,'r') as f:
        lines = f.read()
    flines=''
    block=[]
    for line in lines.split('\n'):
        l = split_line(line)
        if len(l[0]) == 0 and len(l[1]) > 0:
            if '<' in l[1] and '>' in l[1]:
                # a node
                if len(block) > 0:
                    flines += output_block(block) + '\n'
                block = [l]
            else:
                block.append(l)
        else:
            block.append(l)
    flines += output_block(block) 

    return lines, flines


if __name__ == '__main__':
    inplace = sys.argv[1] == '-i'
    if inplace:
        files = sys.argv[2:]
    else:
        files = sys.argv[1:]
    for fname in files:
        orig, formatted = format_file(fname)
        if inplace:
            with open(fname,'w') as f:
                f.write(formatted)
        else:
            diff_lines = list(difflib.unified_diff(
                orig.splitlines(keepends=True),
                formatted.splitlines(keepends=True),
                lineterm='\n'
            ))
            print(''.join(diff_lines))



