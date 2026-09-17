# -*- coding: utf-8 -*-
"""Small HCAM MLM smoke test / benchmark."""
import argparse
import time
from pathlib import Path

import torch

from hcam import HCAMTokenizer, load_hcam, masked_logits


def parse_positions(spec, text):
    if spec:
        out=[int(x.strip()) for x in spec.split(',') if x.strip()]
        if any(i<0 or i>=len(text) for i in out): raise ValueError('position out of text range')
        return out
    targets=[i for i,ch in enumerate(text) if ch in '的地得']
    return targets or [max(0,len(text)//2)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='hcam_100m_base_fp32.pt')
    p.add_argument('--tokenizer',default='tokenizer.model')
    p.add_argument('--text',default='温室里的植物正在安静地生长。')
    p.add_argument('--positions',default='',help='0-based character positions, comma-separated')
    p.add_argument('--topk',type=int,default=5)
    p.add_argument('--repeat',type=int,default=3)
    args=p.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=HCAMTokenizer(args.tokenizer)
    model,payload,result=load_hcam(args.model,device=device)
    positions=parse_positions(args.positions,args.text)
    ids=torch.tensor([tok.encode(args.text,add_special_tokens=True)],dtype=torch.long,device=device)
    pos=torch.tensor([[i+1 for i in positions]],dtype=torch.long,device=device)  # +1 for BOS

    for _ in range(2): _=masked_logits(model,ids,pos)
    if device=='cuda': torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(max(1,args.repeat)): logits=masked_logits(model,ids,pos)
    if device=='cuda': torch.cuda.synchronize()
    elapsed=(time.perf_counter()-t0)/max(1,args.repeat)

    probs=logits.float().softmax(-1)[0]
    print(f'device={device} | chars={len(args.text)} | masked={len(positions)} | avg={elapsed*1000:.2f} ms')
    for row,char_pos in zip(probs,positions):
        vals,idx=row.topk(args.topk)
        preds='  '.join(f'{tok.pieces[int(i)]}:{float(v):.4f}' for v,i in zip(vals,idx))
        print(f'pos {char_pos:>3} | original={args.text[char_pos]!r} | {preds}')


if __name__=='__main__': main()
