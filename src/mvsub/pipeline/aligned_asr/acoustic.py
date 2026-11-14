"""Measure short transient sound boundaries near an existing native anchor."""
import numpy as np


def locate_sound(wav,rate,anchor,left,right):
    left=max(left,anchor-.6,0);right=min(right,anchor+.6,len(wav)/rate)
    if right-left<.03:return None
    hop=max(1,round(rate*.01));clip=wav[round(left*rate):round(right*rate)]
    size=len(clip)//hop
    if size<3:return None
    energy=np.sqrt(np.mean(clip[:size*hop].reshape(size,hop)**2,axis=1))
    if not np.all(np.isfinite(energy)):return None
    peak=float(energy.max());floor=float(np.quantile(energy,.1))
    # Reject silence and steady backgrounds. Text is never used to force a
    # waveform event to exist, and the search cannot jump to another utterance.
    if peak<.001 or peak<max(floor*3,.001):return None
    active=energy>max(floor*2,peak*.18,.0005)
    spans=[];begin=None
    for i,value in enumerate(active):
        if value and begin is None:begin=i
        if begin is not None and (not value or i==len(active)-1):
            end=i if not value else i+1
            a,b=left+begin*hop/rate,left+end*hop/rate
            if b-a>=.03:spans.append((a,b))
            begin=None
    if not spans:return None
    a,b=min(spans,key=lambda span:max(span[0]-anchor,anchor-span[1],0))
    if max(a-anchor,anchor-b,0)>.35:return None
    return a,b
