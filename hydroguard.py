"""Pressure/flow/acoustic anomaly baseline with frozen healthy calibration."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

CHANNELS = ('pressure', 'flow', 'acoustic')
DIRECTION = np.array([-1., 1., 1.])

class FusionDetector:
    def __init__(self, median, scale, threshold=8., drift=.5, min_channels=2, calibrated=None):
        self.median, self.scale = np.asarray(median, float), np.asarray(scale, float)
        if self.median.shape != (3,) or self.scale.shape != (3,) or not np.isfinite(self.median).all() or not (np.isfinite(self.scale) & (self.scale > 0)).all():
            raise ValueError('Invalid calibration')
        if threshold <= 0 or drift < 0 or min_channels not in (1, 2, 3):
            raise ValueError('Invalid detector settings')
        self.threshold, self.drift, self.min_channels = threshold, drift, min_channels
        self.calibrated = np.ones(3,bool) if calibrated is None else np.asarray(calibrated,bool)
        if self.calibrated.shape!=(3,) or self.calibrated.sum()<min_channels: raise ValueError('Too few calibrated sensors')
        self.state = 0.

    @classmethod
    def fit(cls, healthy, **options):
        x = np.asarray(healthy, float)
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 20 or np.isinf(x).any():
            raise ValueError('Need at least 20 healthy rows; never include incident/test rows')
        calibrated = np.isfinite(x).sum(axis=0)>=20
        if calibrated.sum()<options.get('min_channels',2): raise ValueError('Need >=20 healthy readings in at least 2 channels')
        med = np.zeros(3);scale=np.ones(3)
        # MAD is robust to isolated outliers; explicit floor handles constant sensors.
        med[calibrated] = np.nanmedian(x[:,calibrated],axis=0)
        scale[calibrated] = np.maximum(1.4826 * np.nanmedian(np.abs(x[:,calibrated] - med[calibrated]), axis=0), np.maximum(np.abs(med[calibrated]) * .001, 1e-6))
        return cls(med, scale, calibrated=calibrated, **options)

    def update(self, row):
        x = np.asarray(row, float)
        if x.shape != (3,): raise ValueError('Expected pressure, flow, acoustic')
        valid = np.isfinite(x) & self.calibrated
        if valid.sum() < self.min_channels:
            self.state = 0.  # gap breaks a contiguous event
            return {'status':'insufficient_data', 'alarm':False, 'score':None, 'cusum':0., 'channels':int(valid.sum())}
        z = (x - self.median) / self.scale * DIRECTION
        # A one-sensor spike cannot drive an alarm when >=2 channels are required.
        directional = np.maximum(0, z[valid])
        score = float(np.sort(directional)[-self.min_channels])
        self.state = max(0., self.state + min(score, 20.) - self.drift)
        return {'status':'anomaly' if self.state >= self.threshold else 'normal', 'alarm':self.state >= self.threshold,
                'score':round(score, 5), 'cusum':round(self.state, 5), 'channels':int(valid.sum()),
                'channel_z':{k:round(float(v), 5) if ok else None for k,v,ok in zip(CHANNELS,z,valid)}}

    def save(self, path):
        Path(path).write_text(json.dumps({'schema':1, 'median':self.median.tolist(), 'scale':self.scale.tolist(),
            'threshold':self.threshold, 'drift':self.drift, 'min_channels':self.min_channels,'calibrated':self.calibrated.tolist()}), encoding='utf-8')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if d.pop('schema', None) != 1: raise ValueError('Unsupported model schema')
        return cls(**d)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not all(k in rows[0] for k in ('time',)+CHANNELS): raise ValueError('CSV needs time,pressure,flow,acoustic')
    from datetime import datetime
    times = [datetime.fromisoformat(r['time'].replace('Z','+00:00')) for r in rows]
    if any(b <= a for a,b in zip(times,times[1:])): raise ValueError('Times must increase without duplicates')
    x = np.array([[float(r[k]) if r[k].strip() else np.nan for k in CHANNELS] for r in rows])
    return rows, x

def main():
    p = argparse.ArgumentParser(description='Local telemetry anomaly detector; alarms require operator review')
    sub = p.add_subparsers(dest='cmd', required=True)
    fit = sub.add_parser('fit'); fit.add_argument('healthy'); fit.add_argument('--model', required=True)
    detect = sub.add_parser('detect'); detect.add_argument('telemetry'); detect.add_argument('--model', required=True)
    a = p.parse_args()
    try:
        if a.cmd == 'fit':
            _, x = read_csv(a.healthy); FusionDetector.fit(x).save(a.model)
            print('Calibration saved. Tune threshold on a separate validation set.')
        else:
            rows,x = read_csv(a.telemetry); model = FusionDetector.load(a.model)
            for row,values in zip(rows,x): print(json.dumps({'time':row['time'], **model.update(values), 'operator_review_required':True}))
    except (OSError, ValueError, TypeError) as e: p.error(str(e))

if __name__ == '__main__': main()
