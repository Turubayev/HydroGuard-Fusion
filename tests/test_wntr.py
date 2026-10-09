import numpy as np
import pytest
from hydroguard import FusionDetector

def test_uncalibrated_channel_cannot_raise_alarm():
    x=np.tile([50.,10.,np.nan],(30,1));d=FusionDetector.fit(x)
    assert not d.calibrated[2]
    assert not d.update([50.,10.,1000.])['alarm']

def test_real_wntr_adapter():
    pytest.importorskip('wntr')
    from wntr_adapter import simulate
    f=simulate(duration_hours=4,leak_start_hours=2)
    assert np.isfinite(f[['pressure','flow']]).all().all()
    before=f.loc[f.index<7200];after=f.loc[f.index>=7200]
    assert after['pressure'].mean()<before['pressure'].mean()
    assert after['flow'].mean()>before['flow'].mean()
    assert f['acoustic'].isna().all()
