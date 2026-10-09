import numpy as np
import pytest
from hydroguard import FusionDetector, read_csv

def trained():
    rng = np.random.default_rng(42)
    return FusionDetector.fit(rng.normal([50,10,.5],[.3,.1,.02],(200,3)))

def test_event_and_healthy():
    d = trained()
    assert not any(d.update([50,10,.5])['alarm'] for _ in range(100))
    assert any(d.update([47,12,.8])['alarm'] for _ in range(5))

def test_one_sensor_spike_is_not_fusion_event():
    d = trained()
    assert not any(d.update([1,10,.5])['alarm'] for _ in range(10))

def test_missing_breaks_event():
    d = trained(); d.update([47,12,.8])
    assert d.update([np.nan,np.nan,.5])['status'] == 'insufficient_data'
    assert d.state == 0

def test_save_roundtrip(tmp_path):
    d = trained(); p = tmp_path/'model.json'; d.save(p)
    assert d.update([47,12,.8]) == FusionDetector.load(p).update([47,12,.8])

def test_constant_sensor_and_invalid_train():
    assert np.isfinite(FusionDetector.fit(np.ones((20,3))).scale).all()
    with pytest.raises(ValueError): FusionDetector.fit(np.ones((19,3)))
    with pytest.raises(ValueError): FusionDetector.fit(np.full((20,3),np.nan))

def test_nonchronological_csv(tmp_path):
    p=tmp_path/'bad.csv';p.write_text('time,pressure,flow,acoustic\n2026-01-02,1,2,3\n2026-01-01,1,2,3')
    with pytest.raises(ValueError): read_csv(p)
