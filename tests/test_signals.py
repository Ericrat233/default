import numpy as np
import pytest
from openpod.signals import cfar,cfar_nd,shift_zero,process_traces,process_field,TraceConfig,calibrate,snr_db,db

@pytest.mark.parametrize('method',['ca','os'])
def test_cfar_empirical_false_alarm(method):
    rng=np.random.default_rng(1823); power=rng.exponential(size=18000); c=cfar(power,16,4,.01,method)
    rate=c['detected'][c['valid']].mean(); assert .0065<rate<.014
    assert not c['valid'][:20].any()

def test_zero_padding_alignment():
    t=np.arange(256); ref=np.sin(.5*t)*np.exp(-.5*((t-140)/10)**2); delayed=shift_zero(ref,7)
    r=process_traces(delayed,TraceConfig(baseline_samples=40,sg_window=0,align=True,gate=(80,210)),reference=ref)
    assert r['shifts_samples']==-7
    np.testing.assert_allclose(r['processed'],ref,atol=1e-9)
    assert shift_zero(np.array([1,2,3]),1).tolist()==[0,1,2]

def test_nd_shape_and_field():
    rng=np.random.default_rng(5); traces=rng.normal(size=(3,4,128))
    r=process_traces(traces,TraceConfig(axis=2)); assert r['processed'].shape==traces.shape
    field=rng.normal(size=(20,20,12)); p=process_field(field); assert p['processed'].shape==field.shape
    c=cfar_nd(np.ones((20,20)),3,1); assert np.isnan(c['threshold'][:4]).all()

def test_snr_db_calibration():
    assert snr_db(np.ones(10)*2,np.ones(10))==pytest.approx(6.020599913)
    assert db(np.array([10.]))[0]==20
    x=np.arange(6.); p=calibrate(x,2*x+1); np.testing.assert_allclose(p['coefficients'],[2,1],atol=1e-12)

def test_preprocess_rejects_invalid():
    with pytest.raises(ValueError): process_traces(np.ones(100),TraceConfig(sg_window=8))
    with pytest.raises(ValueError): cfar(np.array([-1.,2]))
