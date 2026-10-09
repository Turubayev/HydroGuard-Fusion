"""Optional WNTR hydraulic scenario, with an explicitly synthetic truth label.

Generates pressure/head in metres and pipe flow in m³/s. Acoustic data are absent.
WNTR is used directly; no third-party source code or network data are copied.
"""
from pathlib import Path
import argparse
import json
import numpy as np

def simulate(duration_hours=48, leak_start_hours=24, leak_area=.0003):
    import wntr
    if not 0 < leak_start_hours < duration_hours or leak_area<=0: raise ValueError('Invalid simulation scenario')
    wn=wntr.network.WaterNetworkModel()
    wn.add_reservoir('R',base_head=60.)
    wn.add_junction('J',base_demand=.01,elevation=0.)
    wn.add_pipe('P','R','J',length=1000.,diameter=.2,roughness=100.)
    wn.options.time.duration=int(duration_hours*3600)
    wn.options.time.hydraulic_timestep=300;wn.options.time.report_timestep=300
    wn.options.hydraulic.demand_model='PDD'
    wn.get_node('J').add_leak(wn,area=leak_area,start_time=int(leak_start_hours*3600))
    result=wntr.sim.WNTRSimulator(wn).run_sim(convergence_error=True)
    frame=result.node['pressure'][['J']].rename(columns={'J':'pressure'})
    frame['flow']=result.link['flowrate']['P']
    frame['acoustic']=np.nan
    frame['leak_truth']=frame.index>=leak_start_hours*3600
    frame.index.name='seconds'
    return frame

def main():
    from hydroguard import FusionDetector
    p=argparse.ArgumentParser(description='Synthetic WNTR leak scenario and fixed baseline evaluation')
    p.add_argument('--out',type=Path,default=Path('runs/wntr_demo'));a=p.parse_args()
    frame=simulate()
    # First half is healthy calibration; evaluation excludes calibration rows.
    train=frame.index<12*3600
    detector=FusionDetector.fit(frame.loc[train,['pressure','flow','acoustic']].to_numpy())
    alarms=np.array([detector.update(x)['alarm'] for x in frame.loc[~train,['pressure','flow','acoustic']].to_numpy()])
    truth=frame.loc[~train,'leak_truth'].to_numpy()
    index=frame.index[~train].to_numpy()
    hits=index[alarms&truth]
    report={'data_kind':'synthetic_WNTR_single_network','wntr_version':__import__('wntr').__version__,
            'acoustic_available':False,'false_alarm_steps':int((alarms&~truth).sum()),'detected':bool(len(hits)),
            'detection_delay_seconds':int(hits[0]-24*3600) if len(hits) else None,
            'operational_validation':False,'note':'Constant demand toy network; does not prove real-network leak detection.'}
    a.out.mkdir(parents=True,exist_ok=True);frame.to_csv(a.out/'scenario.csv')
    detector.save(a.out/'model.json');(a.out/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
