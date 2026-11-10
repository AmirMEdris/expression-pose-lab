"""Generate deterministic predictions to compare Keras runtime compatibility."""
import argparse
from pathlib import Path
import time
import numpy as np
from backend import select_backend,device_info,predict_numpy
select_backend()
from expression import ExpressionEstimator


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model=ExpressionEstimator(Path(__file__).parent/'models/expression-baseline.h5').model
    batch=np.random.default_rng(2026).random((8,48,48,1),dtype=np.float32)
    predictions=predict_numpy(model,batch)
    for _ in range(3):predict_numpy(model,batch)
    start=time.monotonic()
    for _ in range(20):predict_numpy(model,batch)
    duration=(time.monotonic()-start)/20
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,probabilities=predictions,milliseconds=duration*1000)
    print(device_info(),f'{duration*1000:.2f} ms / 8-face batch',flush=True)


if __name__=='__main__':main()
