import sys

from pathlib import Path, PosixPath

base_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(base_path))
model_path = base_path / 'models/CLAMP'

try:
    from clamp import PretrainedCLAMP
except:
    from .clamp import PretrainedCLAMP



def load_clamp(path: PosixPath = model_path):
    model = PretrainedCLAMP(path_dir=path, device='cuda')

    return model

