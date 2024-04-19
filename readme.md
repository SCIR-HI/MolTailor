```
conda create -n moltailor python=3.9

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia


pip install ipykernel
pip install pandas
pip install rdkit
pip install icecream
pip install lightning
pip install pydantic
pip install transformers
pip install torch-geometric
pip install loguru
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install joblib
pip install pymatgen
pip install addict
pip install scikit-learn

```


conda create -n moltailor-lite python=3.9
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install ipykernel
pip install pandas
pip install rdkit
pip install icecream
pip install lightning
pip install pydantic
pip install transformers
pip install torch-geometric
pip install loguru
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

dgl: https://www.dgl.ai/pages/start.html