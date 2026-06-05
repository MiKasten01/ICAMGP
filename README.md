环境创建流程
```
conda create -n cam python=3.12 numpy=1.26 pandas scikit-learn matplotlib scipy seaborn statsmodels
conda activate cam
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas-plink hdbscan
```

Dataset的数据源
https://github.com/FelixHeinrich/GP_with_IFS/.