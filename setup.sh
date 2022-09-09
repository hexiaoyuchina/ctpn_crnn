
#conda create --name OCR python=3.6
#source activate OCR
pip install lmdb -i https://mirror.baidu.com/pypi/simple
pip install opencv-python==4.6.0.66 -i https://mirror.baidu.com/pypi/simple
pip install  matplotlib==3.3.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install  numpy==1.19.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install  tensorflow-gpu==1.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install  Cython==0.24.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ipython==5.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#cd ./ctpn/lib/utils
#sh make.sh
