rm -r data
mkdir data
rm -r badpoints
mkdir badpoints
rm -r w_est
mkdir w_est
rm -r ../model
mkdir ../model

cd ../
rm -r venv
mkdir venv
sudo pip install virtualenv
virtualenv venv/openapi
source ./venv/openapi/bin/activate

pip install numpy==1.15.4
pip install scipy==1.2.0
pip install torch==1.1.0
pip install torchvision==0.2.1
pip install sklearn==0.20.0
pip install matplotlib
pip install tqdm
pip install lime==0.1.1.32
sh install.sh

cd -
