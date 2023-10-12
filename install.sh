# Go one level up
cd ..

# Download Isaac Gym
curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=1J4bb5SfY-8H05xXiyF4N1xUOas390tll" > /dev/null
curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=$(awk '/confirm/ {print $NF}' ./cookie.txt)&id=1J4bb5SfY-8H05xXiyF4N1xUOas390tll" -o isaacgym.tar.gz

# Extract Isaac Gym
tar -xzf isaacgym.tar.gz

# Download Furniture-benchmark
git clone git@github.com:ankile/furniture-bench.git

# Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create conda environment
conda create -n rlgpu python=3.8 -y
pip install --upgrade pip wheel
pip install setuptools==58
pip install --upgrade pip==22.2.2

# Activate conda environment
conda activate rlgpu

# Install dependencies
cd isaacgym
pip install -e python

cd ../furniture-bench
pip install -e .
# pip install -e r3m
# pip install -e vip

# Install AWS CLI
cd ~
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
# Do this if you have sudo access
# sudo ./aws/install

# Otherwise we can add the following to .bashrc to emulate the above
# export AWS_COMMAND='/home/larsankile/aws-cli/v2/current/bin/aws'
# alias aws='/home/larsankile/aws-cli/v2/current/bin/aws'

# Install the last required dependencies
cd furniture-diffusion
pip install -r requirements.txt