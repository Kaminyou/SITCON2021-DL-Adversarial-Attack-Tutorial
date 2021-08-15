wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -zxvf imagenette2.tgz
mkdir data
mv ./imagenette2 ./data/
rm imagenette2.tgz