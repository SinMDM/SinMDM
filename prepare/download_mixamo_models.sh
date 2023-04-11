echo "Downloading Mixamo pretrained models"
mkdir -p save
cd save
gdown 1UHP7uNWkSdsmDSV6fbtmJ1nXn6vtr6bY

unzip mixamo.zip
echo -e "Cleaning\n"
rm mixamo.zip

echo "Downloading done!"