echo "Downloading HumanML3D pretrained models"
mkdir -p save
cd save
gdown 1iemoZqbF7IXOJtsrRtegvhd_eK8DwzrW

unzip humanml.zip
echo -e "Cleaning\n"
rm humanml.zip

echo "Downloading done!"