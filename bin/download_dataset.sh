mkdir -p data/Mindbonggle_101
download-gdrive 1f-POOeLWok3KCRKinTXf_0VHqhafZtpD brain_data.zip
unzip brain_data.zip
mv dataverse_files/* data/Mindbonggle_101
rm brain_data.zip