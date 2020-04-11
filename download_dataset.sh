download-gdrive 1fA7hH7xKnt1Y9g9ExlUkb9seDpnC8o37 brain_data.zip
mkdir ../data
mkdir ../data/dataverse_files
unzip brain_data.zip &>/dev/null
mv brain_data/* ../data/dataverse_files
rm brain_data.zip
