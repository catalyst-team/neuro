download-gdrive 1cD0euGOhuOm8w-XXa7b4wQeZ_RfkmxN4 brain_data.zip
mkdir -p data/Mindboggle_101
unzip brain_data.zip &>/dev/null
find . -name '*volumes*' | xargs -I '{}' mv {} data/Mindboggle_101/
find data/Mindboggle_101 -name '*.tar.gz'| xargs -i tar zxvf {} -C data/Mindboggle_101
find data/Mindboggle_101 -name '*.tar.gz'| xargs -i rm {}
rm brain_data.zip
