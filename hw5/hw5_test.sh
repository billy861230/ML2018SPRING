mkdir model
cd model
mkdir action
cd action
wget https://www.dropbox.com/s/sr2pa788yjrae1v/model.h5
cd ../..
python3 hw5.py action test --load_model action --w2v w2v.model --token token.pk --result_path $2 --test_path $1