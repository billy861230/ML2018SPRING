wget -P /model/action https://www.dropbox.com/s/sr2pa788yjrae1v/model.h5
python3 hw5.py action test --load_model action --w2v w2v.model --token token.pk --result_path $2 --test_path $1